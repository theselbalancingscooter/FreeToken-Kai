"""Routed experts of Qwen3.8-Flash-Next read from a GGUF, as host banks for the offload cache.

``--moe-expert-gguf <file>`` points at a llama.cpp checkpoint of the same model (shard 1 of
ISTA-DASLab's ``Qwen3.8-Flash-Next-GSQ-RCO-GGUF`` Q2_0, say). Only ``blk.N.ffn_{gate,up,down}_exps``
are taken from it: the dense projections, the PLE table and the vision tower keep coming from the
safetensors checkpoint named by ``--model``. The banks hold the GGUF's block bytes verbatim --
``gate_up`` ``[E, 2I, row_bytes(H)]`` (gate rows then up rows) and ``down`` ``[E, H, row_bytes(I)]``
-- and the ggml MoE kernels (GPU) / the cpu_moe W*A8 GEMVs dequantize inside the K-loop, exactly
as the gemma4 GGUF path does for Q4_0. ``ModelConfig.expert_quant`` / ``moe_weight_format`` carry
the format tag (``q2_0`` or ``q4_0``) that the cache, the kernels and the CPU executor dispatch on.

The experts of every layer must share one ggml type: the expert slot cache is one allocation.
The "UD" community quantisations vary the type by layer and are refused here (docs/gguf.md).
"""

from __future__ import annotations

import re
from functools import partial

import torch

from freetoken.models.gguf.dequant import GGML_Q2_0, GGML_Q4_0, GGML_NAME, row_bytes
from freetoken.models.gguf.rawreader import RawGguf

GGML_TO_FORMAT = {GGML_Q2_0: "q2_0", GGML_Q4_0: "q4_0"}
FORMAT_TO_GGML = {fmt: t for t, fmt in GGML_TO_FORMAT.items()}
_EXPERT_RE = re.compile(r"^blk\.(?P<layer>\d+)\.ffn_(?P<proj>gate|up|down)_exps\.weight$")


def gguf_expert_format(gguf: "RawGguf | str") -> str:
    """The bank format tag of the GGUF's routed experts, or a ValueError naming what was found."""
    g = RawGguf(gguf) if isinstance(gguf, str) else gguf
    types: dict[int, list[str]] = {}
    for t in g:
        if _EXPERT_RE.match(t.name):
            types.setdefault(t.ggml_type, []).append(t.name)
    if not types:
        raise ValueError(f"{g.path}: no blk.N.ffn_{{gate,up,down}}_exps.weight tensors (not a MoE GGUF?)")
    if len(types) != 1:
        found = ", ".join(f"{GGML_NAME.get(k, k)} x{len(v)}" for k, v in sorted(types.items()))
        raise ValueError(
            f"{g.path}: the routed experts mix ggml types ({found}); the expert cache holds one "
            "layout, so a uniform quantisation is needed (docs/gguf.md)"
        )
    (ggml_type,) = types
    fmt = GGML_TO_FORMAT.get(ggml_type)
    if fmt is None:
        raise ValueError(
            f"{g.path}: routed experts are {GGML_NAME.get(ggml_type, ggml_type)}; the bank kernels serve "
            f"{sorted(FORMAT_TO_GGML)} only"
        )
    return fmt


def _specs(config, fmt: str) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    t = FORMAT_TO_GGML[fmt]
    E, H, I = config.num_experts, config.hidden_size, config.moe_intermediate_size
    return {
        "gate_up": ((E, 2 * I, row_bytes(H, t)), torch.uint8),
        "down": ((E, H, row_bytes(I, t)), torch.uint8),
    }


def _layer_window(config) -> tuple[int, int]:
    """Global MoE layer range this process serves: the pipeline rank's bank window, else all."""
    from freetoken.distributed import try_get_pp_info

    first_dense = int(getattr(config, "first_k_dense_replace", 0) or 0)
    pp = try_get_pp_info()
    if pp is not None:
        return pp.bank_window(first_dense)
    return 0, int(config.num_moe_layers)


def load_gguf_expert_sources(
    model_path: str, config, *, layer_sink=None, quant_format: str | None = None
) -> dict[str, list[torch.Tensor]]:
    """Per-layer host banks of the routed experts' packed GGUF rows (see the module doc).

    ``config.moe_expert_gguf`` names the file; ``quant_format`` (when given) must match what the
    file holds. ``layer_sink=None`` (serving) pins each layer as it completes; a sink (converter)
    receives the layers instead and nothing is pinned here.
    """
    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline, alloc_layer_banks

    path = getattr(config, "moe_expert_gguf", None)
    if not path:
        raise ValueError("GGUF expert banks need --moe-expert-gguf <file> (ModelConfig.moe_expert_gguf)")
    g = RawGguf(path)
    fmt = gguf_expert_format(g)
    if quant_format is not None and fmt != quant_format:
        raise ValueError(f"{path}: experts are {fmt}, the model was configured for {quant_format}")
    E, H, I = config.num_experts, config.hidden_size, config.moe_intermediate_size
    t = FORMAT_TO_GGML[fmt]
    rb_h, rb_i = row_bytes(H, t), row_bytes(I, t)
    lo, hi = _layer_window(config)
    L = hi - lo
    if L != int(config.num_moe_layers):
        raise ValueError(f"layer window [{lo}, {hi}) holds {L} layers but the config serves {config.num_moe_layers}")
    hb = alloc_layer_banks(_specs(config, fmt), L)  # lazy anon mmaps (unpinned)
    banks = {name: [b.tensor for b in hb[name]] for name in hb}
    seen: dict[str, set[int]] = {"gate": set(), "up": set(), "down": set()}

    def _load(sink) -> None:
        tracker = LayerCompletionTracker(3, hb, sink) if sink is not None else None  # gate + up + down
        for tensor in g:
            m = _EXPERT_RE.match(tensor.name)
            if m is None:
                continue
            layer = int(m.group("layer"))
            if not (lo <= layer < hi):
                continue
            local = layer - lo
            proj = m.group("proj")
            if tensor.shape[0] != E:
                raise ValueError(f"{tensor.name}: {tensor.shape[0]} experts in the GGUF, {E} in the model")
            src = g.packed(tensor.name)
            if proj == "down":
                if tuple(tensor.shape) != (E, H, I):
                    raise ValueError(f"{tensor.name}: shape {tensor.shape}, expected {(E, H, I)}")
                banks["down"][local].copy_(src.reshape(E, H, rb_i))
            else:
                if tuple(tensor.shape) != (E, I, H):
                    raise ValueError(f"{tensor.name}: shape {tensor.shape}, expected {(E, I, H)}")
                half = slice(0, I) if proj == "gate" else slice(I, 2 * I)
                banks["gate_up"][local][:, half, :].copy_(src.reshape(E, I, rb_h))
            seen[proj].add(local)
            if tracker is not None:
                tracker.note(local)

    if layer_sink is not None:
        _load(layer_sink)
    elif torch.cuda.is_available():
        with PinPipeline() as pins:
            _load(pins)
    else:
        _load(None)  # CUDA-less: mmap banks stay pageable, never pinned

    want = set(range(L))
    missing = {k: sorted(want - v) for k, v in seen.items() if want - v}
    if missing:
        raise ValueError(f"{path}: expert layers missing for this rank's window [{lo}, {hi}): {missing}")
    return banks


def dummy_gguf_expert_sources(config, *, quant_format: str) -> dict[str, list[torch.Tensor]]:
    """Random but finite banks shaped like ``load_gguf_expert_sources`` output (--use-dummy-weight)."""
    from freetoken.moe.host_banks import alloc_layer_banks, pin_banks

    L = int(config.num_moe_layers)
    hb = alloc_layer_banks(_specs(config, quant_format), L)
    banks = {name: [b.tensor for b in hb[name]] for name in hb}
    for tensor in banks["gate_up"] + banks["down"]:
        tensor.random_(0, 256)
        # every block starts with an fp16 scale: keep it finite and small (0x2C00 = 0.0625)
        blocks = tensor.view(-1, 18)
        blocks[:, 0] = 0x00
        blocks[:, 1] = 0x2C
    if torch.cuda.is_available():
        pin_banks(hb)
    return banks


load_q2_0_expert_sources = partial(load_gguf_expert_sources, quant_format="q2_0")
load_q4_0_expert_sources = partial(load_gguf_expert_sources, quant_format="q4_0")
dummy_q2_0_expert_sources = partial(dummy_gguf_expert_sources, quant_format="q2_0")
dummy_q4_0_expert_sources = partial(dummy_gguf_expert_sources, quant_format="q4_0")

__all__ = [
    "GGML_TO_FORMAT",
    "FORMAT_TO_GGML",
    "gguf_expert_format",
    "load_gguf_expert_sources",
    "dummy_gguf_expert_sources",
    "load_q2_0_expert_sources",
    "load_q4_0_expert_sources",
    "dummy_q2_0_expert_sources",
    "dummy_q4_0_expert_sources",
]
