"""Grouped expert GEMM over native GGUF block banks (borrowed ggml MoE kernels).

Ports vLLM/sglang's ``_fused_moe_gguf`` onto FreeToken's offload-cache interface: the experts
are streamed to the GPU as packed block bytes and dequantized *inside* the kernels -- no bf16
expert copy is materialized. Two kernels:

* **MMVQ** (``ggml_moe_a8_vec``): one GEMV per (token, expert); consumes ``topk_ids`` directly.
  Right for a single decoded token, where every expert is read once anyway.
* **MMQ** (``ggml_moe_a8``): tiled, the tokens sorted by expert (``moe_align_block_size``), so
  an expert's rows are read once per tile of ``MOE_X`` tokens instead of once per token. This is
  what a prefill chunk and a speculative verify window need: with MMVQ a 6-row window read every
  expert six times. Available for ``q4_0`` and ``q2_0``; ``FREETOKEN_GGUF_MMQ_MIN_ROWS`` (default 2)
  is the row count from which MMQ is used, 0 keeps MMVQ everywhere.

``topk_ids`` index the streamed cache slots (decode) or the materialized layer positions
(prefill); the expert dimension of the banks is therefore a slot cache of a few thousand rows,
which the vendored kernel's ``<= 256 experts`` guard was lifted for.
"""

from __future__ import annotations

import os

import torch

from freetoken.layers.activation import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul
from freetoken.models.gguf.dequant import GGML_Q2_0, GGML_Q4_0

_ACT = {"silu": silu_and_mul, "gelu": gelu_and_mul, "gelu_tanh": gelu_tanh_and_mul}

# bank format tag (OffloadMoeCache.quant_format) -> ggml type the kernels dispatch on
GGUF_FORMAT_TYPES = {"q4_0": GGML_Q4_0, "q2_0": GGML_Q2_0}
# the expert bank formats that carry no quant method: the cache's format tag dispatches
GGUF_EXPERT_FORMATS = frozenset(GGUF_FORMAT_TYPES)


def mmq_min_rows() -> int:
    """Rows (tokens x top_k) from which the tiled MMQ kernel replaces the per-row MMVQ; 0 = never."""
    return int(os.environ.get("FREETOKEN_GGUF_MMQ_MIN_ROWS", "2") or 0)


def _mmq(hidden_states, gate_up_q, down_q, topk_ids, act_fn, qt, num_tokens, top_k, n2, h):
    from freetoken.kernel.gguf import ggml_moe_a8, ggml_moe_get_block_size
    from freetoken.moe.fused import moe_align_block_size

    num_slots = int(gate_up_q.shape[0])
    block = int(ggml_moe_get_block_size(qt))
    sorted_ids, expert_ids, ntpp = moe_align_block_size(topk_ids, block, num_slots)
    # the aligner labels its all-padding blocks with the sentinel expert ``num_slots``; the
    # kernel skips negative ids (graph-safe: no host sync, fixed shapes)
    expert_ids = torch.where(expert_ids >= num_slots, expert_ids.new_full((), -1), expert_ids)
    # gate_up: the sorted flat ids / top_k address the tokens
    gate_up = ggml_moe_a8(hidden_states, gate_up_q, sorted_ids, expert_ids, ntpp, qt, n2, top_k, num_tokens)
    inter = act_fn(gate_up)
    # down: the same sorting, now over the num_tokens*top_k intermediate rows (top_k 1)
    return ggml_moe_a8(inter, down_q, sorted_ids, expert_ids, ntpp, qt, h, 1, num_tokens * top_k)


def fused_experts_gguf(
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,  # [num_slots, 2I, row_bytes(H)] uint8
    down_q: torch.Tensor,  # [num_slots, H, row_bytes(I)] uint8
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    quant_type: int,
    *,
    min_rows_mmq: int | None = None,
) -> torch.Tensor:
    from freetoken.kernel.gguf import ggml_moe_a8_vec

    act_fn = _ACT.get(activation)
    if act_fn is None:
        raise ValueError(f"unsupported MoE activation {activation!r}")

    num_tokens = hidden_states.shape[0]
    n2 = gate_up_q.shape[1]  # 2 * intermediate
    h = down_q.shape[1]  # hidden
    top_k = topk_ids.shape[1]
    qt = int(quant_type)
    rows = num_tokens * top_k
    threshold = mmq_min_rows() if min_rows_mmq is None else int(min_rows_mmq)
    if qt == GGML_Q4_0:
        # the vendored q4_0 tiles load 8 blocks (256 weights) per warp iteration whatever the
        # row has left, reading past a row whose K is not a multiple of 256 (gemma4's down
        # GEMM, K = 704); only the q2_0 tiles are bounded, so q4_0 keeps MMVQ there
        K_gate, K_down = hidden_states.shape[1], gate_up_q.shape[1] // 2
        if K_gate % 256 or K_down % 256:
            threshold = 0

    if threshold > 0 and rows >= threshold:
        out = _mmq(hidden_states, gate_up_q, down_q, topk_ids, act_fn, qt, num_tokens, top_k, n2, h)
    else:
        # gate_up: [num_tokens*top_k, 2I] -> activation -> [num_tokens*top_k, I]
        gate_up = ggml_moe_a8_vec(hidden_states, gate_up_q, topk_ids, top_k, qt, n2, num_tokens)
        inter = act_fn(gate_up)
        # down: each of the num_tokens*top_k intermediate rows uses its own expert id.
        out = ggml_moe_a8_vec(inter, down_q, topk_ids, 1, qt, h, num_tokens * top_k)
    out = out.reshape(num_tokens, top_k, h) * topk_weights.reshape(num_tokens, top_k, 1).to(
        out.dtype
    )
    return out.sum(dim=1)


def fused_experts_gguf_format(
    fmt: str,
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,
    down_q: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    """``fused_experts_gguf`` keyed by the bank format tag."""
    return fused_experts_gguf(
        hidden_states, gate_up_q, down_q, topk_weights, topk_ids, activation, GGUF_FORMAT_TYPES[fmt]
    )


__all__ = [
    "GGUF_FORMAT_TYPES",
    "GGUF_EXPERT_FORMATS",
    "mmq_min_rows",
    "fused_experts_gguf",
    "fused_experts_gguf_format",
]
