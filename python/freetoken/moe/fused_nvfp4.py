"""Host orchestration for the inline-dequant NVFP4 fused-MoE path.

Mirrors :mod:`freetoken.moe.fused` (gemm1 -> act -> gemm2 -> sum-reduce) but the two
grouped GEMMs read the NVFP4 expert cache directly and dequantize inside the K-loop,
so no BF16 copy of the experts is ever materialized.
"""

from __future__ import annotations

import functools
import os

import torch

from freetoken.kernel import moe_sum_reduce_triton, nvfp4_moe
from freetoken.kernel.triton.nvfp4_act_quant import quantize_nvfp4
from freetoken.layers import gated_act_and_mul
from freetoken.moe.fused import moe_align_block_size


def _fused_experts_decode_nvfp4(
    gemm_fn,
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    apply_router_weight_on_input: bool,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Shared decode body (gemm1 -> act -> gemm2 -> sum-reduce); ``gemm_fn`` is either
    the marlin-style int32 GEMV (:func:`freetoken.kernel.nvfp4_moe.decode_gemm_marlin`) or the original LUT-gather
    GEMV (:func:`freetoken.kernel.nvfp4_moe.decode_gemm`), both with the same calling convention."""
    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    two_i = gate_up_packed.shape[1]
    inter = two_i // 2
    dev, dt = hidden_states.device, hidden_states.dtype

    ic1 = torch.empty((M, top_k, two_i), device=dev, dtype=dt)
    gemm_fn(
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
        ic1, topk_weights, topk_ids, apply_router_weight_on_input, False,
    )
    ic2 = torch.empty((M * top_k, inter), device=dev, dtype=dt)
    gated_act_and_mul(activation, ic1.view(-1, two_i), ic2, alpha=act_alpha, limit=act_limit)
    ic3 = torch.empty((M, top_k, H), device=dev, dtype=dt)
    gemm_fn(
        ic2, down_packed, down_scale, down_global,
        ic3, topk_weights, topk_ids, not apply_router_weight_on_input, True,
    )
    out = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(ic3, out)
    return out


def fused_experts_decode_nvfp4_marlin(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Decode inline-NVFP4 MoE using the Marlin-style int32 wide-load GEMV."""
    return _fused_experts_decode_nvfp4(
        nvfp4_moe.decode_gemm_marlin,
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
        down_packed, down_scale, down_global,
        topk_weights, topk_ids, activation, apply_router_weight_on_input,
        act_alpha, act_limit,
    )


def fused_experts_decode_nvfp4_serial(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Original LUT-gather decode (one program per route, full K reduction). Retained for
    A/B benchmarking against the marlin decode path; not on the production decode path."""
    return _fused_experts_decode_nvfp4(
        nvfp4_moe.decode_gemm,
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
        down_packed, down_scale, down_global,
        topk_weights, topk_ids, activation, apply_router_weight_on_input,
        act_alpha, act_limit,
    )


@functools.cache
def _scratch_moe_preferred() -> bool:
    """Prefill MoE below Ampere: dequant + cuBLAS instead of the in-kernel-dequant GEMM. Measured
    on an RTX 2060 (T=2785, 256 experts, top-8): the Triton kernel tops out at 0.57 TFLOPS in
    every tile configuration tried, cuBLAS fp16 runs at 19 TFLOPS on the same card.
    ``FREETOKEN_NVFP4_MOE_SCRATCH=0/1`` overrides anywhere."""
    env = os.environ.get("FREETOKEN_NVFP4_MOE_SCRATCH")
    if env is not None:
        return env == "1"
    from freetoken.utils import is_pre_ampere

    return is_pre_ampere()


_SCRATCH_EXPERTS_PER_CHUNK = 16  # 16 experts x (2I x H + H x I) fp16 ~ 100 MB for Qwen3.5-35B-A3B
_SCRATCH_ROWS_PER_BLOCK = 64     # rows per expert per bmm; bounds the loop's buffers (~45 MB total)
_SCRATCH_BUFFERS: dict = {}


def _scratch_buffer(name: str, shape, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """One persistent buffer per (name, shape, dtype, device). The dequant scratches are
    allocated once and reused across chunks, layers and requests instead of being freed and
    re-allocated per chunk: on a full 6 GB card the caching allocator otherwise ends up
    unmapping expandable segments while the copy / host-callback streams are still busy, which
    surfaces as ``CUDA driver error: device not ready`` in whatever op allocates next."""
    key = (name, tuple(int(d) for d in shape), dtype, str(device))
    buf = _SCRATCH_BUFFERS.get(key)
    if buf is None:
        # first allocation only: let the queued work drain first. Growing an expandable
        # segment while kernels and copies are in flight is what failed with "device not
        # ready" on WSL2; a one-time sync per buffer costs nothing afterwards.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        buf = _SCRATCH_BUFFERS[key] = torch.empty(shape, dtype=dtype, device=device)
    return buf


def preallocate_scratch(hidden: int, inter: int, dtype: torch.dtype, device: torch.device) -> int:
    """Allocate every buffer ``_fused_experts_nvfp4_scratch`` uses, once, while the GPU is idle
    (engine init, before the graphs). Returns the bytes taken. Lazily growing them inside the
    first prefill put the allocator's segment growth under load, which on a full card under
    WSL2 failed with ``CUDA driver error: device not ready``."""
    C, LB = _SCRATCH_EXPERTS_PER_CHUNK, _SCRATCH_ROWS_PER_BLOCK
    bufs = [
        _scratch_buffer("gate_up", (C, 2 * inter, hidden), dtype, device),
        _scratch_buffer("down", (C, hidden, inter), dtype, device),
        _scratch_buffer("x", (C * LB, hidden), dtype, device),
        _scratch_buffer("h", (C, LB, 2 * inter), dtype, device),
        _scratch_buffer("a", (C, LB, inter), dtype, device),
        _scratch_buffer("y16", (C, LB, hidden), dtype, device),
        _scratch_buffer("y32", (C, LB, hidden), torch.float32, device),
    ]
    return sum(b.numel() * b.element_size() for b in bufs)


def _fused_experts_nvfp4_scratch(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    apply_router_weight_on_input: bool,
) -> torch.Tensor:
    """silu-gated prefill MoE as batched cuBLAS GEMMs over dequantised fp16/bf16 experts, with a
    bounded working set. Routes are grouped by expert once (argsort); experts are processed in
    chunks of 32, ordered by their route count so the experts of a chunk have similar queue
    lengths; each chunk's queues are walked in blocks of ``_SCRATCH_ROWS_PER_BLOCK`` rows and
    every block goes through two ``bmm`` calls (``x @ W_gu^T -> silu(gate) * up -> @ W_d^T``)
    over the experts that still have rows in it (a prefix of the chunk, thanks to the ordering).
    The route weights and the sum over a token's routes are applied in fp32.

    Every device tensor of the loop lives in a persistent buffer sized by the block, not by the
    prompt: on a full 6 GB card the earlier per-chunk temporaries (``[experts, L_max, H]`` fp32,
    100+ MB for a 4096-token chunk) made the caching allocator release expandable segments,
    which surfaces as ``CUDA driver error: device not ready``. One host sync per layer (route
    counts)."""
    from freetoken.kernel.triton.nvfp4_dequant import dequant_nvfp4

    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    dt = hidden_states.dtype
    dev = hidden_states.device
    inter = gate_up_packed.shape[1] // 2
    C, LB = _SCRATCH_EXPERTS_PER_CHUNK, _SCRATCH_ROWS_PER_BLOCK
    flat_ids = topk_ids.reshape(-1).to(torch.int64)
    flat_w = topk_weights.reshape(-1).to(torch.float32)
    order = torch.argsort(flat_ids)
    counts = torch.bincount(flat_ids.clamp(min=0), minlength=num_experts)
    starts = torch.cumsum(counts, 0) - counts
    counts_cpu = counts.tolist()
    # experts by queue length, longest first; idle experts drop out
    by_len = sorted((e for e in range(num_experts) if counts_cpu[e] > 0), key=lambda e: -counts_cpu[e])
    out = torch.zeros(M, H, dtype=torch.float32, device=dev)
    if not by_len:
        return out.to(dt)
    x_buf = _scratch_buffer("x", (C * LB, H), dt, dev)
    h_buf = _scratch_buffer("h", (C, LB, 2 * inter), dt, dev)
    a_buf = _scratch_buffer("a", (C, LB, inter), dt, dev)
    y16_buf = _scratch_buffer("y16", (C, LB, H), dt, dev)
    y32_buf = _scratch_buffer("y32", (C, LB, H), torch.float32, dev)
    ar = torch.arange(LB, device=dev)
    last = order.numel() - 1
    for c0 in range(0, len(by_len), C):
        chunk = by_len[c0 : c0 + C]
        n = len(chunk)
        slots = torch.tensor(chunk, dtype=torch.int32, device=dev)
        w_gu = dequant_nvfp4(
            gate_up_packed, gate_up_scale, gate_up_global, slots,
            out=_scratch_buffer("gate_up", (C, 2 * inter, H), dt, dev)[:n], dtype=dt,
        )
        w_d = dequant_nvfp4(
            down_packed, down_scale, down_global, slots,
            out=_scratch_buffer("down", (C, H, inter), dt, dev)[:n], dtype=dt,
        )
        e_ids = slots.long()
        for l0 in range(0, counts_cpu[chunk[0]], LB):
            # experts of the chunk with rows in this block: a prefix (queues sorted, longest first)
            na = sum(1 for e in chunk if counts_cpu[e] > l0)
            ids = e_ids[:na]
            # padded route table of the block: pos[j, l] = route l0 + l of expert ids[j]
            valid = (l0 + ar)[None, :] < counts[ids][:, None]                     # [na, LB]
            pos = (starts[ids][:, None] + l0 + ar[None, :]).clamp(max=last)
            rows = order[pos]                                                      # flat route ids
            tok = (rows // top_k) * valid                                          # padding -> token 0
            rw = flat_w[rows] * valid                                              # padding -> weight 0
            xb = torch.index_select(hidden_states, 0, tok.reshape(-1), out=x_buf[: na * LB]).view(na, LB, H)
            if apply_router_weight_on_input:
                xb = (xb.float() * rw[..., None]).to(dt)
            hb = torch.bmm(xb, w_gu[:na].transpose(1, 2), out=h_buf[:na])          # [na, LB, 2I]
            gate, up = hb[..., :inter], hb[..., inter:]
            torch.mul(torch.nn.functional.silu(gate, inplace=True), up, out=a_buf[:na])
            yb = torch.bmm(a_buf[:na], w_d[:na].transpose(1, 2), out=y16_buf[:na])  # [na, LB, H]
            scale = valid.to(torch.float32) if apply_router_weight_on_input else rw
            torch.mul(yb, scale[..., None], out=y32_buf[:na])
            out.index_add_(0, tok.reshape(-1), y32_buf[:na].reshape(-1, H))
    return out.to(dt)


def fused_experts_nvfp4(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Prefill inline-NVFP4 MoE. ``topk_ids`` index rows of the bank tensors in
    ``[0, num_experts)``: full-layer banks with position == expert id (the
    materialized ``[:E]`` slot view or the overlap double buffer), raw ids."""
    M, H = hidden_states.shape
    if activation == "silu" and _scratch_moe_preferred():
        return _fused_experts_nvfp4_scratch(
            hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
            down_packed, down_scale, down_global, topk_weights, topk_ids, num_experts,
            apply_router_weight_on_input,
        )
    top_k = topk_ids.shape[1]
    two_i = gate_up_packed.shape[1]
    inter = two_i // 2
    dev, dt = hidden_states.device, hidden_states.dtype
    cfg = nvfp4_moe.prefill_config(M, top_k, num_experts, hidden_states.device.index)

    sorted_ids, expert_ids, ntpp = moe_align_block_size(topk_ids, cfg["BLOCK_SIZE_M"], num_experts)
    tw = topk_weights.reshape(-1).contiguous()
    num_valid = topk_ids.numel()

    ic1 = torch.empty((M, top_k, two_i), device=dev, dtype=dt)
    nvfp4_moe.prefill_gemm(
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global, ic1,
        tw, sorted_ids, expert_ids, ntpp, num_valid, top_k,
        apply_router_weight_on_input, cfg,
    )
    ic2 = torch.empty((M * top_k, inter), device=dev, dtype=dt)
    gated_act_and_mul(activation, ic1.view(-1, two_i), ic2, alpha=act_alpha, limit=act_limit)
    ic3 = torch.empty((M, top_k, H), device=dev, dtype=dt)
    nvfp4_moe.prefill_gemm(
        ic2, down_packed, down_scale, down_global, ic3,
        tw, sorted_ids, expert_ids, ntpp, num_valid, 1,
        not apply_router_weight_on_input, cfg,
    )
    out = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(ic3, out)
    return out


def fused_experts_nvfp4_w4a4(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    gate_up_input_scale: torch.Tensor,
    down_input_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    activation: str = "silu",
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Prefill NVFP4 MoE with NVFP4 activations (W4A4) where the GPU and the batch favour it, else :func:`fused_experts_nvfp4`.

    ``*_input_scale`` are the per-expert static activation scales of the ``num_experts`` banks; each GEMM input is quantized under their max, as vLLM does.
    """
    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    inter = gate_up_packed.shape[1] // 2
    cfg = nvfp4_moe.w4a4_prefill_config(M, top_k, num_experts, H, inter, activation, hidden_states.dtype, hidden_states.device.index)
    if cfg is None:
        return fused_experts_nvfp4(
            hidden_states, gate_up_packed, gate_up_scale, gate_up_global, down_packed, down_scale, down_global,
            topk_weights, topk_ids, num_experts, activation, False, act_alpha, act_limit,
        )
    dev = hidden_states.device
    g = torch.stack((gate_up_input_scale[:num_experts], down_input_scale[:num_experts])).float().amax(dim=1)
    sorted_ids, expert_ids, ntpp = moe_align_block_size(topk_ids, cfg["BLOCK_SIZE_M"], num_experts)
    tw = topk_weights.reshape(-1).contiguous()
    num_valid = topk_ids.numel()

    xq, xs = quantize_nvfp4(hidden_states, g[0:1])
    q = torch.empty((num_valid, inter // 2), device=dev, dtype=torch.uint8)
    qs = torch.empty((num_valid, inter // 16), device=dev, dtype=torch.float8_e4m3fn)
    nvfp4_moe.w4a4_gate_up_gemm(
        xq, xs, g[0:1], gate_up_packed, gate_up_scale, gate_up_global, q, qs, g[1:2], sorted_ids, expert_ids, ntpp,
        num_valid, top_k, activation, act_alpha, act_limit, cfg["gate_up"],
    )
    out = torch.zeros_like(hidden_states)
    nvfp4_moe.w4a4_down_gemm(
        q, qs, g[1:2], down_packed, down_scale, down_global, out, tw, sorted_ids, expert_ids, ntpp, num_valid, top_k,
        cfg["down"],
    )
    return out


__all__ = [
    "fused_experts_decode_nvfp4_marlin",
    "fused_experts_decode_nvfp4_serial",
    "fused_experts_nvfp4",
    "fused_experts_nvfp4_w4a4",
]
