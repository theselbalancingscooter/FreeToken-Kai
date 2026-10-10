"""The tiled MoE MMQ kernel over Q2_0 banks (``ggml_moe_a8`` type 42: ``load_tiles_q2_0`` /
``vec_dot_q2_0_q8_1_mul_mat`` in vecdotq.cuh, ``moe_q2_0`` in moe.cuh) against the torch
reference and against the per-row MMVQ kernel, and the ``fused_experts_gguf`` switch between
them. The expert dimension is a slot cache here (thousands of rows), which the vendored
kernel's 256-expert guard used to refuse."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _reference(x, packed, topk_ids, K):
    """Dequantize only the experts the routing uses (a 3,000-slot bank is tens of GB in fp32)."""
    from freetoken.models.gguf.dequant import GGML_Q2_0, dequantize

    S, N, _ = packed.shape
    T, top_k = topk_ids.shape
    out = torch.empty(T * top_k, N, device=x.device, dtype=torch.float32)
    cache = {}
    for t in range(T):
        for k in range(top_k):
            e = int(topk_ids[t, k])
            if e not in cache:
                cache[e] = dequantize(packed[e].reshape(-1), GGML_Q2_0, torch.float32).reshape(N, K)
            out[t * top_k + k] = x[t].float() @ cache[e].T
    return out


def _mmq(x, packed, ids, top_k, N):
    from freetoken.kernel.gguf import ggml_moe_a8, ggml_moe_get_block_size
    from freetoken.models.gguf.dequant import GGML_Q2_0
    from freetoken.moe.fused import moe_align_block_size

    S = packed.shape[0]
    block = ggml_moe_get_block_size(int(GGML_Q2_0))
    assert block > 0
    sorted_ids, expert_ids, ntpp = moe_align_block_size(ids, block, S)
    expert_ids = torch.where(expert_ids >= S, expert_ids.new_full((), -1), expert_ids)
    return ggml_moe_a8(x, packed, sorted_ids, expert_ids, ntpp, int(GGML_Q2_0), N, top_k, x.shape[0])


@pytest.mark.parametrize("K,N,S,T,top_k", [
    (2560, 1280, 64, 6, 10),    # a 6-row verify window over a layer's gate_up
    (640, 2560, 64, 60, 1),     # its down GEMM (top_k 1 over the 60 intermediate rows)
    (2560, 1280, 3000, 6, 10),  # a slot cache larger than the old 256-expert guard
    (512, 256, 16, 37, 4),      # tokens not a multiple of the tile
])
def test_moe_mmq_q2_0_matches_reference(K, N, S, T, top_k):
    from freetoken.models.gguf.dequant import quantize_q2_0

    torch.manual_seed(K + N + S)
    dev = torch.device("cuda")
    # quantize in slices: S x N x K bf16 at once would be 20 GB for the 3,000-slot case
    packed = torch.empty(S, N, K // 64 * 18, dtype=torch.uint8)
    for s0 in range(0, S, 64):
        packed[s0:s0 + 64] = quantize_q2_0(torch.randn(min(64, S - s0), N, K, dtype=torch.bfloat16) * 0.05)
    packed = packed.to(dev)
    x = torch.randn(T, K, device=dev, dtype=torch.bfloat16) * 0.5
    ids = torch.stack([torch.randperm(S, device=dev)[:top_k] for _ in range(T)]).to(torch.int32)
    out = _mmq(x, packed, ids, top_k, N).float()
    ref = _reference(x, packed, ids, K)
    rel = (out - ref).abs().max() / (ref.abs().max() + 1e-6)
    assert rel < 2e-2, f"q2_0 mmq K={K} N={N} S={S} rel err {rel.item()}"


def _mmq_type(x, packed, ids, top_k, N, qtype):
    from freetoken.kernel.gguf import ggml_moe_a8, ggml_moe_get_block_size
    from freetoken.kernel.triton.moe_align import moe_align_block_size

    S = packed.shape[0]
    block = ggml_moe_get_block_size(qtype)
    assert block > 0
    sorted_ids, expert_ids, ntpp = moe_align_block_size(ids, block, S)
    expert_ids = torch.where(expert_ids >= S, expert_ids.new_full((), -1), expert_ids)
    return ggml_moe_a8(x, packed, sorted_ids, expert_ids, ntpp, qtype, N, top_k, x.shape[0])


@pytest.mark.parametrize("K,N,S,T,top_k", [
    (2560, 1280, 64, 512, 10),   # a prefill chunk's gate_up (5,120 rows, ~80 per expert)
    (640, 2560, 64, 2048, 1),    # its down GEMM
    (2560, 1312, 16, 300, 4),    # N not a multiple of the 64-row tile (need_check)
])
def test_moe_mmq_q2_0_wide_tile_matches_the_narrow_one(K, N, S, T, top_k):
    """The 32-token tile (type 1042) is the same math as the 4-token tile: equal up to bf16
    rounding of the output (fp32 accumulation order differs)."""
    from freetoken.models.gguf.dequant import GGML_Q2_0, quantize_q2_0

    torch.manual_seed(K + N + T)
    dev = torch.device("cuda")
    packed = quantize_q2_0(torch.randn(S, N, K, dtype=torch.bfloat16) * 0.05).to(dev)
    x = torch.randn(T, K, device=dev, dtype=torch.bfloat16) * 0.5
    ids = torch.stack([torch.randperm(S, device=dev)[:top_k] for _ in range(T)]).to(torch.int32)
    narrow = _mmq_type(x, packed, ids, top_k, N, int(GGML_Q2_0)).float()
    wide = _mmq_type(x, packed, ids, top_k, N, 1042).float()
    rel = (wide - narrow).abs().max() / (narrow.abs().max() + 1e-6)
    assert rel < 2e-3, f"wide vs narrow rel err {rel.item()}"
    ref = _reference(x[:8], packed, ids[:8], K)
    rel_ref = (wide[: 8 * top_k] - ref).abs().max() / (ref.abs().max() + 1e-6)
    assert rel_ref < 2e-2, f"wide vs reference rel err {rel_ref.item()}"


def test_moe_mmq_q2_0_wide_tile_timing():
    """Not an assertion: prints the time of both tiles on a prefill-sized gate_up GEMM."""
    import time

    from freetoken.models.gguf.dequant import GGML_Q2_0, quantize_q2_0

    torch.manual_seed(1)
    dev = torch.device("cuda")
    K, N, S, T, top_k = 2560, 1280, 128, 1024, 10
    packed = quantize_q2_0(torch.randn(S, N, K, dtype=torch.bfloat16) * 0.05).to(dev)
    x = torch.randn(T, K, device=dev, dtype=torch.bfloat16) * 0.5
    ids = torch.stack([torch.randperm(S, device=dev)[:top_k] for _ in range(T)]).to(torch.int32)
    for name, qtype in (("narrow(4)", int(GGML_Q2_0)), ("wide(32)", 1042)):
        for _ in range(2):
            _mmq_type(x, packed, ids, top_k, N, qtype)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            _mmq_type(x, packed, ids, top_k, N, qtype)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 5 * 1e3
        tops = T * top_k * N * K * 2 / (ms * 1e-3) / 1e12
        print(f"{name}: {ms:.1f} ms for {T}x{top_k} rows ({tops:.1f} TOPS)")


def test_moe_mmq_q2_0_matches_mmvq_bitwise_close():
    """Both kernels quantize the activations the same way (q8_1), so they agree far more
    tightly than either does with the fp32 reference."""
    from freetoken.kernel.gguf import ggml_moe_a8_vec
    from freetoken.models.gguf.dequant import GGML_Q2_0, quantize_q2_0

    torch.manual_seed(5)
    dev = torch.device("cuda")
    K, N, S, T, top_k = 2560, 1280, 32, 6, 10
    packed = quantize_q2_0(torch.randn(S, N, K, dtype=torch.bfloat16) * 0.05).to(dev)
    x = torch.randn(T, K, device=dev, dtype=torch.bfloat16) * 0.5
    ids = torch.stack([torch.randperm(S, device=dev)[:top_k] for _ in range(T)]).to(torch.int32)
    a = _mmq(x, packed, ids, top_k, N).float()
    b = ggml_moe_a8_vec(x, packed, ids, top_k, int(GGML_Q2_0), N, T).float()
    rel = (a - b).abs().max() / (b.abs().max() + 1e-6)
    assert rel < 2e-3, f"mmq vs mmvq rel err {rel.item()}"


def test_fused_experts_gguf_mmq_and_mmvq_agree():
    from freetoken.models.gguf.dequant import GGML_Q2_0, quantize_q2_0
    from freetoken.moe.fused_gguf import fused_experts_gguf

    torch.manual_seed(11)
    dev = torch.device("cuda")
    H, I, S, T, top_k = 2560, 640, 48, 6, 10
    gate_up = quantize_q2_0(torch.randn(S, 2 * I, H, dtype=torch.bfloat16) * 0.05).to(dev)
    down = quantize_q2_0(torch.randn(S, H, I, dtype=torch.bfloat16) * 0.05).to(dev)
    x = torch.randn(T, H, device=dev, dtype=torch.bfloat16) * 0.5
    ids = torch.stack([torch.randperm(S, device=dev)[:top_k] for _ in range(T)]).to(torch.int32)
    w = torch.rand(T, top_k, device=dev, dtype=torch.float32)
    via_mmq = fused_experts_gguf(x, gate_up, down, w, ids, "silu", GGML_Q2_0, min_rows_mmq=2).float()
    via_mmvq = fused_experts_gguf(x, gate_up, down, w, ids, "silu", GGML_Q2_0, min_rows_mmq=0).float()
    rel = (via_mmq - via_mmvq).abs().max() / (via_mmvq.abs().max() + 1e-6)
    assert rel < 5e-3, f"fused mmq vs mmvq rel err {rel.item()}"
    # one decoded token stays on MMVQ by default (threshold 2) and both paths agree there too
    one = fused_experts_gguf(x[:1], gate_up, down, w[:1], ids[:1], "silu", GGML_Q2_0).float()
    one_mmq = fused_experts_gguf(x[:1], gate_up, down, w[:1], ids[:1], "silu", GGML_Q2_0, min_rows_mmq=1).float()
    rel1 = (one - one_mmq).abs().max() / (one.abs().max() + 1e-6)
    assert rel1 < 5e-3, rel1
