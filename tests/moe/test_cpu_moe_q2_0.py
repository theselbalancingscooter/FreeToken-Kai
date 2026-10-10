"""CPU MoE executor -- native GGUF Q2_0 experts (``--moe-expert-gguf`` banks, ``--moe-strategy
cpu`` / ``hybrid``).

The CPU W2A8 GEMV (``q2_0_dot_i8_*`` in csrc/cpu_moe/cpu_moe_ext.cpp) reads the *same* packed
Q2_0 banks the GPU offload path streams: 2-bit planes unpacked in the K-loop, the Q8_0
activations permuted once into plane order. Checked against the reference dequant
(models/gguf/dequant.py) + the production bf16 GPU decode kernel on byte-identical banks, and
against the GPU ggml MMVQ (q2_0) kernel; the spread is the int8 activation rounding.
Part 2 covers CUDA-graph capture/replay. Geometry is Flash-Next's (H=2560, I=640, both % 64)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

H, I = 2560, 640


def _dequant_bank(packed: torch.Tensor, K: int, dev) -> torch.Tensor:
    """[S, OUT, K//64*18] packed Q2_0 -> [S, OUT, K] bf16."""
    from freetoken.models.gguf.dequant import GGML_Q2_0, dequantize

    S, OUT, _ = packed.shape
    return dequantize(packed.reshape(-1), GGML_Q2_0, torch.bfloat16).reshape(S, OUT, K).to(dev)


def _make_q2_0_cache(L, E, seed=0):
    """Q2_0 banks quantized from random bf16 experts (pinned host tensors)."""
    from freetoken.kernel.pinned import alloc_pinned_tensor
    from freetoken.models.gguf.dequant import quantize_q2_0

    torch.manual_seed(seed)
    S = L * E

    def rows(OUT, K):
        packed = quantize_q2_0(torch.randn(S, OUT, K, dtype=torch.bfloat16) * 0.05)
        pinned = alloc_pinned_tensor(*packed.shape, dtype=torch.uint8)
        pinned.copy_(packed)
        return pinned

    return SimpleNamespace(
        quant_format="q2_0",
        bank_sources={"gate_up": list(rows(2 * I, H).split(E)), "down": list(rows(H, I).split(E))},
        num_layers=L,
        num_experts=E,
        decode_target="cpu",
        cpu_executor=None,
    )


@pytest.mark.parametrize("bs", [1, 3, 8])
def test_cpu_decode_q2_0_matches_dequant_then_gpu(bs):
    """CPU W2A8 Q2_0 GEMV vs. reference dequant_q2_0 + bf16 GPU decode."""
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused import fused_experts_decode_impl

    torch.manual_seed(200 + bs)
    L, E, top_k = 3, 16, 10
    layer = 1
    dev = torch.device("cuda")
    cache = _make_q2_0_cache(L, E)

    ex = CpuMoeExecutor(
        cache, top_k=top_k, activation="silu", apply_router_weight_on_input=False,
        num_threads=0, max_tokens=bs, device=dev,
    )
    assert "q2_0" in ex.isa, ex.isa

    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16) * 0.5
    ids = torch.stack([torch.randperm(E, device=dev)[:top_k] for _ in range(bs)]).to(torch.int32)
    w = torch.rand(bs, top_k, device=dev, dtype=torch.float32)

    cpu_out = ex.decode(layer, hidden, w, ids).float()
    torch.cuda.synchronize()

    b = cache.bank_sources
    gate_up_layer = _dequant_bank(b["gate_up"][layer], H, dev)  # [E, 2I, H]
    down_layer = _dequant_bank(b["down"][layer], I, dev)        # [E, H, I]
    gpu_out = fused_experts_decode_impl(hidden, gate_up_layer, down_layer, w, ids.clone(), "silu", False).float()

    rel = (cpu_out - gpu_out).abs().max() / (gpu_out.abs().max() + 1e-6)
    assert rel < 2e-2, f"q2_0 bs={bs} rel err {rel.item()}"


def test_cpu_decode_q2_0_matches_ggml_mmvq():
    """The CPU W2A8 GEMV lands close to the GPU ggml MMVQ (q2_0) kernel the offload path uses."""
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused_gguf import fused_experts_gguf_format

    torch.manual_seed(77)
    L, E, top_k = 2, 16, 10
    layer, bs = 1, 4
    dev = torch.device("cuda")
    cache = _make_q2_0_cache(L, E)

    ex = CpuMoeExecutor(
        cache, top_k=top_k, activation="silu", apply_router_weight_on_input=False,
        num_threads=0, max_tokens=bs, device=dev,
    )
    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16) * 0.5
    ids = torch.stack([torch.randperm(E, device=dev)[:top_k] for _ in range(bs)]).to(torch.int32)
    w = torch.rand(bs, top_k, device=dev, dtype=torch.float32)

    cpu_out = ex.decode(layer, hidden, w, ids).float()
    torch.cuda.synchronize()

    b = cache.bank_sources
    gpu_out = fused_experts_gguf_format(
        "q2_0", hidden, b["gate_up"][layer].to(dev), b["down"][layer].to(dev), w, ids.clone(), "silu"
    ).float()
    rel = (cpu_out - gpu_out).abs().max() / (gpu_out.abs().max() + 1e-6)
    assert rel < 6e-2, f"q2_0 vs ggml-mmvq rel err {rel.item()}"


def test_cpu_moe_decode_q2_0_cuda_graph_replay():
    """Q2_0 CPU path under capture/replay: the host nodes recompute the GEMV (and the plane
    permutation of the activations) from the freshly written pinned routing on each replay."""
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused import fused_experts_decode_impl

    torch.manual_seed(9)
    L, E, top_k = 2, 8, 4
    layer, bs = 1, 4
    cache = _make_q2_0_cache(L, E)

    dev = torch.device("cuda")
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    ex = CpuMoeExecutor(
        cache, top_k=top_k, activation="silu", apply_router_weight_on_input=False,
        num_threads=8, max_tokens=bs, device=dev,
    )
    b = cache.bank_sources
    gate_up_layer = _dequant_bank(b["gate_up"][layer], H, dev)
    down_layer = _dequant_bank(b["down"][layer], I, dev)

    def reference(hidden, ids, w):
        return fused_experts_decode_impl(hidden, gate_up_layer, down_layer, w, ids.clone(), "silu", False).float()

    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16) * 0.5
    ids = torch.randint(0, E, (bs, top_k), device=dev, dtype=torch.int32)
    w = torch.rand(bs, top_k, device=dev, dtype=torch.float32)

    ex.decode(layer, hidden, w, ids)  # eager warmup: materialize buffers + task
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        out_cap = ex.decode(layer, hidden, w, ids)
    torch.cuda.synchronize()

    for it in range(3):
        torch.manual_seed(600 + it)
        hidden.copy_(torch.randn(bs, H, dtype=torch.bfloat16) * 0.5)
        ids.copy_(torch.randint(0, E, (bs, top_k), dtype=torch.int32))
        w.copy_(torch.rand(bs, top_k, dtype=torch.float32))
        g.replay()
        torch.cuda.synchronize()
        ref = reference(hidden, ids, w)
        rel = (out_cap.float() - ref).abs().max() / (ref.abs().max() + 1e-6)
        assert rel < 2e-2, f"q2_0 replay {it} rel err {rel.item()}"
