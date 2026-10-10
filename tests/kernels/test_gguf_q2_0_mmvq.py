"""The borrowed ggml MMVQ kernel over Q2_0 banks (``ggml_moe_a8_vec`` type 42, vecdotq.cuh
``vec_dot_q2_0_q8_1``) against the torch reference dequant. W2A8: the activations are
quantized to q8_1 inside the op, so the tolerance is the int8 activation rounding."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _reference(x, packed, topk_ids, K):
    from freetoken.models.gguf.dequant import GGML_Q2_0, dequantize

    S, N, _ = packed.shape
    w = dequantize(packed.reshape(-1).cpu(), GGML_Q2_0, torch.float32).reshape(S, N, K).to(x.device)
    T, top_k = topk_ids.shape
    out = torch.empty(T * top_k, N, device=x.device, dtype=torch.float32)
    for t in range(T):
        for k in range(top_k):
            out[t * top_k + k] = x[t].float() @ w[int(topk_ids[t, k])].T
    return out


@pytest.mark.parametrize("K,N", [(2560, 1280), (640, 2560), (64, 128)])
def test_moe_vec_q2_0_matches_reference(K, N):
    from freetoken.kernel.gguf import ggml_moe_a8_vec
    from freetoken.models.gguf.dequant import GGML_Q2_0, quantize_q2_0

    torch.manual_seed(K + N)
    dev = torch.device("cuda")
    S, T, top_k = 6, 3, 2
    packed = quantize_q2_0(torch.randn(S, N, K, dtype=torch.bfloat16) * 0.05).to(dev)
    x = (torch.randn(T, K, device=dev, dtype=torch.bfloat16) * 0.5)
    ids = torch.stack([torch.randperm(S, device=dev)[:top_k] for _ in range(T)]).to(torch.int32)
    out = ggml_moe_a8_vec(x, packed, ids, top_k, int(GGML_Q2_0), N, T).float()
    ref = _reference(x, packed, ids, K)
    rel = (out - ref).abs().max() / (ref.abs().max() + 1e-6)
    assert rel < 2e-2, f"q2_0 mmvq K={K} N={N} rel err {rel.item()}"


def test_moe_vec_q2_0_single_block_sign_term():
    """One 64-weight block per row, codes forced to -1 / 0 / +1 (the -1 term comes out of the
    q8_1 block sums): a hand-built bank against a hand-built expectation."""
    from freetoken.kernel.gguf import ggml_moe_a8_vec
    from freetoken.models.gguf.dequant import GGML_Q2_0, quantize_q2_0

    dev = torch.device("cuda")
    K, N = 64, 4
    w = torch.zeros(1, N, K)
    w[0, 0] = 1.0          # all +1
    w[0, 1] = -1.0         # all -1
    w[0, 2, ::2] = 1.0     # +1 on even, 0 on odd
    w[0, 3, :32] = -1.0    # -1 on the first q8_1 block, 0 on the second
    packed = quantize_q2_0(w.to(torch.bfloat16)).to(dev)
    x = torch.ones(1, K, device=dev, dtype=torch.bfloat16)
    ids = torch.zeros(1, 1, device=dev, dtype=torch.int32)
    out = ggml_moe_a8_vec(x, packed, ids, 1, int(GGML_Q2_0), N, 1).float().cpu()
    assert torch.allclose(out[0], torch.tensor([64.0, -64.0, 32.0, -32.0]), atol=0.5), out
