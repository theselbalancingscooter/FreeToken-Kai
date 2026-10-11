"""``--dense-quant q8_0``: the Q8_0 quantizer against the vendored dequantizer, and the linear
method's three paths (MMVQ for one row, int8 MMQ for a window, dequant + cuBLAS for prefill)
against the bf16 product."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _layer(w: torch.Tensor, bias: bool = False):
    from freetoken.layers.quantization import LayerKind
    from freetoken.layers.quantization.linear import LinearConfig
    from freetoken.layers.quantization.linear.q8_0 import Q8_0LinearMethod
    from freetoken.layers.quantization.scheme import q8_0_scheme
    from freetoken.models.gguf.dequant import quantize_q8_0

    N, K = w.shape
    cfg = LinearConfig(K, N, (N,), q8_0_scheme())
    method = Q8_0LinearMethod(cfg)
    layer = SimpleNamespace(in_features=K, out_features=N, output_sizes=(N,), quant_layer_kind=LayerKind.LINEAR)
    method.create_weights(layer)
    assert layer.weight.dtype == torch.uint8 and layer.weight.shape == (N, K // 32 * 34)
    layer.weight = quantize_q8_0(w).to(w.device)
    layer.bias = (torch.randn(N, device=w.device, dtype=w.dtype) * 0.1) if bias else None
    method.finalize(layer)
    return method, layer


def test_quantizer_matches_the_vendored_dequantizer_and_the_torch_one():
    from freetoken.kernel.gguf import ggml_dequantize
    from freetoken.models.gguf.dequant import GGML_Q8_0, dequant_q8_0, quantize_q8_0

    torch.manual_seed(0)
    w = torch.randn(96, 320, device="cuda", dtype=torch.bfloat16) * 0.05
    q = quantize_q8_0(w)
    assert q.shape == (96, 320 // 32 * 34)
    back_cuda = ggml_dequantize(q, GGML_Q8_0, 96, 320, torch.float32)
    back_torch = dequant_q8_0(q, torch.float32).reshape(96, 320)
    # the vendored kernel multiplies in fp16, the torch one in fp32: half-ulp differences
    assert torch.allclose(back_cuda, back_torch, rtol=2e-3, atol=1e-5)
    rel = (back_torch - w.float()).abs().max() / w.float().abs().max()
    assert rel < 1e-2, rel  # int8 with a per-32 scale: well under 1%
    # row-chunked quantization is the same bytes
    assert torch.equal(quantize_q8_0(w, chunk_rows=7), q)


@pytest.mark.parametrize("rows,path", [(1, "mmvq"), (6, "mmq"), (64, "mmq"), (200, "dequant")])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_linear_method_paths_match_the_product(rows, path, dtype):
    torch.manual_seed(rows)
    N, K = 1280, 2560
    w = torch.randn(N, K, device="cuda", dtype=dtype) * 0.02
    method, layer = _layer(w, bias=True)
    assert method.kernel.name == "ggml"
    x = torch.randn(rows, K, device="cuda", dtype=dtype)
    out = method.apply(layer, x).float()
    ref = x.float() @ w.float().t() + layer.bias.float()
    rel = (out - ref).abs().max() / ref.abs().max()
    assert rel < 2e-2, f"{path} rows={rows} rel err {rel.item()}"
    assert out.shape == (rows, N)


def test_the_three_paths_agree_with_each_other():
    """The same weights through MMVQ, MMQ and the dequantized GEMM differ only by the q8_1
    activation rounding of the first two."""
    from freetoken.layers.quantization.linear import q8_0 as mod

    torch.manual_seed(3)
    N, K = 640, 2560
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    method, layer = _layer(w)
    x = torch.randn(4, K, device="cuda", dtype=torch.bfloat16)
    old = dict(mmvq=mod.mmvq_max_rows, mmq=mod.mmq_max_rows)
    try:
        mod.mmvq_max_rows = lambda: 8
        via_mmvq = method.apply(layer, x).float()
        mod.mmvq_max_rows = lambda: 0
        via_mmq = method.apply(layer, x).float()
        mod.mmq_max_rows = lambda: 0
        via_gemm = method.apply(layer, x).float()
    finally:
        mod.mmvq_max_rows, mod.mmq_max_rows = old["mmvq"], old["mmq"]
    scale = via_gemm.abs().max()
    assert (via_mmvq - via_gemm).abs().max() / scale < 1e-2
    assert (via_mmq - via_gemm).abs().max() / scale < 1e-2
    assert (via_mmvq - via_mmq).abs().max() / scale < 3e-3


def test_a_row_length_off_the_kernel_step_takes_the_dequant_path_and_stays_finite():
    """K = 320 (the hyper-connection up projection): the vendored kernels' last warp step would
    read past the last row. The method dequantizes instead; the product must be finite and right
    even with NaN-decoding bytes placed right after the weight."""
    torch.manual_seed(7)
    N, K = 1024, 320
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05
    method, layer = _layer(w)
    poison = torch.full((65536,), 0xFF, dtype=torch.uint8, device="cuda")
    buf = torch.cat([layer.weight.reshape(-1), poison])
    layer.weight = buf[: N * (K // 32 * 34)].view(N, K // 32 * 34)
    for rows in (1, 4, 6, 100):
        x = torch.randn(rows, K, device="cuda", dtype=torch.bfloat16)
        out = method.apply(layer, x).float()
        assert torch.isfinite(out).all()
        ref = x.float() @ w.float().t()
        assert (out - ref).abs().max() / ref.abs().max() < 2e-2


@pytest.mark.parametrize("rows", [2, 4, 6, 8, 40])
@pytest.mark.parametrize("nsplit", [2, 3, 4, 8])
def test_splitk_mmq_equals_the_plain_one(rows, nsplit):
    from freetoken.kernel.gguf import ggml_mul_mat_a8, ggml_mul_mat_a8_splitk
    from freetoken.models.gguf.dequant import GGML_Q8_0, quantize_q8_0

    torch.manual_seed(rows * 10 + nsplit)
    N, K = 1280, 2560
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    q = quantize_q8_0(w)
    x = torch.randn(rows, K, device="cuda", dtype=torch.bfloat16)
    plain = ggml_mul_mat_a8(q, x, GGML_Q8_0, N).float()
    split = ggml_mul_mat_a8_splitk(q, x, GGML_Q8_0, N, nsplit).float()
    ref = x.float() @ w.float().t()
    assert (split - plain).abs().max() / ref.abs().max() < 2e-3
    assert (split - ref).abs().max() / ref.abs().max() < 2e-2


def test_leading_dims_and_non_contiguous_input():
    torch.manual_seed(5)
    N, K = 256, 512
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05
    method, layer = _layer(w)
    x = torch.randn(2, 3, 2 * K, device="cuda", dtype=torch.bfloat16)[..., ::2]  # strided view
    out = method.apply(layer, x)
    assert out.shape == (2, 3, N)
    ref = x.float() @ w.float().t()
    assert ((out.float() - ref).abs().max() / ref.abs().max()) < 2e-2
