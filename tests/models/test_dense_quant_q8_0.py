"""``--dense-quant q8_0``: the load-time config names the same projections as fp8 (and keeps the
same ones bf16), a linear built under it holds a packed uint8 row buffer, the at-load quantizer
turns a bf16 checkpoint tensor into that buffer's shape, and the torch fallback kernel computes
the right product without CUDA."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info


@pytest.fixture(autouse=True)
def _single_rank():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def test_q8_config_names_what_fp8_names():
    from freetoken.layers.quantization import AT_LOAD_Q8_0, LoadTimeFp8Config, LoadTimeQ8Config, QuantKind

    q8, fp8 = LoadTimeQ8Config(None), LoadTimeFp8Config(None)
    for prefix in ("model.layers.0.self_attn.o_proj", "model.layers.3.linear_attn.in_proj_qkvz", "lm_head",
                   "model.layers.0.mlp.gate", "model.layers.0.attn_hyper_connection", "visual.blocks.0.attn.qkv",
                   "model.layers.0.linear_attn.in_proj_ba"):
        a, b = q8.scheme_for(prefix), fp8.scheme_for(prefix)
        assert (a is None) == (b is None), prefix
        if a is not None:
            assert a is AT_LOAD_Q8_0 and a.kind is QuantKind.Q8_0


def test_the_wrapper_answers_storage_for_the_inner_dialect_and_for_itself():
    """A ModelOpt export quantizes its attention; under --dense-quant the loader still asks how
    those are stored, and the wrapper must answer with the inner dialect's table (it used to raise
    AttributeError: no STORAGE)."""
    from freetoken.layers.quantization import LoadTimeFp8Config, LoadTimeQ8Config, QuantKind
    from freetoken.layers.quantization.configs.base import NoQuantConfig, Stored

    class Inner(NoQuantConfig):
        STORAGE = {QuantKind.NVFP4: {"weight": "weight", "weight_scale": "weight_scale",
                                     "weight_global": Stored("weight_scale_2", reciprocal=True)}}

    q8 = LoadTimeQ8Config(Inner())
    nv = q8.stored_tensors(QuantKind.NVFP4)
    assert nv["weight_global"] == Stored("weight_scale_2", reciprocal=True)
    assert q8.stored_tensors(QuantKind.Q8_0) == {"weight": Stored("weight")}
    fp8 = LoadTimeFp8Config(Inner())
    assert set(fp8.stored_tensors(QuantKind.FP8_TENSOR)) == {"weight", "weight_scale"}
    assert fp8.stored_tensors(QuantKind.NVFP4) == nv


def test_a_linear_built_under_q8_holds_packed_rows_and_the_head_too():
    from freetoken.layers.embedding import ParallelLMHead
    from freetoken.layers.linear import LinearColParallelMerged
    from freetoken.layers.quantization import LoadTimeQ8Config

    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            lin = LinearColParallelMerged(2560, [1280, 1280], False, quant_config=LoadTimeQ8Config(None),
                                          prefix="model.layers.0.mlp.shared_expert.gate_up_proj")
            head = ParallelLMHead(248320, 2560, quant_config=LoadTimeQ8Config(None), prefix="lm_head")
    finally:
        torch.set_default_dtype(torch.float32)
    assert lin.weight.dtype == torch.uint8 and lin.weight.shape == (2560, 2560 // 32 * 34)
    assert head.weight.dtype == torch.uint8 and head.weight.shape == (248320, 2560 // 32 * 34)
    assert not hasattr(lin, "weight_scale")


def test_the_loader_quantizes_a_bf16_tensor_into_the_packed_buffer():
    from freetoken.engine.engine import _quantize_at_load
    from freetoken.models.gguf.dequant import dequant_q8_0

    w = torch.randn(64, 320, dtype=torch.bfloat16) * 0.05
    model_state = {
        "model.layers.0.self_attn.o_proj.weight": torch.empty(64, 320 // 32 * 34, dtype=torch.uint8),
        "model.layers.0.mlp.gate.weight": torch.empty(64, 320, dtype=torch.bfloat16),
    }
    out = dict(_quantize_at_load(
        [("model.layers.0.self_attn.o_proj.weight", w), ("model.layers.0.mlp.gate.weight", w)], model_state
    ))
    q = out["model.layers.0.self_attn.o_proj.weight"]
    assert q.dtype == torch.uint8 and q.shape == (64, 340)
    back = dequant_q8_0(q, torch.float32).reshape(64, 320)
    assert ((back - w.float()).abs().max() / w.float().abs().max()) < 1e-2
    assert out["model.layers.0.mlp.gate.weight"].dtype == torch.bfloat16  # untouched


def test_torch_fallback_kernel_without_cuda():
    from freetoken.layers.quantization import LayerKind
    from freetoken.layers.quantization.linear import LinearConfig
    from freetoken.layers.quantization.linear.q8_0 import Q8_0LinearMethod, TorchQ8_0LinearKernel
    from freetoken.layers.quantization.scheme import q8_0_scheme
    from freetoken.models.gguf.dequant import quantize_q8_0

    N, K = 48, 256
    w = torch.randn(N, K, dtype=torch.float32) * 0.05
    method = Q8_0LinearMethod(LinearConfig(K, N, (N,), q8_0_scheme()), requested="torch")
    assert isinstance(method.kernel, TorchQ8_0LinearKernel)
    layer = SimpleNamespace(in_features=K, out_features=N, output_sizes=(N,), quant_layer_kind=LayerKind.LINEAR,
                            bias=None)
    method.create_weights(layer)
    layer.weight = quantize_q8_0(w)
    x = torch.randn(3, K)
    out = method.apply(layer, x)
    ref = x @ w.t()
    assert ((out - ref).abs().max() / ref.abs().max()) < 1e-2
