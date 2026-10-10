"""ggml Q2_0 (2.25 bit) support without a model: the reference quantizer / dequantizer, the
gguf-py-free GGUF reader, and the Qwen3.8-Flash-Next expert-bank loader on a GGUF the test
writes itself (--moe-expert-gguf). No CUDA needed (the loader pins banks when it is there)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.models.gguf.dequant import GGML_F32, GGML_Q2_0, GGML_Q4_0, dequantize, quantize_q2_0, row_bytes
from freetoken.models.gguf.rawreader import RawGguf, write_gguf


def test_q2_0_quantize_dequantize_roundtrip():
    torch.manual_seed(0)
    w = torch.randn(5, 256, dtype=torch.bfloat16)  # K = 256 -> 4 blocks of 64
    packed = quantize_q2_0(w)
    assert packed.shape == (5, 4 * 18) and packed.dtype == torch.uint8
    back = dequantize(packed.reshape(-1), GGML_Q2_0, torch.float32).reshape(5, 4, 64)
    d = w.float().reshape(5, 4, 64).abs().amax(dim=-1, keepdim=True)
    ratio = (back / d).round()
    assert set(ratio.unique().tolist()) <= {-1.0, 0.0, 1.0}  # amax delta: codes -1 / 0 / +1 only
    err = (back - w.float().reshape(5, 4, 64)).abs()
    assert bool((err <= d * 0.5 + 1e-3).all())
    # bit layout of one block: element 4j+k sits in bits 2k..2k+1 of byte j (after the fp16 d)
    codes = (ratio[0, 0] + 1).to(torch.uint8)
    qs = packed[0, 2:18]
    for e in range(64):
        assert int((int(qs[e // 4]) >> (2 * (e % 4))) & 3) == int(codes[e])
    # the fp16 delta is the block's max magnitude
    assert torch.allclose(packed[0, :2].contiguous().view(torch.float16).float(), d[0, 0].to(torch.float16).float())


def _expert_gguf(path, E, H, I, L, *, up_type=GGML_Q2_0, seed=1):
    """A GGUF with Flash-Next-named routed experts (Q2_0; ``up_type`` can mix one in) plus a
    stray F32 tensor and some metadata. Returns (per-layer packed gate, up, down) tensors."""
    torch.manual_seed(seed)
    tensors, gate, up, down = [], [], [], []
    for layer in range(L):
        g = quantize_q2_0(torch.randn(E, I, H, dtype=torch.bfloat16) * 0.05)
        d = quantize_q2_0(torch.randn(E, H, I, dtype=torch.bfloat16) * 0.05)
        if up_type == GGML_Q2_0:
            u = quantize_q2_0(torch.randn(E, I, H, dtype=torch.bfloat16) * 0.05)
        else:  # any Q4_0-shaped bytes: the loader must refuse the mix before reading them
            u = torch.randint(0, 256, (E, I, row_bytes(H, GGML_Q4_0)), dtype=torch.uint8)
        gate.append(g), up.append(u), down.append(d)
        tensors.append((f"blk.{layer}.ffn_gate_exps.weight", (H, I, E), GGML_Q2_0, g.numpy().tobytes()))
        tensors.append((f"blk.{layer}.ffn_up_exps.weight", (H, I, E), up_type, u.numpy().tobytes()))
        tensors.append((f"blk.{layer}.ffn_down_exps.weight", (I, H, E), GGML_Q2_0, d.numpy().tobytes()))
        tensors.append((f"blk.{layer}.attn_norm.weight", (H,), GGML_F32, torch.arange(H, dtype=torch.float32).numpy().tobytes()))
    meta = {"general.architecture": "qwen4exp", "qwen4exp.block_count": L, "tokenizer.ggml.tokens": ["a", "b"], "general.quantization_version": 2}
    write_gguf(str(path), tensors, meta)
    return gate, up, down


def test_rawreader_reads_back_what_write_gguf_wrote(tmp_path):
    E, H, I, L = 3, 128, 64, 2
    path = tmp_path / "experts.gguf"
    gate, up, down = _expert_gguf(path, E, H, I, L)
    g = RawGguf(str(path))
    assert g.version == 3 and g.architecture == "qwen4exp"
    assert g.metadata["qwen4exp.block_count"] == L and g.metadata["tokenizer.ggml.tokens"] == ["a", "b"]
    t = g.tensors["blk.1.ffn_gate_exps.weight"]
    assert t.shape == (E, I, H) and t.ggml_type == GGML_Q2_0 and t.type_name == "Q2_0"
    view = g.view("blk.1.ffn_gate_exps.weight")
    assert view.shape == (E * I, row_bytes(H, GGML_Q2_0))
    assert torch.equal(torch.from_numpy(view.copy()).reshape(E, I, -1), gate[1])
    assert torch.equal(g.packed("blk.0.ffn_down_exps.weight").reshape(E, H, -1), down[0])
    norm = g.packed("blk.0.attn_norm.weight").reshape(-1).view(torch.float32)
    assert torch.equal(norm, torch.arange(H, dtype=torch.float32))
    assert [n for n in g.tensors][:3] == ["blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight"]


def test_gguf_expert_format_detects_and_refuses_mixes(tmp_path):
    from freetoken.models.qwen4_exp.gguf_experts import gguf_expert_format

    path = tmp_path / "q2.gguf"
    _expert_gguf(path, 2, 64, 64, 1)
    assert gguf_expert_format(str(path)) == "q2_0"
    mixed = tmp_path / "mixed.gguf"
    _expert_gguf(mixed, 2, 64, 64, 1, up_type=GGML_Q4_0)
    with pytest.raises(ValueError, match="mix"):
        gguf_expert_format(str(mixed))


def test_load_gguf_expert_sources_fills_gate_up_and_down(tmp_path):
    from freetoken.models.qwen4_exp.gguf_experts import dummy_gguf_expert_sources, load_gguf_expert_sources

    E, H, I, L = 3, 128, 64, 2
    path = tmp_path / "experts.gguf"
    gate, up, down = _expert_gguf(path, E, H, I, L)
    config = SimpleNamespace(
        num_experts=E, hidden_size=H, moe_intermediate_size=I, num_moe_layers=L,
        first_k_dense_replace=0, moe_expert_gguf=str(path),
    )
    banks = load_gguf_expert_sources("unused-model-path", config, quant_format="q2_0")
    assert set(banks) == {"gate_up", "down"} and len(banks["gate_up"]) == L
    for layer in range(L):
        gu = banks["gate_up"][layer]
        assert gu.shape == (E, 2 * I, row_bytes(H, GGML_Q2_0)) and gu.dtype == torch.uint8
        assert torch.equal(gu[:, :I].cpu(), gate[layer]) and torch.equal(gu[:, I:].cpu(), up[layer])
        assert torch.equal(banks["down"][layer].cpu(), down[layer])
    with pytest.raises(ValueError, match="q4_0"):
        load_gguf_expert_sources("unused", config, quant_format="q4_0")
    dummy = dummy_gguf_expert_sources(config, quant_format="q2_0")
    assert dummy["gate_up"][0].shape == banks["gate_up"][0].shape
    # the dummy's block scales are finite
    scales = dummy["down"][1].reshape(-1, 18)[:, :2].contiguous().view(torch.float16)
    assert bool(torch.isfinite(scales.float()).all())
    # the file's experts dequantize through the shared reference
    back = dequantize(banks["down"][0][1].reshape(-1).cpu(), GGML_Q2_0, torch.float32)
    assert back.shape == (H * I,)
