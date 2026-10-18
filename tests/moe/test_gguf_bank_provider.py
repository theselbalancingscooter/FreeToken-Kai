"""The fork's GGUF expert banks (--moe-expert-gguf, Q2_0 / Q4_0) load through load_expert_banks without a quant
method: the provider path (_legacy_expert_banks -> _gguf_banks) and the bank schema it keys on. Found on two RTX 3060s
after the 31c6ef4 merge, where the schema import had gone and the Q2_0 server died at load with a NameError."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("fmt", ["q2_0", "q4_0"])
def test_gguf_provider_builds_banks_from_the_schema(monkeypatch, fmt):
    from freetoken.moe import expert_banks as eb
    from freetoken.moe.offload_cache import _BANK_SCHEMAS

    E, L, block = 4, 2, 64 if fmt == "q2_0" else 32
    fake = {
        "gate_up": [torch.zeros(E, 16, 128 // block * 18, dtype=torch.uint8) for _ in range(L)],
        "down": [torch.zeros(E, 128, 8 // 8 * 18, dtype=torch.uint8) for _ in range(L)],
        "stray": [None] * L,  # a provider may hand back more than the schema; the banks keep the schema's roles only
    }
    calls: dict = {}

    def fake_load(model_path, model_config, *, quant_format, dummy=False, layer_sink=None):
        calls.update(path=model_path, quant_format=quant_format, dummy=dummy, sink=layer_sink)
        return fake

    monkeypatch.setattr("freetoken.models.weight.load_gguf_moe_expert_sources", fake_load)
    cfg = SimpleNamespace(expert_quant=fmt, num_moe_layers=L, moe_weight_format=fmt)
    banks = eb.load_expert_banks(
        "/nowhere/model", cfg, method=None, device=torch.device("cpu"), dtype=torch.bfloat16, dummy=True, parallel=False,
    )
    assert calls == {"path": "/nowhere/model", "quant_format": fmt, "dummy": True, "sink": None}
    assert banks.quant_format == fmt
    assert tuple(banks.sources) == _BANK_SCHEMAS[fmt] == ("gate_up", "down")
    assert banks.sources["gate_up"] is fake["gate_up"] and banks.streamed is False


def test_a_format_without_a_provider_is_refused():
    from freetoken.moe import expert_banks as eb

    cfg = SimpleNamespace(expert_quant="nvfp4", num_moe_layers=1)
    with pytest.raises(ValueError, match="format provider"):
        eb.load_expert_banks("/nowhere/model", cfg, method=None, device=torch.device("cpu"), dummy=True, parallel=False)
