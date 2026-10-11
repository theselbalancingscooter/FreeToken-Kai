"""--spec-min-p's cutoff rule and the --dense-quant keep-bf16 knobs."""

from __future__ import annotations

import importlib

import pytest


def test_cut_drafts_keeps_the_confident_prefix():
    from freetoken.engine.spec import cut_drafts

    assert cut_drafts([1, 2, 3], [0.9, 0.8, 0.7], 0.0) == [1, 2, 3]
    assert cut_drafts([1, 2, 3], [0.9, 0.5, 0.95], 0.6) == [1]
    assert cut_drafts([1, 2, 3], [0.2, 0.9, 0.9], 0.6) == []
    assert cut_drafts([1, 2, 3], [0.9, 0.8, 0.7], 0.7) == [1, 2, 3]
    assert cut_drafts([], [], 0.7) == []


@pytest.mark.parametrize("env,gone", [
    ({}, ()),
    ({"FREETOKEN_DENSE_QUANT_HC": "1"}, ("*hyper_connection*",)),
    ({"FREETOKEN_DENSE_QUANT_ROUTER": "1"}, ("*.gate", "*.shared_expert_gate")),
])
def test_keep_bf16_knobs(monkeypatch, env, gone):
    for k in ("FREETOKEN_DENSE_QUANT_HC", "FREETOKEN_DENSE_QUANT_ROUTER"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    from freetoken.layers.quantization import at_load

    pats = at_load._keep_bf16_patterns()
    for g in gone:
        assert g not in pats
    for keep in ("*.indexer", "*.ple", "visual"):
        assert keep in pats
    if not gone:
        assert "*hyper_connection*" in pats and "*.gate" in pats
