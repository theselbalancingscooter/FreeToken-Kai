"""The batched GDN rollback (engine/spec.restore_gdn_stashes) leaves the pools exactly as the
per-layer SpecGdnStash.restore does, for every accepted count, and the PLE context helper gives
the same answer from the id tail as from the whole list."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from freetoken.engine import spec as spec_mod
from freetoken.engine.spec import ngram_context_after, restore_gdn_stashes
from freetoken.models.qwen4_exp.gdn import SpecGdnStash


@pytest.fixture(autouse=True)
def _batched(monkeypatch):
    monkeypatch.setattr(spec_mod, "_ROLLBACK_MODE", "batched")


def _pools(L=3, S=4, dim=8, K=4, HV=2, V=4, KK=4, device="cpu"):
    g = torch.Generator().manual_seed(0)
    return SimpleNamespace(
        conv_states=torch.randn(L, S, dim, K - 1, generator=g).to(device, torch.bfloat16),
        recurrent_states=torch.randn(L, S, HV, V, KK, generator=g).to(device),
    )


def _stashes(pool, T, slot=2, device="cpu"):
    g = torch.Generator().manual_seed(1)
    L, _, dim, Km1 = pool.conv_states.shape
    HV, V, KK = pool.recurrent_states.shape[2:]
    out = []
    for li in range(L):
        out.append(SpecGdnStash(
            pool=pool, li=li, slot=torch.tensor([slot], dtype=torch.int32, device=device),
            prev_conv=torch.randn(1, dim, Km1, generator=g).to(device, torch.bfloat16),
            conv_in=torch.randn(T, dim, generator=g).to(device, torch.bfloat16),
            ht=torch.randn(T, HV, V, KK, generator=g).to(device),
        ))
    return out


@pytest.mark.parametrize("accepted", [1, 2, 4])
@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_batched_restore_equals_per_layer_restore(accepted, device):
    T = 4
    ref_pool, new_pool = _pools(device=device), _pools(device=device)
    ref_stashes = _stashes(ref_pool, T, device=device)
    new_stashes = _stashes(new_pool, T, device=device)
    for s in ref_stashes:
        s.restore(accepted)
    restore_gdn_stashes(new_stashes, accepted)
    assert torch.equal(ref_pool.conv_states, new_pool.conv_states)
    assert torch.equal(ref_pool.recurrent_states, new_pool.recurrent_states)
    # the other slots were not touched
    before = _pools(device=device)
    mask = torch.ones(4, dtype=torch.bool)
    mask[2] = False
    assert torch.equal(new_pool.conv_states[:, mask], before.conv_states[:, mask])


def test_mixed_slots_fall_back_to_the_per_layer_restore():
    pool = _pools()
    stashes = _stashes(pool, 4)
    stashes[1].slot = torch.tensor([0], dtype=torch.int32)  # a second slot in the same window
    ref = copy.deepcopy(pool)
    ref_stashes = copy.deepcopy(stashes)
    for s in ref_stashes:
        s.pool = ref
        s.restore(2)
    restore_gdn_stashes(stashes, 2)
    assert torch.equal(pool.conv_states, ref.conv_states)
    assert torch.equal(pool.recurrent_states, ref.recurrent_states)


def test_ngram_context_from_the_tail_equals_the_whole_list():
    ids = list(range(100, 1000))
    drafts = [7, 8, 9]
    for ctx_len in (3, 5):
        for accepted in (1, 2, 4):
            whole = ngram_context_after(ids, drafts, accepted, ctx_len, -1)
            tail = ngram_context_after(ids[-(ctx_len + 1):], drafts, accepted, ctx_len, -1)
            assert whole == tail
