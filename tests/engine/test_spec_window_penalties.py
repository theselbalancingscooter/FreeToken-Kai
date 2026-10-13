"""Presence / frequency penalties inside a --spec-mtp verify window.

The sampler counts one logits row per request; a verify window is one request with one row
per draft. The window's rows must see what a plain decode would have counted by then."""

from __future__ import annotations

import pytest
import torch

VOCAB = 12
PRESENCE, FREQUENCY = 0.7, 0.4


def _plain_decode(logits: torch.Tensor, counts: torch.Tensor) -> list[int]:
    """Greedy decode one row at a time, as Sampler.sample penalizes and counts it."""
    out = []
    for row in logits:
        tok = int(torch.argmax(row.float() - (FREQUENCY * counts + PRESENCE * (counts > 0))))
        counts[tok] += 1
        out.append(tok)
    return out


def _window(logits: torch.Tensor, counts: torch.Tensor, drafts: list[int]) -> list[int]:
    from freetoken.engine.engine import _count_window_tokens, _penalize_window
    from freetoken.engine.spec import accept_drafts

    penalty = (0, counts, PRESENCE, FREQUENCY)
    sampled = torch.argmax(_penalize_window(logits, penalty, drafts), dim=-1).tolist()
    accepted = accept_drafts(sampled, drafts)
    _count_window_tokens(penalty, accepted)
    return accepted


def _case(seed: int, rows: int = 6):
    g = torch.Generator().manual_seed(seed)
    # a narrow spread, so the penalties decide the argmax and tokens would repeat without them
    logits = torch.rand(rows, VOCAB, generator=g) * 0.5
    start = torch.randint(0, 3, (VOCAB,), generator=g, dtype=torch.int32)
    return logits, start


@pytest.mark.parametrize("seed", range(8))
def test_right_drafts_give_the_plain_decode(seed):
    logits, start = _case(seed)
    plain_counts = start.clone()
    plain = _plain_decode(logits, plain_counts)

    counts = start.clone()
    accepted = _window(logits, counts, drafts=plain[:-1])

    assert accepted == plain
    assert torch.equal(counts, plain_counts)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("wrong_at", [0, 2, 4])
def test_a_wrong_draft_ends_the_window_there(seed, wrong_at):
    logits, start = _case(seed)
    plain = _plain_decode(logits, start.clone())
    drafts = plain[:-1]
    drafts[wrong_at] = (drafts[wrong_at] + 1) % VOCAB

    counts = start.clone()
    accepted = _window(logits, counts, drafts)

    # the drafts before the wrong one, then the target's own sample in its place
    assert accepted == plain[: wrong_at + 1]
    kept = start.clone()
    _plain_decode(logits[: wrong_at + 1], kept)
    assert torch.equal(counts, kept)  # the rejected drafts were never counted


def test_the_penalties_change_the_answer():
    """Guards the cases above: without the penalties these logits pick other tokens."""
    logits, start = _case(0)
    plain = _plain_decode(logits, start.clone())
    assert plain != torch.argmax(logits, dim=-1).tolist()


def test_expanded_args_carry_no_penalties():
    from freetoken.engine.engine import _expand_sampling_args
    from freetoken.engine.sample import BatchSamplingArgs

    counts = torch.zeros(VOCAB, dtype=torch.int32)
    greedy = BatchSamplingArgs(temperatures=None, penalties=[(0, counts, PRESENCE, FREQUENCY)])
    assert _expand_sampling_args(greedy, 4).penalties == []
    assert greedy.penalties  # the batch's own args are left as they were

    sampled = BatchSamplingArgs(torch.tensor([0.8]), penalties=[(0, counts, PRESENCE, FREQUENCY)])
    wide = _expand_sampling_args(sampled, 4)
    assert wide.penalties == [] and wide.temperatures.shape == (4,)
