"""The reasoning budget (``--reasoning-budget`` / a request's ``reasoning_budget``): the
scheduler's host-side bookkeeping of the <think> span and the forced </think>. Shell
scheduler only -- no engine, no GPU."""

from __future__ import annotations

import torch

from freetoken.core import Req, SamplingParams
from freetoken.scheduler.scheduler import Scheduler

THINK, END = 100, 101


class _Handle:
    pass


def _sched(default: int = 0, end_id: int | None = END) -> Scheduler:
    s = Scheduler.__new__(Scheduler)
    s.think_start_id, s.think_end_id = THINK, end_id
    s._reasoning_budget_default = default
    return s


def _req(prompt: list[int], budget: int | None = None, output_len: int = 64) -> Req:
    return Req(
        input_ids=torch.tensor(prompt, dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=output_len,
        uid=1,
        sampling_params=SamplingParams(max_tokens=output_len, reasoning_budget=budget),
        cache_handle=_Handle(),
    )


class _Batch:
    def __init__(self, *reqs):
        self.reqs = list(reqs)


def test_budget_resolution_request_overrides_server_default():
    s = _sched(default=50)
    assert s._reasoning_budget(_req([1])) == 50
    assert s._reasoning_budget(_req([1], budget=7)) == 7
    assert s._reasoning_budget(_req([1], budget=0)) == 0  # a request may switch it off
    assert _sched(default=0)._reasoning_budget(_req([1], budget=None)) == 0
    # a tokenizer without a single </think> token: nothing to force, so no budget
    assert _sched(default=50, end_id=None)._reasoning_budget(_req([1])) == 0


def test_prompt_opener_starts_the_span_and_disabled_thinking_does_not():
    s = _sched(default=3)
    opened = _req([1, 2, THINK, 3])  # the chat template's "<think>\n" opener
    assert s._reasoning_init(opened) and opened.reasoning_open is True
    closed = _req([1, THINK, END, 4])  # enable_thinking=False: "<think>\n\n</think>\n\n"
    assert s._reasoning_init(closed) and closed.reasoning_open is False
    plain = _req([1, 2, 3])
    assert s._reasoning_init(plain) and plain.reasoning_open is False


def test_tokens_inside_the_span_count_and_the_end_is_forced_once():
    s = _sched(default=3)
    req = _req([1, THINK])
    b = _Batch(req)
    s._reasoning_force(b)
    assert req.force_token == -1
    for tok in (10, 11):
        req.append_host(torch.tensor([tok], dtype=torch.int32))
        s._track_reasoning(req, (tok,))
        s._reasoning_force(b)
        assert req.force_token == -1, req.reasoning_tokens
    req.append_host(torch.tensor([12], dtype=torch.int32))
    s._track_reasoning(req, (12,))
    assert req.reasoning_tokens == 3
    s._reasoning_force(b)
    assert req.force_token == END  # budget spent: the next sample is </think>
    # the forced token comes back and closes the span; no further forcing
    req.append_host(torch.tensor([END], dtype=torch.int32))
    s._track_reasoning(req, (END,))
    assert req.reasoning_open is False
    s._reasoning_force(b)
    assert req.force_token == -1
    req.append_host(torch.tensor([13], dtype=torch.int32))
    s._track_reasoning(req, (13,))
    s._reasoning_force(b)
    assert req.force_token == -1


def test_a_model_that_opens_think_itself_is_budgeted_from_that_token():
    s = _sched(default=2)
    req = _req([1, 2, 3])  # no opener in the prompt
    s._track_reasoning(req, (THINK,))
    assert req.reasoning_open is True and req.reasoning_tokens == 0
    s._track_reasoning(req, (20, 21))  # a verify window's accepted tokens, at once
    s._reasoning_force(_Batch(req))
    assert req.force_token == END


def test_no_budget_means_no_bookkeeping():
    s = _sched(default=0)
    req = _req([1, THINK])
    s._track_reasoning(req, (5, 6, 7))
    assert req.reasoning_open is None and req.reasoning_tokens == 0
    s._reasoning_force(_Batch(req))
    assert req.force_token == -1


def test_resolve_sampling_carries_the_field_and_rejects_negative():
    import pytest

    from freetoken.server.generation import resolve_sampling

    p = resolve_sampling(temperature=None, top_k=None, top_p=None, max_tokens=None, ignore_eos=False,
                         model_sampling={}, reasoning_budget=4000)
    assert p.reasoning_budget == 4000
    assert resolve_sampling(temperature=None, top_k=None, top_p=None, max_tokens=None, ignore_eos=False,
                            model_sampling={}).reasoning_budget is None
    with pytest.raises(ValueError):
        resolve_sampling(temperature=None, top_k=None, top_p=None, max_tokens=None, ignore_eos=False,
                         model_sampling={}, reasoning_budget=-1)
