"""``BatchSamplingArgs.force_token``: a row the scheduler's reasoning budget has marked returns
its token from ``Sampler.sample`` whatever the logits say, in a greedy batch, a sampled batch
and an expanded verify window."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import SamplingParams
from freetoken.engine.sample import BatchSamplingArgs, Sampler

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _batch(forces, temps):
    reqs = [
        SimpleNamespace(sampling_params=SamplingParams(temperature=t), force_token=f)
        for f, t in zip(forces, temps)
    ]
    return SimpleNamespace(reqs=reqs)


def test_greedy_batch_forces_only_the_marked_rows():
    dev = torch.device("cuda")
    sampler = Sampler(device=dev, vocab_size=64)
    args = sampler.prepare(_batch([-1, 7, -1], [0.0, 0.0, 0.0]))
    assert args.temperatures is None and args.force_token is not None
    logits = torch.zeros(3, 64, device=dev)
    logits[:, 5] = 10.0  # argmax 5 everywhere
    out = sampler.sample(logits, args)
    assert out.tolist() == [5, 7, 5]


def test_sampled_batch_forces_the_marked_row():
    dev = torch.device("cuda")
    sampler = Sampler(device=dev, vocab_size=64)
    args = sampler.prepare(_batch([-1, 9], [1.0, 1.0]))
    assert args.temperatures is not None and args.force_token is not None
    logits = torch.full((2, 64), -1e4, device=dev)
    logits[:, 3] = 0.0  # the only possible sample is 3
    out = sampler.sample(logits, args)
    assert out.tolist() == [3, 9]


def test_unforced_batch_builds_no_tensor():
    sampler = Sampler(device=torch.device("cuda"), vocab_size=64)
    assert sampler.prepare(_batch([-1, -1], [0.0, 0.0])).force_token is None
    assert sampler.prepare(_batch([-1], [1.0])).force_token is None


def test_verify_window_expansion_carries_the_force():
    from freetoken.engine.engine import _expand_sampling_args

    dev = torch.device("cuda")
    sampler = Sampler(device=dev, vocab_size=64)
    args = sampler.prepare(_batch([11], [0.0]))
    rows = 6
    ex = _expand_sampling_args(args, rows)
    assert ex.force_token is not None and ex.force_token.numel() == rows
    logits = torch.zeros(rows, 64, device=dev)
    logits[:, 2] = 1.0
    assert sampler.sample(logits, ex).tolist() == [11] * rows
    # unforced greedy stays the bare args
    plain = sampler.prepare(_batch([-1], [0.0]))
    assert _expand_sampling_args(plain, rows) is plain
