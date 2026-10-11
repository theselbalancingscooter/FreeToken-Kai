"""Host-side arithmetic of MTP speculative decoding (torch-light, unit-testable).

One iteration verifies a window ``[t_last, d_1, ..., d_k]`` (k drafts) in a single extend
forward, samples the target at every row, keeps the longest prefix of drafts the samples
confirm, and rolls the per-request state back to the last accepted row. The pieces here
are the pure functions that decide and describe that; the GPU work lives in the engine,
the GDN layer and the model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Sequence

import torch


@dataclass
class SpecResult:
    """What one forward produced for the (single) request: the committed tokens (>= 1; the
    last one is the target's own sample) and the MTP drafts for the next window."""

    accepted: list[int]
    drafts: list[int] = field(default_factory=list)


def accept_drafts(sampled: Sequence[int], drafts: Sequence[int]) -> list[int]:
    """Verification rule: row j of the window sampled ``sampled[j]``; draft ``drafts[j]`` (the
    input of row j+1) is accepted iff it equals ``sampled[j]``. Returns ``sampled[:m+1]`` for
    ``m`` accepted drafts -- the accepted drafts themselves plus the sample after them.

    With a deterministic (greedy) draft this is exact rejection sampling: accepting when the
    target's own sample equals the draft happens with probability p(d), and the sample that
    ends the run is drawn from the residual distribution."""
    m = 0
    while m < len(drafts) and m < len(sampled) - 1 and sampled[m] == drafts[m]:
        m += 1
    return [int(t) for t in sampled[: m + 1]]


def cut_drafts(drafts: Sequence[int], probs: Sequence[float], min_p: float) -> list[int]:
    """--spec-min-p: keep the drafts up to (not including) the first one the head scored below
    ``min_p``; everything after a doubtful draft would be verified against a token that is
    probably wrong anyway. ``min_p <= 0`` keeps them all."""
    out = []
    for d, p in zip(drafts, probs):
        if min_p > 0 and p < min_p:
            break
        out.append(int(d))
    return out


def pack_spec_message(result: SpecResult, k: int) -> torch.Tensor:
    """Fixed-length int32 vector ``[a, tok_0..tok_k (pad -1), d_1..d_k (pad -1)]`` for the
    pipeline ranks (same size every step, so the receiver needs no header)."""
    a = len(result.accepted)
    assert 1 <= a <= k + 1, (a, k)
    toks = list(result.accepted) + [-1] * (k + 1 - a)
    drafts = list(result.drafts)[:k] + [-1] * (k - min(len(result.drafts), k))
    return torch.tensor([a, *toks, *drafts], dtype=torch.int32)


def unpack_spec_message(vec: torch.Tensor, k: int) -> SpecResult:
    v = vec.tolist()
    a = int(v[0])
    toks = [int(t) for t in v[1 : 1 + a]]
    drafts = [int(d) for d in v[2 + k : 2 + 2 * k] if d >= 0]
    return SpecResult(accepted=toks, drafts=drafts)


def spec_message_len(k: int) -> int:
    return 2 * k + 2


def pages_to_free(keep_len: int, alloc_len: int, page_size: int) -> tuple[int, int]:
    """Pages ``[first, last)`` that only hold positions ``>= keep_len`` once a request whose
    pages were allocated up to ``alloc_len`` rolls back; the page holding position
    ``keep_len - 1`` stays."""
    first = -(-keep_len // page_size)
    last = -(-alloc_len // page_size)
    return first, max(first, last)


def rebuild_conv_state(prev_state: torch.Tensor, conv_in: torch.Tensor, accepted: int) -> torch.Tensor:
    """Conv left-context after ``accepted`` of the window's tokens: ``prev_state [dim, K-1]``
    (state before the window) followed by the first ``accepted`` conv inputs ``conv_in [T, dim]``,
    keeping the last ``K-1`` columns."""
    width = prev_state.shape[-1]
    cat = torch.cat([prev_state, conv_in[:accepted].to(prev_state.dtype).transpose(0, 1)], dim=-1)
    return cat[..., -width:].contiguous()


# FREETOKEN_SPEC_ROLLBACK: how the GDN states are rolled back after a verify window.
#   graph (default): the per-layer restore, captured once per (window buffers, accepted count)
#     into a CUDA graph and replayed. The restore is ~110 tiny launches per rank (18 layers x 6)
#     and all of its 1.5 ms per window was host issue time (the GPU work is 0.2 ms); the verify
#     window's stash tensors are the graph's static buffers, so the same restore recurs every
#     step and a replay is one launch. A window whose buffers are new (the eager, uncaptured
#     path) is restored eagerly; a graph is captured the second time a buffer set is seen.
#   loop: the per-layer restore, eagerly.
#   batched: one stacked restore (fewer launches, but its slot-equality checks are device
#     syncs, and on an RTX 2060 it measured no faster than the loop).
_ROLLBACK_MODE = os.environ.get("FREETOKEN_SPEC_ROLLBACK", "graph")
_ROLLBACK_GRAPHS: dict = {}
_ROLLBACK_SEEN: dict = {}


def _rollback_key(stashes, accepted: int):
    return (accepted, tuple(
        (s.li, id(s.pool), s.slot.data_ptr(), s.prev_conv.data_ptr(), s.conv_in.data_ptr(), s.ht.data_ptr(), tuple(s.ht.shape))
        for s in stashes))


def _restore_graphed(stashes, accepted: int) -> None:
    key = _rollback_key(stashes, accepted)
    g = _ROLLBACK_GRAPHS.get(key)
    if g is not None:
        g.replay()
        return
    seen = _ROLLBACK_SEEN.get(key, 0) + 1
    if len(_ROLLBACK_SEEN) > 256:  # eager windows bring fresh buffers every step; do not grow forever
        _ROLLBACK_SEEN.clear()
    _ROLLBACK_SEEN[key] = seen
    if seen < 2 or not stashes[0].ht.is_cuda:
        for s in stashes:
            s.restore(accepted)
        return
    # the second time with these buffers: do the restore on a side stream (this step's effect),
    # then capture the same ops into a graph for the steps after
    cur = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    side.wait_stream(cur)
    with torch.cuda.stream(side):
        for s in stashes:
            s.restore(accepted)
    cur.wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for s in stashes:
            s.restore(accepted)
    _ROLLBACK_GRAPHS[key] = g


def restore_gdn_stashes(stashes, accepted: int) -> None:
    """Roll every GDN layer's state back to ``accepted`` rows of the verify window in a handful of
    kernels instead of three per layer. All stashes of one window share the pool and the slot
    (one request), so the conv states are rebuilt as one ``[L, dim, K-1]`` tensor and written
    with one indexed store, and the recurrent states likewise; a window whose stashes do not
    line up (several slots, several pools) falls back to the per-layer restore. On two RTX
    3060s the per-layer loop was 1.7-1.8 ms of host time per rank per window (18 layers, 54
    small launches plus Python)."""
    if not stashes:
        return
    if _ROLLBACK_MODE == "graph":
        _restore_graphed(stashes, accepted)
        return
    if _ROLLBACK_MODE != "batched":  # the per-layer restore, eagerly
        for s in stashes:
            s.restore(accepted)
        return
    first = stashes[0]
    pool = first.pool
    same = all(
        s.pool is pool and s.slot.shape == first.slot.shape and bool((s.slot == first.slot).all())
        and s.prev_conv.shape == first.prev_conv.shape and s.ht.shape == first.ht.shape
        for s in stashes[1:]
    )
    if not same:
        for s in stashes:
            s.restore(accepted)
        return
    slot = first.slot.long()  # [1]
    layers = torch.tensor([s.li for s in stashes], dtype=torch.long, device=slot.device)
    width = first.prev_conv.shape[-1]
    conv_dtype = pool.conv_states.dtype
    prev = torch.stack([s.prev_conv[0] for s in stashes])  # [L, dim, K-1]
    taken = torch.stack([s.conv_in[:accepted] for s in stashes]).to(conv_dtype).transpose(1, 2)  # [L, dim, accepted]
    conv = torch.cat([prev.to(conv_dtype), taken], dim=-1)[..., -width:]
    pool.conv_states[layers, slot] = conv
    rec_dtype = pool.recurrent_states.dtype
    ht = torch.stack([s.ht[accepted - 1] for s in stashes]).to(rec_dtype)  # [L, HV, V, K]
    pool.recurrent_states[layers, slot] = ht.reshape(len(stashes), *pool.recurrent_states.shape[2:])


def ngram_context_after(host_ids: Sequence[int], drafts: Sequence[int], accepted: int, ctx_len: int, boundary: int) -> list[int]:
    """PLE n-gram context (the last ``ctx_len`` processed ids) after ``accepted`` rows of the
    window ``[host_ids[-1], drafts...]`` were kept. ``host_ids`` is the request's committed
    id list before the window (its last id is the window's first input)."""
    seq = list(host_ids) + list(drafts)
    end = len(host_ids) - 1 + accepted  # rows 0..accepted-1 = seq[len-1 : len-1+accepted]
    window = seq[max(0, end - ctx_len) : end]
    return [boundary] * (ctx_len - len(window)) + [int(t) for t in window]


__all__ = [
    "SpecResult",
    "accept_drafts",
    "pack_spec_message",
    "unpack_spec_message",
    "spec_message_len",
    "pages_to_free",
    "rebuild_conv_state",
    "restore_gdn_stashes",
    "cut_drafts",
    "ngram_context_after",
]
