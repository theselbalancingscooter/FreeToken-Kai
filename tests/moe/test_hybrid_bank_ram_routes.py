"""--moe-bank-ram under hybrid decode: routes to rows past the registered prefix go to the CPU.

The hybrid ensure kernel decides which misses to fetch and, since upstream #685, also splits the
routes. A row past ``prefix_pinned_rows`` is not addressable from the GPU, so ``hybrid_routes``
hands the kernel expert 0 for it and turns what comes back into a CPU route with the true id.
The kernel is replaced here by its contract (a resident set, the rest to the CPU)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.layers.moe import hybrid_routes


def _cache(resident: dict[int, int], prefix_pinned_rows):
    """A cache whose ensure kernel maps the resident experts to their slots and the rest to the CPU."""
    seen = {}

    def ensure_experts_hybrid(layer_id, expert_ids, weights, clamp=False):
        seen["ids"] = expert_ids.clone()
        slots = torch.tensor([resident.get(int(e), -1) for e in expert_ids.view(-1)], dtype=expert_ids.dtype)
        slots = slots.view_as(expert_ids)
        on_gpu = slots >= 0
        cpu_ids = torch.where(on_gpu, torch.full_like(expert_ids, -1), expert_ids)
        gpu_w = torch.where(on_gpu, weights, torch.zeros_like(weights))
        expert_ids.copy_(slots.clamp_min(0) if clamp else slots)
        return expert_ids, gpu_w, cpu_ids

    return SimpleNamespace(prefix_pinned_rows=prefix_pinned_rows, ensure_experts_hybrid=ensure_experts_hybrid), seen


IDS = [[0, 3, 9, 12], [5, 0, 11, 2]]
W = [[0.4, 0.3, 0.2, 0.1], [0.25, 0.25, 0.25, 0.25]]
RESIDENT = {0: 7, 3: 1, 5: 4, 9: 6}  # expert 9 is cached though past the prefix (a stale slot)


@pytest.mark.parametrize("clamp", [False, True])
def test_rows_past_the_prefix_become_cpu_routes_with_their_ids(clamp):
    cache, seen = _cache(RESIDENT, prefix_pinned_rows=8)
    ids, w = torch.tensor(IDS, dtype=torch.int32), torch.tensor(W)
    gpu_slots, gpu_w, cpu_ids = hybrid_routes(cache, 0, ids, w, clamp)

    cold = torch.tensor(IDS) >= 8
    # the kernel saw expert 0 where the row is past the prefix: it costs no fetch
    assert torch.equal(seen["ids"], torch.where(cold, 0, torch.tensor(IDS)).to(torch.int32))
    # those routes are the CPU's, with the true ids -- even expert 9, which a slot still held
    assert cpu_ids[cold].tolist() == [9, 12, 11]
    assert gpu_w[cold].abs().sum() == 0
    assert set(gpu_slots[cold].tolist()) == {0 if clamp else -1}
    # the rest is the kernel's split, untouched: 0, 3, 5 resident (GPU), 2 a miss left to the CPU
    assert gpu_slots[~cold].tolist() == ([7, 1, 4, 7, 0] if clamp else [7, 1, 4, 7, -1])
    assert cpu_ids[~cold].tolist() == [-1, -1, -1, -1, 2]
    assert torch.equal(gpu_w[~cold], torch.tensor([0.4, 0.3, 0.25, 0.25, 0.0]))
    # every route is computed exactly once: on the GPU (weight) or on the CPU (id)
    assert ((gpu_w > 0) ^ (cpu_ids >= 0)).all()


def test_without_a_mapped_bank_the_kernel_split_is_returned_as_it_is():
    cache, seen = _cache(RESIDENT, prefix_pinned_rows=None)
    ids, w = torch.tensor(IDS, dtype=torch.int32), torch.tensor(W)
    gpu_slots, gpu_w, cpu_ids = hybrid_routes(cache, 0, ids, w, clamp=False)
    assert torch.equal(seen["ids"], torch.tensor(IDS, dtype=torch.int32))
    assert gpu_slots.tolist() == [[7, 1, 6, -1], [4, 7, -1, -1]]
    assert cpu_ids.tolist() == [[-1, -1, -1, 12], [-1, -1, 11, 2]]
