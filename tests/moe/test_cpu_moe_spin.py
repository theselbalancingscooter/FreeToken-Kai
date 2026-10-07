"""The CPU MoE pool's spin-wait gate (upstream #619, with Kai's --pp-size rule)."""

from types import SimpleNamespace

import pytest

import freetoken.distributed as dist
import freetoken.moe.cpu_executor as ce


@pytest.fixture
def affinity(monkeypatch):
    monkeypatch.setattr(ce.os, "sched_getaffinity", lambda pid: set(range(16)), raising=False)
    monkeypatch.setattr(dist, "try_get_pp_info", lambda: None)


def _env(monkeypatch, value):
    monkeypatch.setattr(ce, "_SPIN_WAIT_ENV", value)
    monkeypatch.setattr(ce, "_SPIN_WAIT", value != "0")


def test_one_rank_spins_when_the_pool_owns_its_cpus_and_two_stay_spare(affinity, monkeypatch):
    _env(monkeypatch, None)
    assert ce._spin_wait_ok(list(range(11)), 11)
    assert not ce._spin_wait_ok(list(range(14)), 14)  # one spare CPU
    assert not ce._spin_wait_ok([0, 0, 1], 2)  # a shared CPU


def test_pipeline_ranks_do_not_spin_unless_forced(affinity, monkeypatch):
    monkeypatch.setattr(dist, "try_get_pp_info", lambda: SimpleNamespace(rank=0, size=2))
    _env(monkeypatch, None)
    assert not ce._spin_wait_ok(list(range(7)), 7)
    _env(monkeypatch, "1")
    assert ce._spin_wait_ok(list(range(7)), 7)
    _env(monkeypatch, "0")
    assert not ce._spin_wait_ok(list(range(7)), 7)
    # a pipeline of one rank is a single process
    monkeypatch.setattr(dist, "try_get_pp_info", lambda: SimpleNamespace(rank=0, size=1))
    _env(monkeypatch, None)
    assert ce._spin_wait_ok(list(range(7)), 7)
