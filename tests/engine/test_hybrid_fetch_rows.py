"""--moe-hybrid-max-fetch auto under the version-6 profile (upstream #734), the fork's part: a layer kind benched with
another CPU worker count stands in for the one the server runs (upstream keeps a fixed cap of 1 there)."""

from __future__ import annotations

from freetoken.engine.engine import _nearest_thread_row
from freetoken.moe.bench_profile import layer_kind_key


def test_the_nearest_thread_count_stands_in():
    rows = {
        layer_kind_key("nvfp4", {}, 2560, 640, 10, 12): {"cpu_moe_gbs": 44.0, "t": 12},
        layer_kind_key("nvfp4", {}, 2560, 640, 10, 6): {"cpu_moe_gbs": 30.0, "t": 6},
        layer_kind_key("nvfp4", {}, 2560, 640, 8, 7): {"cpu_moe_gbs": 1.0, "t": "other top_k"},
        layer_kind_key("q2_0", {}, 2560, 640, 10, 7): {"cpu_moe_gbs": 1.0, "t": "other format"},
        "nvfp4:2560x640:top10:tx": "garbage",
    }
    want = layer_kind_key("nvfp4", {}, 2560, 640, 10, 7)
    assert want not in rows
    row, threads = _nearest_thread_row(rows, want, 7)
    assert (row["t"], threads) == (6, 6)
    row, threads = _nearest_thread_row(rows, want, 11)
    assert (row["t"], threads) == (12, 12)
    # a tie goes to the smaller count (fewer threads undercount the bandwidth, which errs toward fetching less)
    row, threads = _nearest_thread_row(rows, want, 9)
    assert (row["t"], threads) == (6, 6)
    assert _nearest_thread_row(rows, layer_kind_key("bf16", {}, 2560, 640, 10, 7), 7) == (None, None)
