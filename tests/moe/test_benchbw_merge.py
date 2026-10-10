"""``ft bench bw`` writes one profile per GPU; a run that measures one format must keep the
others' entries (the hybrid fetch split of every other model reads them)."""

from __future__ import annotations

from freetoken.moe.benchbw import merge_profile


def _profile(uuid, dtypes, workloads=None, version=4):
    return {
        "version": version,
        "gpu": {"index": 0, "name": "x", "uuid": uuid},
        "dtypes": {f: "hybrid" for f in dtypes},
        "dtype_kernels": {f: {"cpu_moe_gbs": 40.0, "pcie_gather_gbs": 10.0, "fmt": f} for f in dtypes},
        "workloads": workloads or {},
    }


def test_merge_keeps_the_formats_this_run_did_not_measure():
    old = _profile("GPU-1", ["nvfp4", "mxfp4_triton"], {"flash-next": {"kernels": {"nvfp4": {}}}})
    new = _profile("GPU-1", ["q2_0"])
    merged = merge_profile(old, new)
    assert set(merged["dtype_kernels"]) == {"nvfp4", "mxfp4_triton", "q2_0"}
    assert merged["dtype_kernels"]["q2_0"]["fmt"] == "q2_0"
    assert set(merged["dtypes"]) == {"nvfp4", "mxfp4_triton", "q2_0"}
    assert "flash-next" in merged["workloads"]
    assert merged["carried_over"] == ["dtype_kernels:mxfp4_triton", "dtype_kernels:nvfp4", "workloads:flash-next"]


def test_merge_prefers_the_new_measurement_and_ignores_other_gpus_or_versions():
    old = _profile("GPU-1", ["nvfp4"])
    old["dtype_kernels"]["nvfp4"]["cpu_moe_gbs"] = 1.0
    new = _profile("GPU-1", ["nvfp4"])
    assert merge_profile(old, new)["dtype_kernels"]["nvfp4"]["cpu_moe_gbs"] == 40.0
    assert "carried_over" not in merge_profile(old, new)
    assert merge_profile(_profile("GPU-2", ["nvfp4"]), new) == new
    assert merge_profile(_profile("GPU-1", ["nvfp4"], version=3), new) == new
    assert merge_profile(None, new) == new
