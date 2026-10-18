"""The console's benchmark: the search over flags (webui/search.py), the job that runs it
(webui/tuner.py) and the decisions in the hardware bench (webui/hwbench.py).

The job runs against fakes: a manager that records lifecycle calls, a child process that prints
canned JSON lines, and a serve whose speed follows the flags it was started with. What is pinned:
the previous engine comes back, each candidate is one change on the best so far and is kept only on
the measurement, candidates that undo each other are not both tried, a failure is a run not kept,
generation on code is measured where MTP is in play and the chosen use decides."""

from __future__ import annotations

import io
import json
import os

import pytest

from freetoken.webui import hwbench, search, tuner

GiB = 1 << 30


# ------------------------------------------------------------------ flags and decisions
def test_set_flags_overrides_adds_and_drops_in_place():
    args = ["--dtype", "float16", "--moe-cache-auto", "--moe-cpu-threads", "8", "--host", "0.0.0.0"]
    out = tuner.set_flags(args, {"--moe-cpu-threads": "6", "--kv-reserve-tokens": "65536", "--moe-collect-stats": None},
                          drop=("--dtype",))
    assert out == ["--moe-cache-auto", "--moe-cpu-threads", "6", "--host", "0.0.0.0",
                   "--kv-reserve-tokens", "65536", "--moe-collect-stats"]
    assert tuner.parse_flags(["--moe-hybrid-max-fetch", "-1", "--x=3"]) == [("--moe-hybrid-max-fetch", "-1"), ("--x", "3")]


def test_quant_backend_entries_join_instead_of_replacing():
    args = ["--quant-backend", "moe.nvfp4=marlin"]
    assert search.merge_changes(args, {"--quant-backend": "linear.nvfp4=triton"}) == {"--quant-backend": "linear.nvfp4=triton,moe.nvfp4=marlin"}
    assert search.merge_changes(args, {"--quant-backend": "moe.nvfp4=b12x"}) == {"--quant-backend": "moe.nvfp4=b12x"}
    assert search.merge_changes(args, {"--spec-mtp": "3"}) == {"--spec-mtp": "3"}


def _t(prefill, decode, code=None, ok=True):
    return {"ok": ok, "prefill_tps": prefill, "decode_tps": decode, "decode_code_tps": code}


def test_keep_by_goal_and_use():
    base = _t(600, 30.0)
    gen = search.Candidate("x", {"--a": "1"})
    assert tuner.keep(base, _t(560, 31.0), gen, "prose")          # +3.3% generation, prefill holds 90%
    assert not tuner.keep(base, _t(500, 31.0), gen, "prose")      # prefill fell below 90%
    pre = search.Candidate("y", {"--b": "1"}, goal="prefill", gain=1.05)
    assert tuner.keep(base, _t(640, 29.2), pre, "prose") and not tuner.keep(base, _t(640, 28.0), pre, "prose")
    either = search.Candidate("z", {"--c": "1"}, goal="either")
    assert tuner.keep(base, _t(580, 31.2), either, "prose") and tuner.keep(base, _t(700, 29.5), either, "prose")
    assert not tuner.keep(base, _t(600, 30.0, ok=False), either, "prose")
    # MTP: faster on code, slower on prose -- the use decides
    mtp = search.Candidate("m", {"--spec-mtp": "3"}, code=True, hold_prefill=0.85)
    drafted, base_both = _t(560, 26.0, code=45.0), _t(600, 30.0, code=30.0)
    assert tuner.keep(base_both, drafted, mtp, "code") and not tuner.keep(base_both, drafted, mtp, "prose")
    assert tuner.keep(base_both, drafted, mtp, "both")  # sqrt(26*45)=34.2 > 30.9
    assert not tuner.keep(base, drafted, mtp, "code")  # no code figure for the base: prose against prose
    # a base measured on code and prose against a run measured on prose only: prose against prose
    base_code = _t(260, 18.18, code=18.34)
    assert tuner.keep(base_code, _t(569, 17.75), gen, "code")  # the 3060's 16-bit KV cache
    assert not tuner.close_on_generation(base_code, _t(260, 17.9), gen, "code")  # 98% of prose: no swing to check
    # tried for generation, kept for prompt processing: a 16-bit KV cache on two 3060s went 251 -> 517
    assert tuner.keep(_t(251, 18.4), _t(517, 18.4), gen, "prose")
    assert not tuner.keep(_t(251, 18.4), _t(517, 17.0), gen, "prose")  # but not at 8% of generation


def _w(prefill, prose, code, short, prose_runs, code_runs):
    """A trial measured under an MTP window: 600-token runs, the median, every run on record."""
    return {"ok": True, "prefill_tps": prefill, "decode_tps": prose, "decode_code_tps": code, "short_s": short,
            "gen_tokens": tuner.DECODE_TOKENS_MTP, "decode_runs": prose_runs, "decode_code_runs": code_runs}


def test_under_a_window_a_generation_difference_inside_the_swing_decides_nothing():
    # two RTX 3060s, Flash-Next with the Q2_0 experts, the night of 2026-10-12: the figures the benchmark saw
    base = _w(698.0, 33.99, 40.58, 1.62, [36.0, 28.1, 34.0], [40.9, 39.3, 40.6])
    budget = search.Candidate("budget_075", {"--prefill-chunk-budget": "0.75"}, goal="prefill", gain=1.05)
    for run in (_w(919.9, 32.58, 39.92, 1.74, [31.8, 32.6, 32.8], [36.8, 39.9, 41.0]),
                _w(915.5, 29.89, 37.49, 1.74, [29.9, 30.1, 27.9], [38.1, 37.5, 36.7]),
                _w(914.8, 32.40, 38.08, 1.65, [34.0, 31.0, 32.4], [37.9, 42.3, 38.1])):
        # +31% of prompt processing, generation 3-10% lower as the same settings swing: turned down three times then
        assert tuner.keep(base, run, budget, "both")
    assert tuner.gen_band(base, base) == tuner.GEN_BAND_MAX  # 36.0 / 28.1 / 34.0 is wider than the cap
    # changes that moved nothing are still not kept
    threads = search.Candidate("threads_fewer", {"--moe-cpu-threads": "7"})
    assert not tuner.keep(base, _w(685.5, 30.60, 37.56, 1.75, [31.7, 28.7, 30.6], [37.6, 40.3, 34.0]), threads, "both")
    kv = search.Candidate("kv_q8", {"--kv-cache-dtype": "q8_0"}, group="kv")
    assert not tuner.keep(base, _w(248.2, 30.32, 37.28, 2.70, [29.8, 30.3, 30.9], [37.3, 37.3, 37.1]), kv, "both")
    # the checkpoint's experts the same night: min-p over a 5-token window read 3-5% faster on one start and
    # cost 17% of prompt processing -- a gain inside the swing, paid for with more than it gains
    nv = _w(689.9, 22.76, 29.18, 2.93, [23.2, 22.8, 22.2], [29.2, 31.7, 27.8])
    minp = search.Candidate("minp_06", {"--spec-min-p": "0.6", "--spec-mtp": "5"}, code=True, hold_prefill=0.85)
    assert not tuner.keep(nv, _w(570.7, 23.42, 30.74, 2.91, [24.9, 23.4, 22.9], [26.7, 33.2, 30.7]), minp, "both")
    # a window where there was none is far outside any swing (16.9 -> 28.4 tok/s on seven answered prompts)
    plain = {"ok": True, "prefill_tps": 645.0, "decode_tps": 16.9, "decode_code_tps": 17.0, "short_s": 2.9, "gen_tokens": tuner.DECODE_TOKENS,
             "decode_runs": [16.8, 16.9, 16.7], "decode_code_runs": [17.0, 16.9, 16.8]}
    mtp = search.Candidate("mtp_3", {"--spec-mtp": "3"}, code=True, hold_prefill=0.85)
    windowed = _w(662.0, 28.4, 29.0, 2.9, [28.6, 28.4, 28.1], [29.2, 29.0, 28.7])
    assert tuner.keep(plain, windowed, mtp, "both") and tuner.gen_band(plain, windowed) == tuner.GEN_BAND_MIN
    # plain decoding is judged as it always was
    assert tuner.gen_band(_t(600, 30.0), _t(640, 29.2)) == 0.0 and tuner.gen_band(plain, plain) == 0.0


def test_the_short_prompt_must_hold_and_can_win_alone():
    base = dict(_t(600, 30.0), short_s=3.0)
    gen = search.Candidate("x", {"--a": "1"})
    # faster on the long prompt and in generation, but a chat turn waits a third longer: not kept
    assert not tuner.keep(base, dict(_t(700, 32.0), short_s=4.0), gen, "prose")
    # within the swing seen between runs, the other figures decide as before
    assert tuner.keep(base, dict(_t(620, 31.0), short_s=3.5), gen, "prose")
    # the short prompt alone gained enough, the rest held
    assert tuner.keep(base, dict(_t(600, 30.0), short_s=2.4), gen, "prose")
    assert not tuner.keep(base, dict(_t(500, 30.0), short_s=2.4), gen, "prose")  # prompt processing fell
    # a result saved before the short prompt was measured is compared on the other figures only
    pre = search.Candidate("y", {"--b": "1"}, goal="prefill", gain=1.05)
    assert tuner.keep(_t(600, 30.0), dict(_t(640, 29.2), short_s=9.0), pre, "prose")
    # two runs of the same settings: the slower short prompt is the one that counts
    assert tuner.slower_of(dict(_t(600, 30.0), short_s=2.0), dict(_t(620, 29.0), short_s=2.5))["short_s"] == 2.5


def _geo(now=16384, kv=10880, slots=1104, per_expert=1775616, kv_max=249909):
    return {"num_pages": now, "page_size": 1, "moe_cache_size": slots,
            "unit_bytes": {"kv_per_token": kv, "moe_per_expert": per_expert},
            "limits": {"kv_tokens": {"max": kv_max}}}


def test_context_candidates_stop_at_the_model_and_the_budget():
    assert tuner.context_candidates(_geo(), 262144) == [32768, 65536, 131072]  # 262144 > budget max
    assert tuner.context_candidates(_geo(), 32768) == [32768]
    assert tuner.context_candidates(_geo(now=65536), 262144) == [131072]


def test_longer_context_is_still_answered_for_old_callers():
    c = tuner.longer_context(_geo(), [{"gpu": {"free_bytes": int(2.5 * GiB)}}], cache_auto=False, model_max=262144)
    assert c["tokens"] == 131072


def test_corpus_is_this_repository():
    corpus = tuner.load_corpus()
    assert "FreeToken" in corpus["prose"] and "def " in corpus["code"]
    import random
    part = tuner.excerpt(corpus["code"], 1000, random.Random(1))
    assert len(part) == 1000
    assert tuner.load_corpus("/nonexistent")["prose"]  # a wheel install still has something to read


# ------------------------------------------------------------------ the hardware bench's decisions
def test_threads_are_the_knee_not_the_maximum():
    assert hwbench.pick_threads({1: 3.0, 2: 5.8, 4: 9.9, 6: 11.6, 8: 11.9, 12: 12.1}) == 6
    assert hwbench.thread_candidates(12) == [1, 2, 4, 6, 8, 10, 12]
    assert hwbench.thread_candidates(3) == [1, 2, 3]


def test_expert_formats():
    assert hwbench.expert_format("modelopt") == "nvfp4"
    assert hwbench.expert_format("mxfp4") == "mxfp4_triton"
    assert hwbench.expert_format("fp8") == "fp8_block"
    assert hwbench.expert_format(None) == "bf16"


def _hw(cpu_best, gather):
    return {"cpu_moe": {"best_gbs": cpu_best, "threads": 6}, "gather": {"0": {"gbs": gather}},
            "overlap": {"cpu_gbs": 6.0, "pcie_gbs": 3.0, "fetch_fraction": 0.333}}


def test_derive_hybrid_when_the_cpu_is_twice_the_link():
    notes = {n["flag"]: n for n in hwbench.derive(_hw(12.0, 5.0))}
    assert notes["--moe-strategy"]["value"] == "hybrid" and notes["--moe-cpu-threads"]["value"] == "6"
    assert all(n["why"] and n["why_en"] for n in notes.values())


def test_derive_offload_when_the_link_keeps_up():
    notes = {n["flag"]: n for n in hwbench.derive(_hw(12.0, 7.0))}
    assert notes["--moe-strategy"]["value"] == "offload" and "--moe-cpu-threads" not in notes


def test_bench_profiles_say_which_version_measured_them(tmp_path, monkeypatch):
    import os

    from freetoken.moe.bench_profile import PROFILE_VERSION

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    d = tmp_path / "freetoken" / "benchbw"
    os.makedirs(d)
    for uuid, extra in (("GPU-a", {}), ("GPU-b", {"freetoken_version": "0.0.1"})):
        with open(d / f"{uuid}.json", "w") as fh:
            json.dump({"version": PROFILE_VERSION, "epoch": 1, "gpu": {"index": 0, "name": "RTX", "uuid": uuid},
                       "layer_kinds": {"nvfp4:2560x640:top10:t7": {"format": "nvfp4", "cpu_moe_gbs": 40.0, "pcie_gather_gbs": 10.0},
                                       "q2_0:2560x640:top10:t7": {"format": "q2_0", "cpu_moe_gbs": 30.0, "pcie_gather_gbs": 10.0}}, **extra}, fh)
    tuner.stamp_profile(str(d / "GPU-a.json"))
    doc = tuner.bench_profiles()
    assert doc["current"] == tuner.FT_VERSION
    assert [(p["uuid"], p["version"]) for p in doc["profiles"]] == [("GPU-a", tuner.FT_VERSION), ("GPU-b", "0.0.1")]
    assert doc["profiles"][0]["formats"] == ["nvfp4", "q2_0"] and doc["profiles"][0]["kinds"] == 2
    from freetoken.moe.bench_profile import fetch_fraction, load_layer_kinds

    rows = load_layer_kinds("RTX", gpu_uuid="GPU-a")  # the stamp does not disturb the reader
    assert fetch_fraction(rows["nvfp4:2560x640:top10:t7"]) == 0.25

# ------------------------------------------------------------------ what gets tried
ORNITH = dict(model_type="qwen3_5_moe", num_experts=256, num_layers=40, quant="modelopt", mtp=True, max_context=262144,
              layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 10,
              ram_total=24 * GiB, weight_bytes=21 * GiB)
ORNITH_ARGS = ["--dtype", "float16", "--moe-strategy", "hybrid", "--moe-cpu-layers", "auto", "--moe-cpu-threads", "8",
               "--moe-cache-auto", "--disable-moe-prefill-overlap", "--kv-cache-dtype", "q8_0", "--kv-reserve-tokens", "16384",
               "--max-seq-len-override", "16384", "--host-embedding", "--prefill-mixer-pieces", "2", "--max-running-req", "1"]
HW_2060 = {"measurements": {"cpu_moe": {"best_gbs": 36.4, "threads": 8, "cores": 12}, "gather": {"0": {"gbs": 10.1}},
                            "pcie": {"0": {"h2d_gbs": 11.2}}}}
FLASH = dict(model_type="qwen4_exp", num_experts=512, num_layers=48, quant="modelopt", mtp=True, max_context=262144,
             layer_types=["linear_attention"] * 36 + ["full_attention"] * 12, ram_total=108 * GiB, weight_bytes=135 * GiB)
FLASH_ARGS = ["--pp-size", "2", "--gpu", "0,1", "--moe-strategy", "hybrid", "--moe-cpu-layers", "auto", "--moe-cpu-threads", "8",
              "--moe-cache-auto", "--kv-cache-dtype", "q8_0", "--kv-reserve-tokens", "65536", "--dense-quant", "fp8",
              "--prefill-mixer-pieces", "2", "--max-running-req", "1"]
HW_3060 = {"measurements": {"cpu_moe": {"best_gbs": 30.3, "threads": 8, "cores": 8}, "gather": {"0": {"gbs": 11.5}, "1": {"gbs": 6.3}},
                            "pcie": {"0": {"h2d_gbs": 23.1}, "1": {"h2d_gbs": 6.3}}}}


ALL_MODULES = {"vllm", "flashinfer"}


def _keys(args, facts, hw, mode, modules=ALL_MODULES):
    return [c.key for c in search.plan(args, facts, hw, mode, modules=modules)]


def test_kernels_that_cannot_run_here_are_not_tried_and_say_why():
    turing = {"measurements": dict(HW_2060["measurements"], gpus=[{"index": 0, "compute_cap": 7.5}])}
    keys = _keys(ORNITH_ARGS, ORNITH, turing, "thorough", modules=set())
    assert "kernel_moe_triton" in keys
    assert not {"kernel_moe_marlin", "kernel_moe_b12x", "kernel_linear_marlin"} & set(keys)
    why = {x["flag"]: x["why_en"] for x in search.unavailable(ORNITH, turing, modules=set())}
    assert "vLLM" in why["--quant-backend moe.nvfp4=marlin"] and "sm_75" in why["--quant-backend moe.nvfp4=b12x"]
    blackwell = {"measurements": {"gpus": [{"index": 0, "compute_cap": 12.0}]}}
    assert "kernel_moe_b12x" in _keys(ORNITH_ARGS, ORNITH, blackwell, "standard")


def test_no_expert_kernel_of_the_checkpoints_format_when_the_experts_come_from_a_gguf():
    # the GGUF's experts are Q2_0: moe.nvfp4=<kernel> selects nothing, so trying it measures only the swing
    plain = _keys(FLASH_ARGS, FLASH, HW_3060, "thorough")
    gguf = _keys(FLASH_ARGS + ["--moe-expert-gguf", "/m/x-Q2_0.gguf"], FLASH, HW_3060, "thorough")
    assert [k for k in plain if k.startswith("kernel_moe_")] and not [k for k in gguf if k.startswith("kernel_moe_")]
    assert [k for k in gguf if not k.startswith("kernel_moe_")] == [k for k in plain if not k.startswith("kernel_moe_")]


def test_standard_plan_on_a_2060_with_ornith():
    keys = _keys(ORNITH_ARGS, ORNITH, HW_2060, "standard")
    assert keys == ["strategy_offload", "threads_all", "threads_fewer", "kernel_moe_triton", "kernel_moe_marlin", "kernel_moe_b12x",
                    "kv_16bit", "overlap_on", "budget_075", "pieces_4", "mtp_3", "mtp_5", "minp_06"]  # no draft head: Ornith's lm_head is NVFP4


def test_thorough_plan_adds_the_rarely_winning_ones():
    standard = set(_keys(ORNITH_ARGS, ORNITH, HW_2060, "standard"))
    thorough = set(_keys(ORNITH_ARGS, ORNITH, HW_2060, "thorough"))
    assert standard < thorough
    assert {"fetch_none", "kv_q4", "hit_d2d", "pieces_1", "max_prefill_16k", "no_host_embedding",
            "kernel_linear_marlin"} <= thorough - standard
    assert "budget_090" not in thorough  # never won, and killed the server with the Q2_0 experts on two RTX 3060s
    assert "mtp_4" not in thorough  # a 4-token window is tried where it was measured (Flash-Next)


def test_two_3060s_with_flash_next():
    keys = _keys(FLASH_ARGS, FLASH, HW_3060, "standard")
    assert "pp_layers_25" in keys  # rank 1 sits on the x4 link: rank 0 takes one more layer
    assert "threads_fewer" in keys and "threads_all" not in keys  # already at every core
    assert "mtp_3" in keys and "mtp_4" in keys and "mtp_5" in keys
    assert "mtp_4" not in _keys(FLASH_ARGS + ["--spec-mtp", "4"], FLASH, HW_3060, "standard")  # the window the settings already have
    assert "ple_pinned" not in keys  # 135 GiB of weights do not leave RAM for the PLE table
    thorough = _keys(FLASH_ARGS, FLASH, HW_3060, "thorough")
    assert {"pp_layers_26", "send_ahead_3", "prefill_group_2", "dense_bf16"} <= set(thorough)


def test_dense_q8_is_standard_without_fp8_tensor_cores_and_thorough_with_them():
    ampere = {"measurements": dict(HW_3060["measurements"], gpus=[{"index": 0, "compute_cap": 8.6}, {"index": 1, "compute_cap": 8.6}])}
    ada = {"measurements": dict(HW_3060["measurements"], gpus=[{"index": 0, "compute_cap": 8.9}])}
    assert "dense_q8" in _keys(FLASH_ARGS, FLASH, ampere, "standard")
    assert "dense_q8" not in _keys(FLASH_ARGS, FLASH, ada, "standard")
    assert "dense_q8" in _keys(FLASH_ARGS, FLASH, ada, "thorough")
    q8_args = [a if a != "fp8" else "q8_0" for a in FLASH_ARGS]
    keys = _keys(q8_args, FLASH, ampere, "thorough")
    assert "dense_fp8" in keys and "dense_bf16" in keys and "dense_q8" not in keys
    # the three undo each other: one group
    groups = {c.key: c.group for c in search.plan(q8_args, FLASH, ampere, "thorough")}
    assert groups["dense_fp8"] == groups["dense_bf16"] == "dense"


def test_min_p_and_the_draft_head_follow_the_window():
    plan = {c.key: c for c in search.plan(FLASH_ARGS, FLASH, HW_3060, "standard")}
    # without a window in the base flags they wait for a kept mtp candidate (any of them)
    assert plan["minp_06"].after == ("mtp_3", "mtp_4", "mtp_5") and plan["minp_06"].opened_by({"mtp_4"}) and not plan["minp_06"].opened_by(set())
    assert "draft_q4" not in plan  # fp8 dense: the 4-bit head packed from an fp8 head loses acceptance
    q8_args = [a if a != "fp8" else "q8_0" for a in FLASH_ARGS] + ["--spec-mtp", "3"]
    plan = {c.key: c for c in search.plan(q8_args, FLASH, HW_3060, "standard")}
    assert plan["draft_q4"].after is None and plan["minp_06"].after is None  # the base already has a window
    assert "mtp_3" not in plan and "mtp_5" in plan  # the base's own K is not re-tried
    assert plan["minp_06"].changes == {"--spec-min-p": "0.6", "--spec-mtp": "5"}


def test_a_pickup_offers_a_candidate_the_last_result_never_knew_even_in_a_closed_group():
    """A thorough run before 2026-10-12 tried dense_bf16; today that key sits in the "dense" group
    with dense_q8, which that run never saw. Picking up from it must still offer dense_q8 (and the
    other candidates added since) while keeping the group closed for the members it did know."""
    from freetoken.webui import tuner

    ampere = {"measurements": dict(HW_3060["measurements"], gpus=[{"index": 0, "compute_cap": 8.6}, {"index": 1, "compute_cap": 8.6}])}
    prev = {"args": list(FLASH_ARGS), "base_args": list(FLASH_ARGS), "hw": ampere,
            "tried": ["dense_bf16", "kv_16bit", "mtp_3", "mtp_5"], "kept": ["mtp_5"],
            "trials": [{"key": "base", "decision": "base", "short_s": 1.0}]}
    tried, kept = tuner.prior_for_pickup(prev, FLASH, ampere)
    known = tuner.known_to(prev, FLASH, ampere)
    assert "dense" in tried  # dense_bf16 closed the group it now belongs to
    assert "dense_q8" not in known and "dense_bf16" in known
    plan = {c.key: c for c in search.plan(list(FLASH_ARGS), FLASH, ampere, "thorough")}
    assert tuner.pickup_open(plan["dense_q8"], tried, kept, known)      # never seen: offered
    assert not tuner.pickup_open(plan["dense_bf16"], tried, kept, known)  # tried itself
    assert tuner.pickup_open(plan["minp_06"], tried, kept, known)       # mtp_5 was kept, so the window's follow-ups open
    # a member of the closed group that the last result did know stays closed
    other = search.Candidate("dense_other", {"--dense-quant": "other"}, group="dense")
    assert not tuner.pickup_open(other, tried, kept, known | {"dense_other"})
    assert tuner.pickup_open(other, tried, kept, known)  # ... and one it did not know is open
    # within a run, a group this run closed binds every member, known or not
    assert not tuner.pickup_open(other, tried | {"dense_q8", "dense"}, kept, known, run={"dense_q8", "dense"})
    # a result that recorded its plan: whatever it did not list is new to it
    prev2 = dict(prev, plan_keys=["dense_bf16", "kv_16bit", "kv_q4", "mtp_3", "mtp_5"])
    known2 = tuner.known_to(prev2, FLASH, ampere)
    assert known2 == set(prev2["plan_keys"]) and tuner.pickup_open(plan["dense_q8"], tried, kept, known2)


def test_no_moe_candidates_for_a_dense_model_and_no_mtp_without_a_head():
    dense = dict(model_type="llama", num_layers=32, quant=None, mtp=False)
    keys = _keys(["--max-running-req", "1"], dense, {}, "thorough")
    assert keys and all(not k.startswith(("strategy", "kernel", "kv_", "mtp", "threads", "minp", "draft")) for k in keys)


def test_every_skip_has_a_reason_in_both_languages():
    assert all(flag and ja and en for flag, ja, en in search.SKIPPED)


# ------------------------------------------------------------------ the job, with fakes
class FakeManager:
    def __init__(self, running=None):
        self.calls = []
        self.cfg = running  # (model, port, args) or None
        self.fail_with = set()  # arg tokens that make a start exit
        self.refuse_stop = 0  # how many stops fail their accounting request (a dead backend)

    def status(self):
        running = self.cfg is not None
        return {"running": running, "starting": False, "model": self.cfg[0] if running else None,
                "port": self.cfg[1] if running else None, "lastExitCode": None if running else 1}

    def serve_args(self):
        return list(self.cfg[2]) if self.cfg else []

    def stop(self, *a, **k):
        self.calls.append(("stop",))
        self.cfg = None
        return {"stopped": True}

    def switch(self, model, port, args, force=False):
        if self.refuse_stop and self.cfg is not None and not force:
            self.refuse_stop -= 1
            raise RuntimeError("prepare-stop request failed: <urlopen error timed out>")
        self.calls.append(("switch", model, port, list(args)))
        self.cfg = None if self.fail_with & set(args) else (model, port, list(args))
        return {"pid": 1}


class FakeProc:
    def __init__(self, lines):
        self.stdout = io.StringIO("".join(json.dumps(x) + "\n" for x in lines))
        self.returncode = 0

    def wait(self):
        return 0

    def terminate(self):
        pass


HW_LINES = [
    {"type": "plan", "steps": [{"id": "ram"}]},
    {"type": "step", "id": "ram"}, {"type": "sample", "id": "ram", "value": 20.0}, {"type": "done", "id": "ram", "value": 20.0},
    {"type": "result", "measurements": {"ram": {"read_gbs": 20.0}}, "notes": [
        {"flag": "--moe-strategy", "value": "hybrid", "why": "w", "why_en": "w"},
        {"flag": "--moe-cpu-threads", "value": "6", "why": "w", "why_en": "w"},
        {"flag": "--moe-hybrid-max-fetch", "value": "-1", "why": "w", "why_en": "w"},
    ]},
]

# a dense model keeps the plan to the prefill candidates; a MoE facts dict brings the rest
DENSE_REC = {
    "flags": ["--dtype", "float16", "--kv-reserve-tokens", "16384", "--max-seq-len-override", "16384", "--max-running-req", "1"],
    "notes": [{"flag": f, "value": v, "why": "r", "why_en": "r"} for f, v in
              [("--dtype", "float16"), ("--kv-reserve-tokens", "16384"), ("--max-seq-len-override", "16384"), ("--max-running-req", "1")]],
    "host": {"model": {"model_type": "llama", "num_layers": 32, "max_context": 262144}, "memory": {"total": 24 * GiB}, "weight_bytes": 8 * GiB},
}


def _moe_rec(mtp=False):
    rec = json.loads(json.dumps(DENSE_REC))
    rec["flags"] += ["--moe-strategy", "hybrid", "--moe-cpu-layers", "auto", "--moe-cpu-threads", "8", "--moe-cache-auto"]
    rec["host"]["model"] = dict(model_type="qwen3_5_moe", num_experts=256, num_layers=40, quant=None, mtp=mtp, max_context=262144)
    return rec


class FakeServe:
    """A serve whose speed follows its flags. ``base`` is (prose, prefill) at 16k context; ``effects``
    maps "--flag=value" (or "--flag" for a switch, "-flag" for a dropped one) to factors
    (prose, prefill, code); ``fail`` is a predicate on the flags that errors the prompt request."""

    def __init__(self, manager, base=(30.0, 600.0), effects=None, fail=None, free=2.5 * GiB, kv_extra=0, lucky=None):
        self.m, self.base, self.effects, self.fail, self.free = manager, base, effects or {}, fail, free
        self.kv_extra = kv_extra  # KV pages beyond the advertised context, as the allocator rounds up
        self.lucky = lucky or {}  # "--flag=value" -> factors on the first start with it only
        self.t = 0.0
        self.code_seen = False
        self.gen_asked = []  # max_tokens of every generation, in order
        self.per_chunk = 1  # tokens a stream chunk carries (a verify window under MTP)
        self.swing = None  # an iterator of factors, one per generation: the same settings giving different figures

    def clock(self):
        return self.t

    def flags(self):
        return dict(tuner.parse_flags(self.m.cfg[2]))

    def ctx(self):
        return int(self.flags().get("--kv-reserve-tokens") or 16384)

    def rates(self):
        f = self.flags()
        prose, prefill = self.base
        code = prose
        for key, (dp, pp, dc) in self.effects.items():
            if key.startswith("-") and not key.startswith("--"):
                hit = ("-" + key) not in f
            elif "=" in key:
                k, v = key.split("=", 1)
                hit = f.get(k) == v
            else:
                hit = key in f
            if hit:
                prose, prefill, code = prose * dp, prefill * pp, code * dc
        starts = [c for c in self.m.calls if c[0] == "switch"]
        for key, (dp, pp, dc) in self.lucky.items():
            k, v = key.split("=", 1)
            if f.get(k) == v and sum(1 for c in starts if dict(tuner.parse_flags(c[3])).get(k) == v) == 1:
                prose, prefill, code = prose * dp, prefill * pp, code * dc
        return prose, prefill, code

    def http(self, port, path, body=None, timeout=0):
        if path == "/health":
            return {"status": "ok"}
        if path == "/v1/models":
            return {"data": [{"id": "m"}]}
        if path == "/v1/cache/status":
            return {"geometry": _geo(now=self.ctx() + self.kv_extra)}
        if path.startswith("/v1/kai/experts"):
            return {"layers": [{"active": 100, "miss": 10}], "ranks": [{"gpu": {"free_bytes": int(self.free)}}]}
        if path == "/v1/completions":
            assert isinstance(body["prompt"], str)  # token-id prompts are refused by ft serve
            n = int(len(body["prompt"]) / 4)
            limit = int(self.flags().get("--max-seq-len-override") or 1 << 30)
            if n + body["max_tokens"] > limit:
                raise RuntimeError("HTTP Error 400: Bad Request")
            if body["max_tokens"] == 1:
                if self.fail and self.fail(self.flags()):
                    raise RuntimeError("HTTP Error 500: Internal Server Error")
                self.t += n / self.rates()[1]
            return {"usage": {"prompt_tokens": n, "completion_tokens": body["max_tokens"]}}
        raise AssertionError(path)

    def stream(self, port, path, body):
        prose, _, code = self.rates()
        is_code = "Python module" in body["prompt"]
        self.code_seen |= is_code
        rate = (code if is_code else prose) * (next(self.swing) if self.swing else 1.0)
        self.gen_asked.append(body["max_tokens"])
        for i in range(0, body["max_tokens"], self.per_chunk):
            n = min(self.per_chunk, body["max_tokens"] - i)
            self.t += n / rate
            yield "data: " + json.dumps({"choices": [{"text": "x" * n}]}) + "\n"
        yield "data: " + json.dumps({"choices": [], "usage": {"completion_tokens": body["max_tokens"]}}) + "\n"
        yield "data: [DONE]\n"


class FourCharTokenizer:
    """One token per four characters, as FakeServe counts them."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [text[i:i + 4] for i in range(0, len(text), 4)]}

    def decode(self, ids):
        return "".join(ids)


def test_prompts_are_cut_to_the_token_count_not_estimated():
    tok = FourCharTokenizer()
    assert tuner.fit_tokens(tok, "x" * 1000, 100) == "x" * 400
    assert tuner.fit_tokens(tok, "short", 100) == "short"
    assert tuner.fit_tokens(None, "anything", 1) == "anything"


CORPUS = {"prose": "FreeToken docs. " * 4000, "code": "def f():\n    return 1\n" * 3000}


def _job(tmp_path, manager, serve, rec=DENSE_REC, lines=HW_LINES):
    return tuner.TuneJob(manager, state_dir=str(tmp_path), python="python", default_port=1919,
                         recommend=lambda m: rec, spawn=lambda argv: FakeProc(lines), corpus=CORPUS, tokenizer=lambda m: FourCharTokenizer(),
                         http=serve.http, stream=serve.stream, clock=serve.clock, sleep=lambda s: None, gpu_indices=lambda: [0])


def _run(job, model="/models/M", trials=True, mode="standard", use="both"):
    job.start(model, trials, mode=mode, use=use)
    job._thread.join(30)
    return job.status()


def _tried(r):
    return [(t["label"], t.get("key"), t.get("decision")) for t in r["trials"]]


def test_full_run_tries_each_candidate_once_and_brings_the_old_engine_back(tmp_path):
    prev = ("/models/Old", 1919, ["--old"])
    m = FakeManager(running=prev)
    # the budget does nothing; 32k context costs 2%, 65k costs 10%
    serve = FakeServe(m, effects={"--kv-reserve-tokens=32768": (0.98, 1.0, 0.98), "--kv-reserve-tokens=65536": (0.90, 1.0, 0.90)})
    st = _run(_job(tmp_path, m, serve))
    assert st["state"] == "done", st["error"]
    r = st["result"]
    assert _tried(r) == [("A", None, "base"), ("B", "budget_075", "rejected"), ("C", "context_32768", "kept"),
                         ("D", "context_65536", "rejected")]
    assert r["chosen"] == "C"
    flags = dict(tuner.parse_flags(r["args"]))
    assert flags["--kv-reserve-tokens"] == flags["--max-seq-len-override"] == "32768"
    assert "--prefill-chunk-budget" not in flags and "--moe-collect-stats" not in flags
    assert [n for n in r["notes"] if n.get("key") == "budget_075"][0]["rejected"]
    assert m.calls[0] == ("stop",) and m.calls[-1] == ("switch", *prev) and m.cfg == prev
    assert json.load(open(tmp_path / "tune" / "M.json"))["chosen"] == "C"
    assert r["skipped"] and all(s["why"] and s["why_en"] for s in r["skipped"])
    kinds = {e["k"] for e in st["events"]}
    assert {"phase", "hw", "trial", "sample", "plan", "decision"} <= kinds
    plans = [e for e in st["events"] if e["k"] == "plan"]
    assert not plans[0]["update"] and plans[-1]["update"] and plans[-1]["items"] == []  # sent before every run, empty at the end
    assert plans[0]["eta_s"] is not None and all(p["eta_s"] >= 0 for p in plans)  # the time left, from the runs so far
    # every run measured a short prompt's first token too: 256 fresh tokens at the run's own rate
    assert all(t["short_s"] > 0 for t in r["trials"] if t["ok"])
    shorts = [e for e in st["events"] if e["k"] == "trial" and e.get("step") == "short" and e.get("state") == "done"]
    assert len(shorts) == len([t for t in r["trials"] if t["ok"]])


def test_a_kept_change_is_carried_into_every_later_run(tmp_path):
    m = FakeManager()
    serve = FakeServe(m, effects={"--prefill-chunk-budget=0.75": (1.0, 1.3, 1.0), "--kv-reserve-tokens=32768": (0.5, 1.0, 0.5)})
    r = _run(_job(tmp_path, m, serve))["result"]
    assert _tried(r)[1] == ("B", "budget_075", "kept")
    assert all(dict(tuner.parse_flags(t["args"]))["--prefill-chunk-budget"] == "0.75" for t in r["trials"][2:])
    assert dict(tuner.parse_flags(r["args"]))["--prefill-chunk-budget"] == "0.75"
    assert m.cfg is None  # nothing was running before: nothing is left running


def test_a_run_that_fails_mid_measurement_is_not_kept_and_the_job_goes_on(tmp_path):
    m = FakeManager()
    serve = FakeServe(m, effects={"--kv-reserve-tokens=32768": (0.5, 1.0, 0.5)}, fail=lambda f: f.get("--prefill-chunk-budget") == "0.75")
    st = _run(_job(tmp_path, m, serve))
    assert st["state"] == "done", st["error"]
    r = st["result"]
    assert _tried(r)[1] == ("B", "budget_075", "failed") and "500" in r["trials"][1]["error"]
    assert "--prefill-chunk-budget" not in dict(tuner.parse_flags(r["args"]))


def test_a_win_is_measured_again_and_a_lucky_run_is_not_kept(tmp_path):
    m = FakeManager()
    serve = FakeServe(m, lucky={"--prefill-chunk-budget=0.75": (1.0, 1.3, 1.0)}, effects={"--kv-reserve-tokens=32768": (0.5, 1.0, 0.5)})
    r = _run(_job(tmp_path, m, serve))["result"]
    # lucky first run, a miss, and a third run that misses too: two out of three missed
    assert _tried(r)[:4] == [("A", None, "base"), ("B", "budget_075", "rejected"), ("C", "budget_075", "rejected"),
                             ("D", "budget_075", "rejected")]
    assert r["trials"][2]["recheck"] and "--prefill-chunk-budget" not in dict(tuner.parse_flags(r["args"]))
    decisions = [e["decision"] for e in _run_events(m, serve, tmp_path) if e["k"] == "decision"][:3]
    assert decisions == ["recheck", "tiebreak", "rejected"]


def _run_events(m, serve, tmp_path):
    m2 = FakeManager()
    serve2 = FakeServe(m2, lucky=serve.lucky, effects=serve.effects)
    return _run(_job(tmp_path / "again", m2, serve2))["events"]


def test_an_unlucky_check_is_outvoted_by_a_third_run(tmp_path):
    m = FakeManager()

    class Disturbed(FakeServe):
        def rates(self):
            prose, prefill, code = super().rates()
            starts = [c for c in self.m.calls if c[0] == "switch"]
            # the second start with the budget (the check) lands on a busy PC
            if dict(tuner.parse_flags(starts[-1][3])).get("--prefill-chunk-budget") == "0.75" and \
                    sum(1 for c in starts if dict(tuner.parse_flags(c[3])).get("--prefill-chunk-budget") == "0.75") == 2:
                return prose * 0.8, prefill * 0.85, code * 0.8
            return prose, prefill, code

    serve = Disturbed(m, effects={"--prefill-chunk-budget=0.75": (1.0, 1.15, 1.0), "--kv-reserve-tokens=32768": (0.5, 1.0, 0.5)})
    st = _run(_job(tmp_path, m, serve))
    r = st["result"]
    assert _tried(r)[:4] == [("A", None, "base"), ("B", "budget_075", "kept"), ("C", "budget_075", "kept"), ("D", "budget_075", "kept")]
    assert dict(tuner.parse_flags(r["args"]))["--prefill-chunk-budget"] == "0.75"
    assert [e["decision"] for e in st["events"] if e["k"] == "decision"][:2] == ["recheck", "tiebreak"]


def test_a_change_that_holds_twice_needs_no_third_run(tmp_path):
    m = FakeManager()
    serve = FakeServe(m, effects={"--prefill-chunk-budget=0.75": (1.0, 1.3, 1.0), "--kv-reserve-tokens=32768": (0.5, 1.0, 0.5)})
    r = _run(_job(tmp_path, m, serve))["result"]
    assert [k for _, k, _ in _tried(r)].count("budget_075") == 2


def test_a_confirmed_win_is_kept_at_the_slower_of_its_two_runs(tmp_path):
    m = FakeManager()
    serve = FakeServe(m, effects={"--prefill-chunk-budget=0.75": (1.0, 1.3, 1.0), "--kv-reserve-tokens=32768": (0.5, 1.0, 0.5)},
                      lucky={"--prefill-chunk-budget=0.75": (1.0, 1.2, 1.0)})
    r = _run(_job(tmp_path, m, serve))["result"]
    assert _tried(r)[:3] == [("A", None, "base"), ("B", "budget_075", "kept"), ("C", "budget_075", "kept")]
    note = [n for n in r["notes"] if n["flag"] == "--prefill-chunk-budget"][0]
    assert "780" in note["why_en"] and "936" not in note["why_en"]  # 600 x 1.3, not the lucky 600 x 1.56


def test_the_prompt_fits_the_advertised_context_when_the_kv_pages_are_rounded_up(tmp_path):
    m = FakeManager()
    serve = FakeServe(m, kv_extra=8192, effects={"--kv-reserve-tokens=32768": (0.5, 1.0, 0.5)})
    st = _run(_job(tmp_path, m, serve))
    assert st["state"] == "done", st["error"]
    assert all(t["ok"] for t in st["result"]["trials"])


def test_a_measurement_that_fails_does_not_drop_float16(tmp_path):
    m = FakeManager()
    serve = FakeServe(m, fail=lambda f: True)
    st = _run(_job(tmp_path, m, serve))
    assert st["state"] == "error"
    assert [t for t in st["result"]["trials"] if t.get("change")] == [] if st.get("result") else True
    assert all("--dtype" in c[3] for c in m.calls if c[0] == "switch")


def test_the_recommendation_follows_the_last_benchmark(tmp_path):
    m = FakeManager()
    serve = FakeServe(m, effects={"--prefill-chunk-budget=0.75": (1.0, 1.3, 1.0), "--kv-reserve-tokens=32768": (0.5, 1.0, 0.5)})
    job = _job(tmp_path, m, serve)
    result = _run(job)["result"]
    rec = json.loads(json.dumps(DENSE_REC))
    rec["notes"].append({"flag": "--new-rule", "value": "1", "why": "r", "why_en": "r"})
    out = tuner.with_benchmark(rec, job.last("/models/M"))
    assert out["flags"] == result["args"] and out["benchmark"]["trials"] == len(result["trials"])
    assert out["benchmark"]["version"] == out["benchmark"]["current"] == tuner.FT_VERSION
    by = {n["flag"]: n for n in out["notes"]}
    assert by["--prefill-chunk-budget"]["value"] == "0.75" and by["--prefill-chunk-budget"]["source"] == "measured"
    assert "--new-rule" in by  # a rule the benchmark never saw is still offered
    assert not [n for n in out["notes"] if n.get("rejected")]
    # every recommended flag is carried by the benchmark's flags: a profile saved from it has nothing to apply
    flags = dict(tuner.parse_flags(result["args"]))
    assert all((n["flag"] not in flags) if n.get("removed") else n["flag"] in flags for n in out["notes"] if n["flag"] != "--new-rule")
    info = dict(result, notes=result["notes"] + [{"flag": "--moe-hybrid-max-fetch", "value": "-1", "why": "w", "why_en": "w",
                                                   "source": "measured", "info": True}])
    assert "--moe-hybrid-max-fetch" not in {n["flag"] for n in tuner.with_benchmark(rec, info)["notes"]}
    assert tuner.with_benchmark(rec, None) is rec and tuner.with_benchmark(rec, {"error": "x", "args": ["--a"]}) is rec


def test_the_warm_up_is_counted_in_tokens_not_in_chunks():
    # plain decoding, one token a chunk at 20 tok/s after a slow first 60: chunk 61 on is timed, as before
    slow = [i * 0.2 for i in range(60)]
    plain = slow + [slow[-1] + 0.2 + i * 0.05 for i in range(240)]
    assert tuner.rate_after_warmup(plain, 300) == pytest.approx(20.0)
    # MTP, three tokens a window: 300 tokens are 100 chunks, the warm-up is the first 20 of them and
    # 79 windows are timed -- counting 60 chunks would have left 39
    windows = [i * 0.6 for i in range(20)] + [19 * 0.6 + 0.6 + i * 0.1 for i in range(80)]
    assert tuner.rate_after_warmup(windows, 300) == pytest.approx(30.0)
    slower_tail = windows[:61] + [windows[60] + (i + 1) * 0.3 for i in range(39)]  # the last 39 windows at 10 tok/s
    assert tuner.rate_after_warmup(slower_tail, 300) < 20.0  # the 40 windows before them count too
    # too short for a warm-up: the whole run; nothing to time: None
    assert tuner.rate_after_warmup([0.0, 1.0, 2.0], 30) == pytest.approx(14.5)
    assert tuner.rate_after_warmup([1.0], 5) is None and tuner.rate_after_warmup([], None) is None
    assert tuner.rate_after_warmup(plain, None) == pytest.approx(20.0)  # no usage count: a chunk is a token


def test_a_run_with_an_mtp_window_generates_longer_and_takes_the_median(tmp_path):
    m = FakeManager()
    serve = FakeServe(m)
    serve.per_chunk = 3
    job = _job(tmp_path, m, serve, rec=_moe_rec(mtp=True))
    # one generation in three comes out 60% faster, as when the text happens to take every draft
    import itertools
    serve.swing = itertools.cycle([1.0, 1.6, 1.0, 1.0])
    plain = job._trial("A", "/models/M", 1919, ["--max-seq-len-override", "16384"], code=True)
    asked, serve.gen_asked = serve.gen_asked, []
    assert set(asked) == {tuner.DECODE_TOKENS} and len(asked) == 8  # a warm-up and three runs, on prose and on code
    assert plain["gen_tokens"] == 300 and plain["decode_tps"] == pytest.approx(48.0, rel=0.01)  # no window: the best run
    assert len(plain["decode_runs"]) == 3 and len(plain["decode_code_runs"]) == 3
    serve.swing = itertools.cycle([1.0, 1.6, 1.0, 1.0])
    win = job._trial("B", "/models/M", 1919, ["--max-seq-len-override", "16384", "--spec-mtp", "3"], code=True)
    assert serve.gen_asked == [300, 600, 600, 600] * 2  # the warm-up stays short
    assert win["gen_tokens"] == 600 and win["decode_tps"] == pytest.approx(30.0, rel=0.01)  # a window: the median, not the lucky run
    assert sorted(win["decode_runs"]) == [pytest.approx(30.0, rel=0.01), pytest.approx(30.0, rel=0.01), pytest.approx(48.0, rel=0.01)]
    zero = job._trial("C", "/models/M", 1919, ["--max-seq-len-override", "16384", "--spec-mtp", "0"], code=True)
    assert zero["gen_tokens"] == 300  # --spec-mtp 0 is no window


def test_a_pickup_from_a_result_that_timed_few_windows_judges_mtp_again(tmp_path):
    m = FakeManager()
    serve = FakeServe(m, effects={"--kv-reserve-tokens=32768": (0.5, 1.0, 0.5)})
    job = _pickup_job(tmp_path, m, serve, rec=_moe_rec(mtp=True))
    first = _run(job, mode="standard")["result"]
    assert first["gen_method"] == tuner.GEN_METHOD and {"mtp_3", "mtp_5"} <= set(first["tried"])
    assert job.pending("/models/M", "standard")["items"] == [] and not job.pending("/models/M", "standard")["windows_missing"]
    # the same result as an older build saved it: no record of how generation was measured
    path = tmp_path / "tune" / "M.json"
    old = json.loads(path.read_text())
    del old["gen_method"]
    path.write_text(json.dumps(old))
    pend = job.pending("/models/M", "standard")
    assert {"mtp_3", "mtp_5"} <= {i["key"] for i in pend["items"]} and pend["windows_missing"]
    assert not {i["key"] for i in pend["items"]} - tuner.MTP_UNJUDGED  # and nothing else is opened again
    job._spawn = _no_hw
    job.start("/models/M", True, mode="standard", from_last=True)
    job._thread.join(30)
    st = job.status()
    assert st["state"] == "done", st["error"]
    assert {"mtp_3", "mtp_5"} <= {t.get("key") for t in st["result"]["trials"]}
    assert st["result"]["gen_method"] == tuner.GEN_METHOD and job.pending("/models/M", "standard")["items"] == []


GGUF_FLAGS = ["--moe-expert-gguf", "/models/M-GGUF/M-Q2_0.gguf", "--reasoning-budget", "6000"]


def _experts_job(tmp_path, m, asked):
    """A job whose recommendation proposes the Q2_0 GGUF unless the checkpoint's experts are asked for."""
    def recommend(model, experts=None):
        asked.append(experts)
        rec = json.loads(json.dumps(DENSE_REC))
        if experts != "checkpoint":
            rec["flags"] += GGUF_FLAGS
        return rec

    job = _job(tmp_path, m, FakeServe(m))
    job._recommend = recommend
    return job


def _start(job, **kw):
    job.start("/models/M", **kw)
    job._thread.join(30)
    st = job.status()
    assert st["state"] == "done", st["error"]
    return st["result"]


def test_each_experts_choice_keeps_its_own_result(tmp_path):
    asked: list = []
    job = _experts_job(tmp_path, FakeManager(), asked)
    plain, q2 = tmp_path / "tune" / "M.json", tmp_path / "tune" / "M.q2_0.json"

    ckpt = _start(job, experts="checkpoint")
    assert ckpt["experts"] == "checkpoint" and "--moe-expert-gguf" not in ckpt["args"]
    assert plain.exists() and not q2.exists()
    before = plain.read_bytes()

    fast = _start(job, experts="q2")
    assert fast["experts"] == "q2" and "--moe-expert-gguf" in fast["args"] and "--moe-expert-gguf" in fast["base_args"]
    assert plain.read_bytes() == before  # measuring the GGUF did not replace the checkpoint's result
    assert job.last("/models/M")["experts"] == "checkpoint" and job.last("/models/M", "checkpoint")["experts"] == "checkpoint"
    assert job.last("/models/M", "q2")["experts"] == "q2"

    # a pickup continues the result of the experts asked for, and files what it finds there
    assert job.pending("/models/M", experts="q2")["available"] and job.pending("/models/M")["available"]
    again = _start(job, from_last=True, experts="q2")
    assert again["from_last"] and again["experts"] == "q2" and "--moe-expert-gguf" in again["args"]
    assert plain.read_bytes() == before and json.loads(q2.read_text())["from_last"]
    assert "checkpoint" in asked and "q2" in asked

    # no choice: the recommendation's own (the GGUF, since it is there), filed under what it served
    asked.clear()
    assert _start(job)["experts"] == "q2" and asked[0] is None and plain.read_bytes() == before


def test_a_q2_result_saved_under_the_plain_name_is_not_taken_for_the_checkpoints(tmp_path):
    # before results were filed by experts, a run with the GGUF next to the checkpoint wrote M.json
    job = _experts_job(tmp_path, FakeManager(), [])
    os.makedirs(tmp_path / "tune")
    (tmp_path / "tune" / "M.json").write_text(json.dumps({"args": GGUF_FLAGS, "finished": 1.0}))
    assert job.last("/models/M") is None and job.last("/models/M", "q2")["finished"] == 1.0
    assert job.pending("/models/M")["reason"] == "no_last"


def test_asking_for_the_gguf_where_there_is_none_fails_before_any_server_starts(tmp_path):
    m = FakeManager()
    job = _job(tmp_path, m, FakeServe(m))  # this recommendation never proposes a GGUF
    job._recommend = lambda model, experts=None: DENSE_REC
    job.start("/models/M", experts="q2")
    job._thread.join(30)
    st = job.status()
    assert st["state"] == "error" and "Q2_0" in st["error"]
    assert not [c for c in m.calls if c[0] == "switch"] and not (tmp_path / "tune").exists()


def test_a_change_dropped_for_generation_alone_is_judged_against_the_base_measured_again(tmp_path):
    m = FakeManager()
    # the budget gains 20% prefill; its start lands on a slow patch of the PC (generation -12%), and the
    # base started right after is just as slow: the change is kept

    class Swinging(FakeServe):
        def rates(self):
            prose, prefill, code = super().rates()
            starts = len([c for c in self.m.calls if c[0] == "switch"])
            return (prose * 0.88, prefill, code * 0.88) if starts in (2, 3) else (prose, prefill, code)

    serve = Swinging(m, effects={"--prefill-chunk-budget=0.75": (1.0, 1.2, 1.0), "--kv-reserve-tokens=32768": (0.5, 1.0, 0.5)})
    r = _run(_job(tmp_path, m, serve))["result"]
    got = _tried(r)[:4]
    assert got[1] == ("B", "budget_075", "kept") and got[2] == ("C", "base_again", "base") and got[3][1:] == ("budget_075", "kept")
    assert dict(tuner.parse_flags(r["args"]))["--prefill-chunk-budget"] == "0.75"


def test_close_on_generation_only_when_generation_alone_fell():
    c = search.Candidate("x", {"--a": "1"})
    base = _t(600, 30.0)
    assert tuner.close_on_generation(base, _t(700, 27.0), c, "prose")       # prefill +17%, generation -10%
    assert not tuner.close_on_generation(base, _t(700, 20.0), c, "prose")   # generation -33%: not a swing
    assert not tuner.close_on_generation(base, _t(600, 30.5), c, "prose")   # nothing fell
    assert not tuner.close_on_generation(base, _t(500, 27.0), c, "prose")   # prefill fell too


def test_every_run_reads_the_same_prompts(tmp_path):
    m = FakeManager()
    serve = FakeServe(m, effects={"--kv-reserve-tokens=32768": (0.5, 1.0, 0.5)})
    seen = []
    http = serve.http

    def record(port, path, body=None, timeout=0):
        if path == "/v1/completions" and body["max_tokens"] == 1:
            seen.append((dict(tuner.parse_flags(m.cfg[2])).get("--prefill-chunk-budget"), body["prompt"].split("\n", 1)))
        return http(port, path, body, timeout)

    serve.http = record
    _run(_job(tmp_path, m, serve))
    base = [b for flag, (_, b) in seen if flag is None][:2]
    budget = [b for flag, (_, b) in seen if flag == "0.75"][:2]
    assert base and base == budget  # the same text, run for run
    heads = [h for _, (h, _) in seen]
    assert len(set(heads)) == len(heads)  # but never the same first line


def test_a_server_that_cannot_answer_its_stop_is_replaced_by_force(tmp_path):
    m = FakeManager()
    m.refuse_stop = 1
    serve = FakeServe(m, effects={"--kv-reserve-tokens=32768": (0.5, 1.0, 0.5)})
    st = _run(_job(tmp_path, m, serve))
    assert st["state"] == "done", st["error"]
    assert all(t["ok"] for t in st["result"]["trials"])
    assert any(e["k"] == "log" and "force" in e["msg"] for e in st["events"])


def test_undoing_candidates_are_not_both_tried(tmp_path):
    m = FakeManager()
    # offload wins; from offload args the plan offers hybrid again, which must not be tried
    serve = FakeServe(m, effects={"--moe-strategy=offload": (1.2, 1.0, 1.2), "--kv-reserve-tokens=32768": (0.5, 1.0, 0.5)})
    r = _run(_job(tmp_path, m, serve, rec=_moe_rec()))["result"]
    keys = [k for _, k, _ in _tried(r)]
    assert "strategy_offload" in keys and "strategy_hybrid" not in keys
    flags = dict(tuner.parse_flags(r["args"]))
    assert flags["--moe-strategy"] == "offload" and "--moe-cpu-threads" not in flags
    assert "threads_fewer" not in keys[keys.index("strategy_offload"):]  # hybrid-only candidates close after the switch


def test_mtp_is_measured_on_code_and_the_use_decides(tmp_path):
    effects = {"--spec-mtp=3": (0.85, 0.95, 1.6), "--spec-mtp=5": (0.80, 0.95, 2.0), "--kv-reserve-tokens=32768": (0.5, 1.0, 0.5)}
    m = FakeManager()
    serve = FakeServe(m, effects=effects)
    r = _run(_job(tmp_path, m, serve, rec=_moe_rec(mtp=True)), use="code")["result"]
    assert serve.code_seen
    decisions = {k: d for _, k, d in _tried(r)}
    assert decisions["mtp_3"] == "kept" and decisions["mtp_5"] == "kept"  # 5 tokens: a quarter faster on code than 3
    assert dict(tuner.parse_flags(r["args"]))["--spec-mtp"] == "5"
    # between two windows a gain inside the swing of windowed generation (6% here) is not a gain
    m3 = FakeManager()
    close = dict(effects, **{"--spec-mtp=5": (0.80, 0.95, 1.7)})
    r3 = _run(_job(tmp_path / "close", m3, FakeServe(m3, effects=close), rec=_moe_rec(mtp=True)), use="code")["result"]
    d3 = {k: d for _, k, d in _tried(r3)}
    assert d3["mtp_3"] == "kept" and d3["mtp_5"] == "rejected" and dict(tuner.parse_flags(r3["args"]))["--spec-mtp"] == "3"
    mtp_runs = [t for t in r["trials"] if (t.get("key") or "").startswith("mtp_")]
    assert all(t["decode_code_tps"] > t["decode_tps"] for t in mtp_runs)
    assert all(t.get("decode_code_tps") for t in r["trials"] if t.get("ok"))  # every run, so the figures compare

    m2 = FakeManager()
    r2 = _run(_job(tmp_path / "prose", m2, FakeServe(m2, effects=effects), rec=_moe_rec(mtp=True)), use="prose")["result"]
    assert "--spec-mtp" not in dict(tuner.parse_flags(r2["args"]))


def test_float16_that_does_not_load_is_dropped(tmp_path):
    m = FakeManager()
    m.fail_with = {"float16"}
    serve = FakeServe(m, effects={"--kv-reserve-tokens=32768": (0.5, 1.0, 0.5)})
    st = _run(_job(tmp_path, m, serve))
    r = st["result"]
    assert st["state"] == "done", st["error"]
    assert "--dtype" not in dict(tuner.parse_flags(r["args"]))
    assert [n for n in r["notes"] if n["flag"] == "--dtype"][0].get("removed")
    assert r["trials"][0]["ok"] is False


def test_upstreams_bench_benches_the_models_own_layer_kinds():
    # profile version 6 (upstream #734): the engine reads each MoE layer's own row, so the bench takes the checkpoint;
    # the Q2_0 tier (the fork's GGUF banks at the checkpoint's expert shape) rides along as --formats q2_0
    assert tuner.bench_model_args("/models/M", "checkpoint", facts={"num_experts": 512}) == ["--model", "/models/M"]
    assert tuner.bench_model_args("/models/M", "q2", facts={"num_experts": 512}) == ["--model", "/models/M", "--formats", "q2_0"]
    assert tuner.bench_model_args("/models/dense", "checkpoint", facts={"num_experts": None}) == []
    assert tuner.bench_model_args("/nowhere", "q2") == []  # no config.json to read: the bare bench


def test_upstreams_bench_runs_first_on_every_gpu_and_is_stamped(tmp_path):
    prof = tmp_path / "GPU-x.json"
    started = []

    def spawn(argv, env=None):
        started.append(argv[2:5])
        if "bench" in argv:
            assert env["FREETOKEN_BENCH_PROGRESS"] == "1" and argv[-2:] == ["--gpu", str(len(started) - 1)]
            prof.write_text(json.dumps({"version": 4, "gpu": {"name": "RTX"}}))
            return _TextProc(["FTBENCH 0 2 ceilings", "FTBENCH 1 2 dtype:nvfp4", f"FTBENCH_OUT {prof}"])
        return FakeProc(HW_LINES)

    m = FakeManager()
    job = tuner.TuneJob(m, state_dir=str(tmp_path), python="python", default_port=1919, recommend=lambda x: DENSE_REC,
                        spawn=spawn, gpu_indices=lambda: [0, 1], http=FakeServe(m).http)
    st = _run(job, trials=False)
    assert st["state"] == "done", st["error"]
    assert started == [["freetoken.cli", "bench", "bw"], ["freetoken.cli", "bench", "bw"], ["freetoken.webui.hwbench", "--model", "/models/M"]]
    assert json.loads(prof.read_text())["freetoken_version"] == tuner.FT_VERSION
    assert [u["gpu"] for u in st["result"]["upstream"]] == [0, 1] and st["result"]["freetoken_version"] == tuner.FT_VERSION
    samples = [e for e in st["events"] if e["k"] == "hw" and e.get("id") == "upstream0" and e["kind"] == "sample"]
    assert [s["value"] for s in samples] == [0, 50]
    phases = [e["phase"] for e in st["events"] if e["k"] == "phase"]
    assert phases.index("upstream") < phases.index("hw")


class _TextProc(FakeProc):
    def __init__(self, lines):
        self.stdout = io.StringIO("".join(x + "\n" for x in lines))
        self.returncode = 0


def test_a_failing_upstream_bench_stops_the_benchmark(tmp_path):
    class Failed(_TextProc):
        def wait(self):
            return 1

    m = FakeManager()
    job = tuner.TuneJob(m, state_dir=str(tmp_path), python="python", default_port=1919, recommend=lambda x: DENSE_REC,
                        spawn=lambda argv, env=None: Failed(["RuntimeError: no CUDA device"]), gpu_indices=lambda: [0])
    st = _run(job, trials=False)
    assert st["state"] == "error" and "no CUDA device" in st["error"]


def test_hardware_only_does_not_start_a_server(tmp_path):
    m = FakeManager()
    st = _run(_job(tmp_path, m, FakeServe(m), rec=_moe_rec()), trials=False)
    assert st["state"] == "done"
    assert not [c for c in m.calls if c[0] == "switch"]
    assert dict(tuner.parse_flags(st["result"]["args"]))["--moe-strategy"] == "hybrid"


def test_offload_from_the_bench_drops_the_cpu_flags(tmp_path):
    lines = [dict(x) for x in HW_LINES]
    lines[-1] = {"type": "result", "measurements": {}, "notes": [{"flag": "--moe-strategy", "value": "offload", "why": "w", "why_en": "w"}]}
    m = FakeManager()
    st = _run(_job(tmp_path, m, FakeServe(m), rec=_moe_rec(), lines=lines), trials=False)
    flags = dict(tuner.parse_flags(st["result"]["args"]))
    assert flags["--moe-strategy"] == "offload" and "--moe-cpu-threads" not in flags and "--moe-cpu-layers" not in flags


def test_prompt_progress_is_reported_while_the_request_runs(tmp_path):
    import time as _time

    counter = {"n": 5000}

    def http(port, path, body=None, timeout=0):
        if path.startswith("/v1/kai/experts"):
            counter["n"] += 2304  # one chunk per look
            return {"ranks": [{"counters": {"prefill_new_tokens": counter["n"]}}]}
        if path == "/v1/completions":
            _time.sleep(2.3)
            return {"usage": {"prompt_tokens": 8192}}
        raise AssertionError(path)

    job = tuner.TuneJob(FakeManager(running=("/m", 1919, [])), state_dir=str(tmp_path), python="python", default_port=1919, http=http)
    doc = job._prefill_once("A", 1919, {"prompt": "x"}, 8192)
    assert doc["usage"]["prompt_tokens"] == 8192
    progress = [e for e in job.events if e["k"] == "progress"]
    assert len(progress) >= 2
    assert progress[0]["done"] == 2304 and progress[0]["total"] == 8192 and progress[0]["rate"] > 0
    assert all(e["done"] <= 8192 for e in progress)


def test_a_prompt_to_a_server_that_exited_is_not_waited_on(tmp_path):
    import threading
    import time as _time

    release = threading.Event()
    m = FakeManager(running=("/m", 1919, []))

    def http(port, path, body=None, timeout=0):
        if path.startswith("/v1/kai/experts"):
            return {"ranks": []}
        release.wait(30)  # the reply that never comes
        return {"usage": {"prompt_tokens": 1}}

    job = tuner.TuneJob(m, state_dir=str(tmp_path), python="python", default_port=1919, http=http)
    threading.Timer(1.5, lambda: setattr(m, "cfg", None)).start()
    t0 = _time.monotonic()
    with pytest.raises(RuntimeError, match="exited"):
        job._prefill_once("A", 1919, {"prompt": "x"}, 8192)
    assert _time.monotonic() - t0 < 10
    release.set()


def test_a_prompt_that_stops_progressing_is_given_up(tmp_path):
    import threading

    release = threading.Event()
    clock = {"t": 0.0}

    def http(port, path, body=None, timeout=0):
        if path.startswith("/v1/kai/experts"):
            clock["t"] += 20  # every look is 20 s later; the counter never moves again
            return {"ranks": [{"counters": {"prefill_new_tokens": 4096}}]}
        release.wait(30)
        return {"usage": {"prompt_tokens": 1}}

    job = tuner.TuneJob(FakeManager(running=("/m", 1919, [])), state_dir=str(tmp_path), python="python", default_port=1919,
                        http=http, clock=lambda: clock["t"])
    with pytest.raises(RuntimeError, match="no progress"):
        job._prefill_once("A", 1919, {"prompt": "x"}, 8192)
    assert clock["t"] <= tuner.PROMPT_STALL_S + 60
    release.set()


def test_a_hardware_step_that_goes_quiet_is_stopped(tmp_path):
    import threading

    release = threading.Event()

    class HungProc:
        killed = False

        def __init__(self):
            def lines():
                yield json.dumps({"type": "step", "id": "gather1"}) + "\n"
                release.wait(10)  # a CUDA kernel that never returns

            self.stdout = lines()
            self.returncode = None

        def kill(self):
            HungProc.killed = True
            release.set()

        terminate = kill

        def wait(self):
            return -9

    job = tuner.TuneJob(FakeManager(), state_dir=str(tmp_path), python="python", default_port=1919,
                        spawn=lambda argv: HungProc())
    with pytest.raises(RuntimeError, match="gather1"):
        job._hardware("/models/M", quiet_s=0.3)
    assert HungProc.killed


def test_a_second_start_while_running_is_refused(tmp_path):
    m = FakeManager()
    job = _job(tmp_path, m, FakeServe(m))
    job.state = "running"
    with pytest.raises(tuner.Busy):
        job.start("/models/M")


def test_cancel_restores_and_reports(tmp_path):
    prev = ("/models/Old", 1919, ["--old"])
    m = FakeManager(running=prev)
    job = _job(tmp_path, m, FakeServe(m))
    job._cancel.set()
    job._run("/models/M", True, 1919)
    assert job.state == "cancelled" and m.cfg == prev


# ------------------------------------------------------------------ picking up from the last result
def _no_hw(argv, env=None):
    raise AssertionError(f"picking up must not measure the hardware again: {argv}")


def _pickup_job(tmp_path, m, serve, rec=DENSE_REC):
    job = _job(tmp_path, m, serve, rec=rec)
    return job


def test_picking_up_runs_only_what_the_last_run_did_not_try(tmp_path):
    m = FakeManager()
    serve = FakeServe(m, effects={"--kv-reserve-tokens=32768": (0.5, 1.0, 0.5)})
    job = _pickup_job(tmp_path, m, serve)
    first = _run(job, mode="standard")["result"]
    assert first["tried"] and "budget_075" in first["tried"]
    expected = [x["key"] for x in job.pending("/models/M", "thorough")["items"]]
    assert expected, "the thorough plan should hold candidates a standard run did not try"
    assert job.pending("/models/M", "standard")["items"] == []

    job._spawn = _no_hw
    job.start("/models/M", True, mode="thorough", from_last=True)
    job._thread.join(30)
    st = job.status()
    assert st["state"] == "done", st["error"]
    r = st["result"]
    keys = [t.get("key") for t in r["trials"] if t.get("decision") != "base"]
    assert [k for k in keys if not k.startswith("context_")] == expected
    assert not any(k.startswith("context_") for k in keys)  # nothing new kept: the last tiers stand
    assert r["from_last"]["finished"] == first["finished"] and r["hw"] == first["hw"]
    assert set(first["tried"]) | set(expected) <= set(r["tried"])
    base = [t for t in r["trials"] if t.get("decision") == "base"][0]
    assert dict(tuner.parse_flags(base["args"]))["--kv-reserve-tokens"] == dict(tuner.parse_flags(first["args"]))["--kv-reserve-tokens"]
    assert json.load(open(tmp_path / "tune" / "M.json"))["from_last"]  # the saved result is this one
    # and a third pick-up has nothing left
    assert job.pending("/models/M", "thorough")["items"] == []


def test_a_new_win_when_picking_up_runs_the_context_tiers_again(tmp_path):
    m = FakeManager()
    serve = FakeServe(m, effects={"--kv-reserve-tokens=32768": (0.5, 1.0, 0.5)})
    job = _pickup_job(tmp_path, m, serve)
    _run(job, mode="standard")
    new = job.pending("/models/M", "thorough")["items"][0]
    flag, value = new["change"]["flag"], new["change"]["value"]
    serve.effects[f"{flag}={value}" if value is not None else flag] = (1.0, 1.4, 1.0)
    job._spawn = _no_hw
    job.start("/models/M", True, mode="thorough", from_last=True)
    job._thread.join(30)
    r = job.status()["result"]
    assert any(t.get("key") == new["key"] and t.get("decision") == "kept" for t in r["trials"])
    assert any((t.get("key") or "").startswith("context_") for t in r["trials"])
    assert new["key"] in r["kept"]


def test_nothing_to_pick_up_from_is_refused_before_anything_stops(tmp_path):
    m = FakeManager(running=("/models/Old", 1919, ["--old"]))
    job = _pickup_job(tmp_path, m, FakeServe(m))
    with pytest.raises(tuner.NoLastResult) as exc:
        job.start("/models/M", True, from_last=True)
    assert str(exc.value) == "no_last" and m.calls == []  # the running engine was left alone
    assert job.pending("/models/M")["available"] is False


def test_a_result_saved_before_the_record_still_counts_its_trials():
    old = {"trials": [{"key": None, "decision": "base"}, {"key": "budget_075", "decision": "kept"},
                      {"key": "max_prefill_16k", "decision": "rejected"}, {"key": "base_again", "decision": "base"}]}
    assert tuner.prior_trials(old) == ({"budget_075", "max_prefill_16k"}, {"budget_075"})
    assert tuner.last_refusal({"args": ["--x"], "hw": {}, "trials": []}, [0]) == "last_incomplete"
    assert tuner.last_refusal({"args": ["--x"], "hw": {"a": 1}, "upstream": [{}], "trials": [{"decision": "base"}]}, [0, 1]) \
        == "last_other_gpus"


def test_a_kept_change_closes_the_one_that_would_undo_it_in_an_older_result():
    # an older result has no record of groups: the plan its first settings opened says that the kept
    # strategy_offload closed the "strategy" group, so strategy_hybrid -- offered by the final
    # (offload) settings -- is not listed as untested
    rec = _moe_rec()
    facts, hw = tuner.host_facts(rec), {"measurements": {}}
    base = list(rec["flags"])
    final = tuner.set_flags(base, {"--moe-strategy": "offload"}, drop=("--moe-cpu-layers", "--moe-cpu-threads"))
    offered = {c.key: c.group for c in search.plan(final, facts, hw, "thorough")}
    assert offered.get("strategy_hybrid") == "strategy"
    prev = {"base_args": base, "args": final, "trials": [{"key": None, "decision": "base"},
                                                         {"key": "strategy_offload", "decision": "kept"}]}
    tried, _ = tuner.prior_trials(prev)
    assert "strategy" in tuner.closed_groups(prev, tried, facts, hw)


def test_a_result_measured_before_the_short_prompt_tries_hybrid_or_offload_again(tmp_path):
    # offload was kept on the 16k prompt and generation alone; with it a chat turn loses the split
    # prefill, which the old result never saw -- picking up tries the choice again and says why
    rec = _moe_rec()
    facts, hw = tuner.host_facts(rec), {"measurements": {}}
    base = list(rec["flags"])
    final = tuner.set_flags(base, {"--moe-strategy": "offload"}, drop=("--moe-cpu-layers", "--moe-cpu-threads"))
    old = {"base_args": base, "args": final, "hw": hw, "upstream": [{"gpu": 0}], "finished": 1.0, "mode": "standard",
           "tried": ["strategy_offload", "strategy", "budget_075"], "kept": ["strategy_offload"],
           "trials": [{"key": None, "decision": "base"}, {"key": "strategy_offload", "decision": "kept"}]}
    tried, kept = tuner.prior_for_pickup(old, facts, hw)
    assert not tried & tuner.SHORT_UNJUDGED and "budget_075" in tried and kept == {"strategy_offload"}
    measured = dict(old, trials=old["trials"] + [{"key": "strategy_hybrid", "decision": "rejected", "short_s": 3.0}])
    assert "strategy" in tuner.prior_for_pickup(measured, facts, hw)[0]  # judged once with it: closed again

    m = FakeManager()
    job = _pickup_job(tmp_path, m, FakeServe(m), rec=rec)
    os.makedirs(tmp_path / "tune")
    with open(tmp_path / "tune" / "M.json", "w") as fh:
        json.dump(old, fh)
    doc = job.pending("/models/M", "standard")
    assert doc["short_missing"] and "strategy_hybrid" in [x["key"] for x in doc["items"]]
    with open(tmp_path / "tune" / "M.json", "w") as fh:
        json.dump(measured, fh)
    doc = job.pending("/models/M", "standard")
    assert not doc["short_missing"] and "strategy_hybrid" not in [x["key"] for x in doc["items"]]


def test_after_picking_up_the_retried_choice_is_no_longer_listed(tmp_path):
    # the run the page showed on two 3060s: hybrid measured again with the short prompt, offload
    # tried and not kept -- afterwards nothing is left to try and the notice is gone
    rec = _moe_rec()
    base = list(rec["flags"])
    old = {"base_args": base, "args": base, "hw": {"measurements": {}}, "upstream": [{"gpu": 0}], "finished": 1.0,
           "mode": "thorough", "tried": ["strategy_offload", "strategy", "budget_075"], "kept": [],
           "trials": [{"key": None, "decision": "base"}, {"key": "strategy_offload", "decision": "rejected"}]}
    m = FakeManager()
    # offload halves generation and the short prompt waits longer, as on the 3060s (2.86 -> 7.70 s)
    serve = FakeServe(m, effects={"--moe-strategy=offload": (0.48, 0.4, 0.46)})
    job = _pickup_job(tmp_path, m, serve, rec=rec)
    os.makedirs(tmp_path / "tune")
    with open(tmp_path / "tune" / "M.json", "w") as fh:
        json.dump(old, fh)
    before = job.pending("/models/M", "thorough")
    assert before["short_missing"] and "strategy_offload" in [x["key"] for x in before["items"]]
    job._spawn = _no_hw
    job.start("/models/M", True, mode="thorough", from_last=True)
    job._thread.join(60)
    st = job.status()
    assert st["state"] == "done", st["error"]
    r = st["result"]
    decided = {t.get("key"): t.get("decision") for t in r["trials"]}
    assert decided[None] == "base" and decided["strategy_offload"] == "rejected"
    assert all(t.get("short_s") for t in r["trials"] if t.get("ok"))
    after = job.pending("/models/M", "thorough")
    assert not after["short_missing"] and "strategy_offload" not in [x["key"] for x in after["items"]]
    assert after["items"] == []  # everything the pick-up listed was run


def test_a_candidate_the_settings_already_carry_is_not_listed(tmp_path):
    # four prefill pieces set the chunk ceiling to 16384, so "raise the ceiling to 16384" changes
    # nothing: the run skips it without a start, and the list shown before must not count it
    m = FakeManager()
    job = _pickup_job(tmp_path, m, FakeServe(m))
    args = DENSE_REC["flags"] + ["--max-prefill-length", "16384"]
    os.makedirs(tmp_path / "tune")
    with open(tmp_path / "tune" / "M.json", "w") as fh:
        json.dump({"args": args, "base_args": DENSE_REC["flags"], "hw": {"measurements": {}}, "upstream": [{"gpu": 0}],
                   "trials": [{"key": None, "decision": "base"}], "tried": ["budget_075"], "kept": [],
                   "finished": 1.0, "mode": "standard"}, fh)
    items = [x["key"] for x in job.pending("/models/M", "thorough")["items"]]
    assert "max_prefill_16k" not in items


def test_picking_up_keeps_the_earlier_runs_on_record(tmp_path):
    m = FakeManager()
    serve = FakeServe(m, effects={"--kv-reserve-tokens=32768": (0.5, 1.0, 0.5)})
    job = _pickup_job(tmp_path, m, serve)
    first = _run(job, mode="standard")["result"]
    job._spawn = _no_hw
    for _ in range(2):
        job.start("/models/M", True, mode="thorough", from_last=True)
        job._thread.join(30)
    r = job.status()["result"]
    assert [e["finished"] for e in r["earlier"]][0] == first["finished"] and len(r["earlier"]) == 2
    assert r["earlier"][0]["trials"] == json.loads(json.dumps(first["trials"]))  # as saved


def test_an_explicit_kv_size_skips_the_context_tiers(tmp_path):
    # one card (Flash-Next on 12 GB): the KV is sized by --num-tokens, which --kv-reserve-tokens
    # cannot move, so the tiers would only change the advertised length
    rec = json.loads(json.dumps(DENSE_REC))
    rec["flags"] += ["--num-tokens", "270336"]
    m = FakeManager()
    serve = FakeServe(m, effects={"--kv-reserve-tokens=32768": (1.0, 1.0, 1.0)})
    r = _run(_job(tmp_path, m, serve, rec=rec))["result"]
    assert not any((t.get("key") or "").startswith("context_") for t in r["trials"])
    assert dict(tuner.parse_flags(r["args"]))["--num-tokens"] == "270336"
