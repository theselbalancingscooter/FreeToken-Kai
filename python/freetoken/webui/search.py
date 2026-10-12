"""What the benchmark tries on the running model, and what it deliberately leaves alone.

Every ``ft serve`` flag that can change speed is either a candidate here or in ``SKIPPED`` with the
reason it is not measured. A candidate is one change against the best settings found so far; the
tuner keeps it only when the measurement says so (``keep``). Which candidates apply depends on the
model (MoE, GDN, MTP head, expert format), the host (GPUs, their links, RAM) and the base flags.

"standard" runs the candidates that moved speed in this fork's measurements (an RTX 2060, two RTX 3060s);
"thorough" adds the ones that have not, or only rarely, so a machine unlike ours gets the chance.

stdlib only: the manager imports this."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field

# the expert formats whose kernels --quant-backend can pick, as ft serve names them
MOE_KERNELS = {"nvfp4": ("triton", "marlin", "b12x"), "mxfp4": ("triton", "triton_gptoss")}

# Candidates added after results stopped recording the plan they chose from (``plan_keys``, since
# 2026-10-12). A pickup from such a result must not take a group another member closed as closing
# these: the result never saw them. Newer results carry their plan, and this list stays as it is.
ADDED_WITHOUT_PLAN_RECORD = frozenset({"dense_q8", "dense_fp8", "minp_06", "draft_q4"})
LINEAR_KERNELS = {"nvfp4": ("marlin", "triton")}


@dataclass
class Candidate:
    key: str
    changes: dict
    drop: tuple = ()
    # which figure has to improve: "gen" (generation, weighted by the chosen use), "prefill", or
    # "either" (one of them by its gain while the other holds)
    goal: str = "gen"
    gain: float = 1.03
    hold_gen: float = 0.97
    hold_prefill: float = 0.90
    thorough: bool = False
    after: str | tuple | None = None  # only when this earlier candidate (one of these) was kept
    what_ja: str = ""
    what_en: str = ""
    code: bool = False  # generation on code is measured too (MTP acceptance depends on the text)
    # candidates that undo each other share a group: once one is tried, the others are not
    group: str = ""

    def __post_init__(self) -> None:
        self.group = self.group or self.key

    def change(self) -> dict:
        if self.changes:
            flag, value = next(iter(self.changes.items()))
            return {"flag": flag, "value": value}
        return {"flag": self.drop[0], "value": None, "removed": True}

    def opened_by(self, kept: set) -> bool:
        """Whether the candidate this one waits for (any of them, for a tuple) was kept."""
        if self.after is None:
            return True
        after = self.after if isinstance(self.after, tuple) else (self.after,)
        return any(a in kept for a in after)


def no_fp8_tensor_cores(hw: dict) -> bool:
    """True when every measured GPU predates fp8 tensor cores (sm_89 Ada; Ampere and Turing have
    none), or when the measurement has no GPU list -- this fork's measured hosts are such cards."""
    caps = [g.get("compute_cap") for g in ((hw or {}).get("measurements") or {}).get("gpus") or [] if g.get("compute_cap")]
    return not caps or max(caps) < 8.9


def flag_value(args: list[str], flag: str) -> str | None:
    for i, a in enumerate(args):
        if a == flag:
            return args[i + 1] if i + 1 < len(args) and not args[i + 1].startswith("--") else ""
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def merge_changes(args: list[str], changes: dict) -> dict:
    """--quant-backend holds one entry per kernel table: a new entry joins the kept ones."""
    if "--quant-backend" not in changes:
        return changes
    entries = {}
    for item in (flag_value(args, "--quant-backend") or "").split(",") + [changes["--quant-backend"] or ""]:
        if "=" in item:
            table, name = item.split("=", 1)
            entries[table.strip()] = name.strip()
    return dict(changes, **{"--quant-backend": ",".join(f"{t}={n}" for t, n in sorted(entries.items()))})


def expert_format(quant: str | None) -> str | None:
    q = (quant or "").lower()
    if q in ("modelopt", "nvfp4", "modelopt_fp4"):
        return "nvfp4"
    if q == "mxfp4":
        return "mxfp4"
    return None


def installed_modules() -> set[str]:
    """The optional packages some kernels need; ft serve runs in the same environment as ft mgr."""
    out = set()
    for name in ("vllm", "flashinfer"):
        try:
            if importlib.util.find_spec(name) is not None:
                out.add(name)
        except (ImportError, ValueError):
            pass
    return out


def kernel_unusable(name: str, hw: dict, modules: set[str]) -> tuple[str, str] | None:
    """Why a kernel cannot run on this PC (the same checks the engine makes when it loads), or None."""
    caps = [g.get("compute_cap") for g in ((hw or {}).get("measurements") or {}).get("gpus") or [] if g.get("compute_cap")]
    if name == "marlin" and "vllm" not in modules:
        return "vLLM が入っていないので使えない。", "vLLM is not installed."
    if name == "b12x":
        if caps and min(caps) < 12.0:
            return f"sm_120 以上の GPU が必要（この GPU は sm_{int(min(caps) * 10)}）。", f"Needs an sm_120 GPU or newer (this one is sm_{int(min(caps) * 10)})."
        if "flashinfer" not in modules:
            return "flashinfer が入っていないので使えない。", "flashinfer is not installed."
    return None


def offload_unpinnable(hw: dict) -> tuple[str, str] | None:
    """Why a plain ``offload`` start cannot happen on this host, when it cannot.

    ``offload`` page-locks every expert bank, so a host whose measured cap is short of them
    refuses the start; the candidate is then not worth a trial (FreeToken-Kai#2). The figures
    come from the hardware measurement, which measures the cap before anything else pins."""
    pin = ((hw or {}).get("measurements") or {}).get("pin") or {}
    cap, banks = pin.get("cap_bytes"), pin.get("banks_bytes")
    if not cap or not banks or banks <= cap:
        return None
    gib = 1 << 30
    return (f"このマシンがページロックできる RAM は実測 {cap / gib:.2f} GiB で、エキスパート {banks / gib:.1f} GiB を"
            "全部固定する offload は起動しないので試さない。",
            f"This machine page-locks {cap / gib:.2f} GiB (measured), so a plain offload start, which pins all "
            f"{banks / gib:.1f} GiB of experts, does not start and is not tried.")


# Share of the expert banks the measured page-lock cap has to cover before the PCIe fetch is
# worth keeping. Below it, only a couple of layers can fetch at all and the fetch loses to the
# CPU; above it the fetch wins. The two measured points are 11% (loses) and 99% (wins), so this
# sits between them -- guides/58 7.3.
PIN_FETCH_SHARE = 1 / 3


def pin_starves_the_fetch(hw: dict) -> tuple[str, str] | None:
    """Why fetching nothing deserves a standard trial on this host, when it does.

    A miss is gathered straight out of page-locked host memory, so a cap far below the banks
    leaves only a couple of layers able to fetch at all. Measured on a 2060 with the cap forced
    to 1 GiB against 9.47 GiB of experts, fetching from those two layers was 12% SLOWER than
    fetching nothing (12.64 vs 14.17 tok/s); with the cap covering the banks the same flag loses
    (16.78 vs 14). It is the ratio that decides, which is why this is a gate and not a default."""
    pin = ((hw or {}).get("measurements") or {}).get("pin") or {}
    cap, banks = pin.get("cap_bytes"), pin.get("banks_bytes")
    if not cap or not banks or cap >= banks * PIN_FETCH_SHARE:
        return None
    gib = 1 << 30
    return (f"ページロックできる RAM は実測 {cap / gib:.2f} GiB で、エキスパート {banks / gib:.1f} GiB の"
            f"ごく一部しか GPU から読めない。取り寄せる層が数層しか無いときは、取り寄せないほうが速いことがある。",
            f"This machine page-locks {cap / gib:.2f} GiB (measured) of the {banks / gib:.1f} GiB of experts, so only a "
            f"couple of layers can fetch at all -- and with that few, not fetching can be faster.")


def unavailable(facts: dict, hw: dict, modules: set[str] | None = None) -> list[dict]:
    """What was left out of the plan, for the page's list of what was not measured."""
    modules = installed_modules() if modules is None else modules
    fmt = expert_format(facts.get("quant")) if facts.get("num_experts") else None
    out = []
    if facts.get("num_experts"):
        why = offload_unpinnable(hw)
        if why:
            out.append({"flag": "--moe-strategy offload", "why": why[0], "why_en": why[1]})
    for kind, table in (("moe", MOE_KERNELS), ("linear", LINEAR_KERNELS)):
        for name in table.get(fmt or "", ()):
            why = kernel_unusable(name, hw, modules)
            if why:
                out.append({"flag": f"--quant-backend {kind}.{fmt}={name}", "why": why[0], "why_en": why[1]})
    return out


def plan(args: list[str], facts: dict, hw: dict, mode: str = "standard", modules: set[str] | None = None) -> list[Candidate]:
    modules = installed_modules() if modules is None else modules
    m = (hw or {}).get("measurements") or {}
    is_moe = bool(facts.get("num_experts"))
    model_type = facts.get("model_type")
    gdn = any("linear" in str(x) for x in facts.get("layer_types") or [])
    pp = int(flag_value(args, "--pp-size") or 1)
    strategy = flag_value(args, "--moe-strategy")
    out: list[Candidate] = []

    def add(c: Candidate) -> None:
        if mode == "thorough" or not c.thorough:
            out.append(c)

    if is_moe:
        # --- how the experts run: the kernel bench cannot see the whole step, so both are started
        if strategy == "hybrid" and not offload_unpinnable(hw):
            add(Candidate("strategy_offload", {"--moe-strategy": "offload"}, drop=("--moe-cpu-layers", "--moe-cpu-threads"), group="strategy",
                          what_ja="足りないエキスパートを CPU で計算せず、すべて GPU に送る（offload）",
                          what_en="send every missing expert to the GPU instead of computing some on the CPU (offload)"))
        elif strategy in (None, "offload", "auto"):
            add(Candidate("strategy_hybrid", {"--moe-strategy": "hybrid", "--moe-cpu-layers": "auto"}, group="strategy",
                          what_ja="足りないエキスパートの一部を CPU で計算する（hybrid）",
                          what_en="compute some of the missing experts on the CPU (hybrid)"))
        cpu = m.get("cpu_moe") or {}
        if strategy == "hybrid" and cpu.get("cores") and str(cpu["cores"]) != flag_value(args, "--moe-cpu-threads"):
            add(Candidate("threads_all", {"--moe-cpu-threads": str(cpu["cores"])},
                          what_ja=f"CPU スレッドを全物理コア（{cpu['cores']}）にする。カーネル単体では頭打ちでも、1 ステップ全体では違うことがある",
                          what_en=f"use every physical core ({cpu['cores']}) for the CPU experts; the whole step can differ from the kernel alone"))
        threads = flag_value(args, "--moe-cpu-threads")
        if strategy == "hybrid" and threads and threads.isdigit() and int(threads) > 2:
            fewer = str(int(threads) - 1)
            add(Candidate("threads_fewer", {"--moe-cpu-threads": fewer},
                          what_ja=f"CPU スレッドを 1 つ減らして {fewer} にする（空いたコアで GPU への指示や転送が待たされなくなることがある）",
                          what_en=f"one CPU thread fewer ({fewer}): the core left free can keep GPU launches and transfers from waiting"))
        if strategy == "hybrid":
            starved = pin_starves_the_fetch(hw)
            add(Candidate("fetch_none", {"--moe-hybrid-max-fetch": "0"}, thorough=starved is None,
                          what_ja="足りないエキスパートを GPU へ送らず、すべて CPU で計算する"
                                  + (f"。{starved[0]}" if starved else ""),
                          what_en="fetch no missing expert over PCIe: the CPU computes them all"
                                  + (f" {starved[1]}" if starved else "")))

        # --- kernels for the expert format
        fmt = expert_format(facts.get("quant"))
        # experts read from a GGUF (--moe-expert-gguf) are in the GGUF's format, not the checkpoint's:
        # a kernel for the checkpoint's expert format is then picked for nothing. Two RTX 3060s with the
        # Q2_0 experts kept moe.nvfp4=triton on three runs that differed only by the swing of MTP generation
        gguf_experts = flag_value(args, "--moe-expert-gguf") is not None
        for name in MOE_KERNELS.get(fmt or "", ()) if not gguf_experts else ():
            if kernel_unusable(name, hw, modules):
                continue
            add(Candidate(f"kernel_moe_{name}", {"--quant-backend": f"moe.{fmt}={name}"}, goal="either",
                          what_ja=f"エキスパートの {fmt} カーネルを {name} にする", what_en=f"run the {fmt} experts with the {name} kernel"))
        for name in LINEAR_KERNELS.get(fmt or "", ()):
            if kernel_unusable(name, hw, modules):
                continue
            add(Candidate(f"kernel_linear_{name}", {"--quant-backend": f"linear.{fmt}={name}"}, goal="either", thorough=True,
                          what_ja=f"エキスパート以外の {fmt} 層のカーネルを {name} にする（その形式の層があるときだけ差が出る）",
                          what_en=f"run the non-expert {fmt} layers with the {name} kernel (only matters when there are such layers)"))

        # --- the KV cache: VRAM it frees goes to the expert cache, its decode costs time
        kv = flag_value(args, "--kv-cache-dtype")
        if kv in ("q8_0", "q4_0"):
            add(Candidate("kv_16bit", {}, drop=("--kv-cache-dtype",), group="kv",
                          what_ja="KV を量子化しない（16bit）。KV の読み出しは速いが、エキスパートの枠は減る",
                          what_en="keep the KV cache at 16 bits: faster to read, fewer expert slots"))
        else:
            add(Candidate("kv_q8", {"--kv-cache-dtype": "q8_0"}, group="kv",
                          what_ja="KV を q8_0 にして、空いた VRAM をエキスパートの枠に回す",
                          what_en="store the KV cache as q8_0 and give the VRAM to the expert cache"))
        if model_type != "gpt_oss" and kv != "q4_0":
            add(Candidate("kv_q4", {"--kv-cache-dtype": "q4_0"}, thorough=True,
                          what_ja="KV を q4_0 にする（さらに小さいが、読み出しの展開が重い）",
                          what_en="store the KV cache as q4_0 (smaller still, heavier to decode)"))

        # --- prefill: the overlap of two expert buffers, and resident rows copied on the GPU
        if flag_value(args, "--disable-moe-prefill-overlap") is not None:
            add(Candidate("overlap_on", {}, drop=("--disable-moe-prefill-overlap",), goal="prefill", gain=1.05,
                          what_ja="プロンプト処理の 2 バッファの重ね合わせを有効にする（VRAM が足りれば速くなる）",
                          what_en="enable the two-buffer overlap for prefill expert copies (faster when the VRAM allows)"))
        add(Candidate("hit_d2d", {"--moe-prefill-hit-d2d": None}, goal="prefill", gain=1.05, thorough=True,
                      what_ja="プロンプト処理で GPU に載っているエキスパートは GPU 内でコピーし、足りない分だけ送る",
                      what_en="copy the experts already on the GPU device-side during prefill and stream only the misses"))

        if flag_value(args, "--moe-bank-ram") is not None:
            add(Candidate("bank_prefetch", {"--moe-bank-prefetch": None},
                          what_ja="CPU が計算する前に、SSD 側のエキスパートの行を先読みさせる",
                          what_en="ask the kernel for the SSD-side expert rows before the CPU computes a layer"))

    # --- prefill width
    if flag_value(args, "--prefill-chunk-budget") is None:
        add(Candidate("budget_075", {"--prefill-chunk-budget": "0.75"}, goal="prefill", gain=1.05,
                      what_ja="プロンプト処理の 1 チャンクが空き VRAM の 75% まで使えるようにして、チャンクを広げる",
                      what_en="let one prefill chunk take 75% of the free VRAM, so chunks are wider"))
    # 90% is not tried: it never won (two RTX 3060s: 597 against 669 tok/s on 2026-09-17, 614 against
    # 662 on 2026-10-13), and with the Q2_0 experts the backend died in the middle of a 16k-token prompt
    pieces = flag_value(args, "--prefill-mixer-pieces")
    if gdn and (model_type == "qwen4_exp" or (model_type == "qwen3_5_moe" and pp == 1)):
        if pieces != "4":
            add(Candidate("pieces_4", {"--prefill-mixer-pieces": "4", "--max-prefill-length": "16384"}, goal="prefill", gain=1.05, group="pieces",
                          what_ja="GDN と attention を 4 つに分け、チャンクの上限を 16384 に上げる（2060・Ornith で 722 tok/s の実測）",
                          what_en="run GDN and attention in four pieces with a 16384-token chunk ceiling (722 tok/s on a 2060 with Ornith)"))
        if pieces:
            add(Candidate("pieces_1", {}, drop=("--prefill-mixer-pieces",), goal="prefill", gain=1.05, thorough=True, group="pieces",
                          what_ja="GDN と attention を分けない（片分けが逆効果になる機械の確認）",
                          what_en="do not split GDN and attention (for a machine where the pieces cost more than they give)"))
    add(Candidate("max_prefill_16k", {"--max-prefill-length": "16384"}, goal="prefill", gain=1.05, thorough=True,
                  what_ja="チャンクの上限を 16384 に上げる", what_en="raise the prefill chunk ceiling to 16384 tokens"))

    # --- memory placement
    if flag_value(args, "--host-embedding") is not None:
        add(Candidate("no_host_embedding", {}, drop=("--host-embedding",), goal="either", thorough=True,
                      what_ja="埋め込み表を GPU に置く（VRAM を使うが、読み出しは GPU 内で済む）",
                      what_en="keep the embedding table on the GPU (uses VRAM, no PCIe gather)"))
    if model_type == "qwen4_exp":
        # the dense weights: fp8 (W8A16, e4m3 expanded in the kernel), Q8_0 (int8 dp4a), or bf16.
        # On a card without fp8 tensor cores the Q8_0 kernels run a verify window's few-row GEMMs
        # at the memory bandwidth where fp8 ran at half of it: two RTX 3060s, Flash-Next with
        # --spec-mtp 3, generation +2..6 tok/s and prefill +3..6% (guides/70 4.15); on Ada and
        # newer it is a thorough trial
        dq = flag_value(args, "--dense-quant")
        q8_standard = no_fp8_tensor_cores(hw)
        if dq != "q8_0":
            add(Candidate("dense_q8", {"--dense-quant": "q8_0"}, goal="either", group="dense", thorough=not q8_standard,
                          what_ja="エキスパート以外の重みを ggml Q8_0（int8）にして、同梱の int8 カーネルで動かす"
                                  "（fp8 のテンソルコアが無いカードでは fp8 より速い: 3060 で生成 +2〜6 tok/s）",
                          what_en="store the non-expert weights as ggml Q8_0 (int8) and run them through the vendored int8 kernels "
                                  "(faster than fp8 on a card without fp8 tensor cores: +2..6 tok/s on a 3060)"))
        if dq == "q8_0":
            add(Candidate("dense_fp8", {"--dense-quant": "fp8"}, goal="either", group="dense", thorough=q8_standard,
                          what_ja="エキスパート以外の重みを fp8 にする（fp8 のテンソルコアがあるカード向け）",
                          what_en="store the non-expert weights as fp8 (for a card with fp8 tensor cores)"))
        if dq in ("fp8", "q8_0"):
            add(Candidate("dense_bf16", {}, drop=("--dense-quant",), goal="either", thorough=True, group="dense",
                          what_ja="エキスパート以外の重みを量子化せず bf16 のままにする",
                          what_en="keep the non-expert weights at bf16"))
        ram, weights = (facts.get("ram_total") or 0), (facts.get("weight_bytes") or 0)
        if ram and weights and ram > weights * 1.15:
            add(Candidate("ple_pinned", {"--ple-backend": "pinned"},
                          what_ja="PLE の表を RAM に常駐させる（RAM に余裕があるときだけ）",
                          what_en="keep the PLE table in page-locked RAM (only when RAM has room)"))

    # --- two GPUs: where the split falls, and how the ranks hand chunks over
    if pp > 1:
        layers = facts.get("num_layers") or 0
        links = [((m.get("pcie") or {}).get(g) or {}).get("h2d_gbs") for g in (flag_value(args, "--gpu") or "").split(",") if g]
        if pp == 2 and layers and len(links) == 2 and all(links):
            # give the rank on the faster link more layers: its experts cross PCIe faster
            toward = 1 if links[0] > links[1] * 1.2 else -1 if links[1] > links[0] * 1.2 else 0
            if toward:
                for k, thorough in ((1, False), (2, True)):
                    split = layers // 2 + toward * k
                    add(Candidate(f"pp_layers_{split}", {"--pp-layers": str(split)}, goal="either", thorough=thorough,
                                  what_ja=f"層の分け方を {split} / {layers - split} にして、速いリンクの GPU に多く載せる",
                                  what_en=f"split the layers {split} / {layers - split}, more on the GPU with the faster link"))
        add(Candidate("send_ahead_3", {"--pp-send-ahead": "3"}, goal="prefill", gain=1.05, thorough=True,
                      what_ja="1 枚目が 2 枚目へ残差を 3 チャンク先まで送れるようにする",
                      what_en="let the first rank have three residual streams in flight to the next"))
        if model_type == "qwen4_exp":
            add(Candidate("prefill_group_2", {"--pp-prefill-group": "2"}, goal="prefill", gain=1.05, thorough=True,
                          what_ja="2 枚目がチャンクを 2 つためて層ごとにまとめて処理する",
                          what_en="let the second rank hold two chunks and run them layer by layer"))

    # --- speculative decoding with the checkpoint's MTP head: its gain depends on the text
    if facts.get("mtp") and flag_value(args, "--max-running-req") in ("1", None):
        base_k = flag_value(args, "--spec-mtp")
        # 4 for Flash-Next, where it was measured: with the Q2_0 experts on two RTX 3060s a window of
        # 3, 4 or 5 answered at the same 38 tok/s and 4 left prompt processing the most (1,010-1,028
        # tok/s against 987-995 and 902-927)
        for k in ("3", "4", "5") if model_type == "qwen4_exp" else ("3", "5"):
            if base_k == k:
                continue
            add(Candidate(f"mtp_{k}", {"--spec-mtp": k}, code=True, hold_prefill=0.85,
                          what_ja=f"MTP で {k} トークン先まで予測して、まとめて確かめる（コードやツール呼び出しで効きやすい）",
                          what_en=f"draft {k} tokens ahead with the MTP head and verify them together (helps most on code and tool calls)"))
        # the two settings on top of a kept window (or of the base's own --spec-mtp). Measured on
        # two RTX 3060s with the Q2_0 experts: min-p 0.6 over K=5 wrote an essay that never ends at
        # 62 tok/s where fixed K=3 wrote 37 (guides/70 4.18), but on seven prompts answered to their
        # end it wrote at the same 38 tok/s as a fixed window and cost a tenth of prompt processing
        # (2026-10-13). The 4-bit draft head is only for a bf16 or Q8_0 lm_head -- packed from an
        # fp8 head it lost a fifth of the prose acceptance
        opened = None if base_k else ("mtp_3", "mtp_4", "mtp_5")
        if flag_value(args, "--spec-min-p") is None:
            add(Candidate("minp_06", {"--spec-min-p": "0.6", "--spec-mtp": "5"}, code=True, hold_prefill=0.85, after=opened,
                          what_ja="窓を 5 トークンにして、ヘッドの確信度が 0.6 を下回った最初のドラフトで窓を切る（終わらない文章では効く。最後まで答えさせた質問では固定の窓と同じだった）",
                          what_en="a 5-token window cut at the first draft the head scores below 0.6 (helps text that runs on; on prompts answered to their end it matched a fixed window)"))
        # the packed head needs a bf16 (or Q8_0) lm_head: Flash-Next's is, Ornith's is NVFP4 and the engine skips it
        if model_type == "qwen4_exp" and flag_value(args, "--spec-draft-head-quant") is None and flag_value(args, "--dense-quant") != "fp8":
            add(Candidate("draft_q4", {"--spec-draft-head-quant": "q4_0"}, code=True, hold_prefill=0.85, after=opened,
                          what_ja="ドラフトの採点を 4 bit に詰めた lm_head の複製で行う（ドラフト 1 回のヘッド読みが 1.9 → 1.1 ms。fp8 の dense とは組まない）",
                          what_en="score the drafts through a 4-bit copy of the lm_head (1.9 -> 1.1 ms per draft; not with fp8 dense)"))
    return out


# flags that can change speed but are not searched, with the reason shown on the page
SKIPPED = [
    ("--memory-ratio", "上げると速くなりうるが、あとで画面やほかのアプリが VRAM を使ったときに落ちる。その危険は測定では見えないので、0.85 のままにする。",
     "Raising it can help, but the server dies once the display or another app takes VRAM later; a benchmark cannot see that risk, so 0.85 stays."),
    ("--max-running-req", "複数の要求を同時に処理する速さ（スループット）で、1 人で使う速さではないので 1 のまま。",
     "It trades single-request speed for concurrent throughput; one person's speed is measured, so it stays 1."),
    ("--cuda-graph-max-bs", "同時処理が 1 のときは使われない。", "Unused with one running request."),
    ("--moe-cpu-layers", "WSL で GPU に固定できる RAM の上限（このベンチマークが実測して記録する）から auto が層を決める。上限を超える指定は起動できないので探さない。",
     "Under WSL auto picks the layers from the pinnable-RAM limit, which this benchmark measures and records; a list beyond it does not start, so it is not searched."),
    ("--attention-backend", "GPU とモデルに合うものを auto が選ぶ。合わないものは起動しないか結果が崩れるので探さない。",
     "auto picks the backend that fits the GPU and model; others fail to start or change the output."),
    ("--moe-bank-ram", "RAM に重みが入らないときに必要かどうかで決まる（速さの比較ではない）。",
     "Decided by whether the weights fit RAM, not by speed."),
    ("--linear-state-cache-ratio / --prefix-disk-cache / --enable-special-token-ckpt",
     "会話の続きや切り替えのときの待ち時間に効く。1 回のプロンプトの測定では差が出ないので、今は測っていない。",
     "They shorten follow-up and switched conversations; a single-prompt measurement cannot show them yet."),
    ("--moe-bank-rewarm", "アイドル中に追い出されたページを読み戻す機能で、待ち時間の測定には何分もの放置が要る。",
     "It reads evicted pages back while idle; measuring it takes minutes of idling."),
    ("--mm-encoder-weights / --image-*", "画像入力の速さと VRAM に効く。文章の測定では差が出ない。", "They affect image input, not text."),
    ("--expert-load", "起動にかかる時間だけに効く。", "Only affects load time."),
    ("--dtype", "Turing では bfloat16 が使えないので float16 に決まる（起動できなければ外す）。それ以外の GPU では auto。",
     "Turing has no bfloat16, so float16 (dropped if the model does not load); auto elsewhere."),
    ("--moe-expert-gguf", "エキスパートを 2.25 bit の GGUF から読む構成。別のファイルが要り、答えも少し変わる（10 問で 9/10）ので、速さの探索では切り替えない。"
     "「改善提案」が、同じモデルの Q2_0 GGUF が隣にあるときに提案する。",
     "The 2.25-bit expert tier reads another file and changes the answers (9 of 10 on the reference prompts), so the speed search "
     "does not switch to it; the recommendation proposes it when the model's Q2_0 GGUF sits next to the checkpoint."),
    ("--reasoning-budget", "思考の長さの上限で、速さではなく答えの出方を決める。", "A cap on the thinking, which decides how answers end, not how fast."),
]
