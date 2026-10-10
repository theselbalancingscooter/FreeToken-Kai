# GGUF experts: the 2.25-bit bank (2026-10-10) and what stopped the 3-bit idea (2026-09-07)

## Q2_0 experts from a GGUF: `--moe-expert-gguf`

Since 2026-10-10 the fork serves Qwen3.8-Flash-Next with its routed experts read from a llama.cpp
checkpoint while everything else -- the dense projections, the PLE table, the vision tower --
keeps coming from the safetensors checkpoint named by `--model`:

```bash
ft serve --model models/Qwen3.8-Flash-Next-NVFP4 \
  --moe-expert-gguf models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --moe-strategy hybrid ...   # the rest as for the NVFP4 checkpoint
```

What it is for: [ISTA-DASLab's GSQ-RCO builds](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
quantise the experts to ggml's **Q2_0** -- 64 weights per fp16 scale, codes -1 / 0 / +1,
**2.25 bits per weight** -- and calibrate them. The expert banks shrink from 63.5 GiB (NVFP4) to
34 GB, the bytes a decode step reads per token from 1.18 GB to 0.66 GB, and the same GPU cache
holds twice the experts. The experts of every layer in these builds share one ggml type, which is
the condition the section below explains. Only shard 1 is read (shard 2 holds the PLE table, which
the fork takes from the safetensors checkpoint in FP8 instead).

How it works: the GGUF's block bytes **are** the banks -- nothing is repacked -- and the kernels
dequantise inside the K-loop: on the GPU the borrowed ggml MMVQ kernel for a single decoded token
(`vec_dot_q2_0_q8_1`, W2A8) and a tiled MMQ kernel for everything wider (the `--spec-mtp` verify
windows and the prefill chunks: the tokens are sorted by expert and each expert's rows are read
once per tile of 4 tokens, or of 32 from 256 rows up), and the CPU executor's `q2_0_dot_i8_*`
(AVX-VNNI / AVX2 / scalar, W2A8 against Q8_0 activations). The checkpoint's own expert scheme steps aside for these layers;
`--dense-quant fp8` and the MTP draft head still work, the head's bf16 experts being rounded to
Q2_0 with ggml's reference quantizer (amax delta) at load. The GGUF is parsed by a small reader of
the fork's own (`models/gguf/rawreader.py`): gguf-py 0.19.0 predates Q2_0 and refuses the file.
Q4_0 experts load through the same path.

What it costs: quality. Measured on two RTX 3060s against the NVFP4 build on the same ten
prompts, the Q2_0 build answered 9 of 10 (the miss: a translation that kept English terms),
every code task included, but thought 35% longer in total; ISTA-DASLab reports LiveCodeBench
81.1 against 87.4 for bf16, AIME and GPQA unchanged. Treat it as the fast tier and NVFP4 as the
accurate one. Two things follow from the 2-bit rounding. A greedy (temperature 0) request can
think in circles until `max_tokens` -- one of the ten prompts did so on one kernel path and not
the other, the two differing by one bf16 ulp. `--reasoning-budget N` (or a request's
`reasoning_budget`) caps the tokens between `<think>` and `</think>`: at the cap the sampler
returns `</think>` and the model answers from what it has; 0 (the default) leaves it unlimited.
And prefer the checkpoint's recommended sampling (`--sampling-defaults model`, the default) to
greedy for the same reason.

Speed on two RTX 3060s (`--pp-size 2`, hybrid, `--spec-mtp 5`): decode 28 tok/s (NVFP4: 19-20), the
verify window 108 ms of which the experts are 11 ms and the dense fixed cost 22; prefill of a
25k-token prompt 829 tok/s with the 4-token tile (NVFP4: 575-700; MMVQ alone: 264), compute-bound
with the x4 transfer of 16.5 GiB per chunk hidden behind it.

Refused: experts whose ggml type varies by layer (every "UD" community build), IQ types (codebooks),
and anything that is not Q2_0 or Q4_0.

The rest of this page is the 2026-09-07 survey that found the blockers above, kept for the numbers.

**The idea.** A 125B MoE with NVFP4 experts needs 63 GiB of host RAM for the banks. The published
GGUF quantisations of the same checkpoint are much smaller — UD-Q2_K_XL is 78.9 GB in total against
135 GB. If FreeToken could serve those, the RAM problem would be solved by arithmetic rather than by
the file mapping in [bank-ram.md](bank-ram.md).

**The conclusion (2026-09-07).** It is possible, it is not cheap, and the published quantisations do
not load as they are. If you want smaller experts today, `--moe-bank-ram` is the shorter road.
(2026-10-10: a uniform Q2_0 build appeared and the loader above serves it.)

## The blocker you will hit first

Every published UD-series quantisation of Qwen3.8-Flash-Next has **expert tensors whose ggml type
differs from layer to layer**. From the headers of `unsloth/Qwen3.8-Flash-Next-GGUF` (read from the
first 12 MiB of each shard; the totals match the sizes Hugging Face reports), UD-Q3_K_XL is:

```
ffn_gate_exps   IQ3_XXS x47,  IQ4_XS x1   (layer 2)
ffn_up_exps     IQ3_XXS x47,  IQ4_XS x1   (layer 2)
ffn_down_exps   IQ4_NL  x43,  Q8_0   x5   (layers 2, 4, 30, 46, 47)
```

The expert slot pool is one allocation across all layers, so a bank whose type varies by layer
raises `NotImplementedError`. This is not a property of one bad file: the "UD" (unsloth dynamic)
method deliberately raises precision on the layers it measures as sensitive, so **every** UD build
has this shape. A uniform quantisation would load, but you would be making it yourself.

## Sizes, if you get past that

| Build | Experts | bpw | PLE / Ngram | dense | total |
|---|---|---|---|---|---|
| NVFP4 (what this fork runs) | **68.1 GB** | 4.51 | 51.2 GB (fp8) | — | 135 GB |
| UD-Q2_K_XL | 46.1 GB | 3.05 | 28.8 GB (IQ4_NL) | 4.0 | 78.9 |
| UD-Q3_K_XL | 55.8 GB | 3.70 | 28.8 GB | 5.4 | 90.0 |
| UD-IQ4_XS | 59.5 GB | 3.94 | 28.8 GB | 5.4 | 93.7 |
| UD-Q4_K_XL | 77.0 GB | 5.10 | 28.8 GB | 5.5 | 111.3 |

Worth noting that most of the saving is in the PLE table (51.2 -> 28.8 GB), not in the experts.

## The rest of the bill

Upstream has GGUF support in flight —
[FlashML-org/FreeToken#131](https://github.com/FlashML-org/FreeToken/pull/131) (`feat/generic-gguf`,
opened 2026-08-24, 37 files and 5,920 lines, still open as of 2026-09-07). It dispatches all 21 GGUF
types, adds GGUF expert banks and K-quant CPU kernels, and has loaders for qwen3_5_moe, qwen3_moe
and deepseek_v4. Against the use case above:

| Constraint | What it costs you |
|---|---|
| No `qwen4exp` (Flash-Next) among the supported architectures | Write the loader; the qwen3_5_moe one is 1,261 lines |
| Per-layer type variation raises `NotImplementedError` | See above — rules out every published UD build |
| CPU kernels exist only for Q4_0 / Q4_K / Q6_K | Anything else forces `--moe-backend offload` |
| `gate_up` and `down` must share a type for the CPU path | Kills the Q4_K_M family, where `down` is Q6_K |
| CPU kernels are scalar reference implementations (AVX2/VNNI explicitly deferred) | This is the one to worry about: hybrid decode on this hardware needs ~44 GB/s from the CPU MoE path, and a scalar kernel will not be the same order |
| TP=1 only, no parallel reader, NextN/MTP dropped | Extra work to put it on the layer split |

The last row is the quiet one. Shrinking the experts helps only if the CPU can still dequantise and
multiply them fast enough; on this host that rate, not capacity, is what caps decode. See the
`ft bench bw` section of [kai.md](kai.md).

## If you are going to try anyway

Read the headers before you download 90 GB. `tools/gguf_probe.py` in the fork's working notes takes
the first 12 MiB of each shard and prints the per-tensor ggml types and the totals, which is how the
table above was made. A build whose expert tensors are all one type is the only kind worth pursuing.

The borrowed ggml kernels (`kernel/gguf.py`) are compiled on first use, with `clang++` as nvcc's
host compiler: install clang (`apt install clang`). gcc 13 to 15 stop on torch 2.11's
`ATen/core/List_inl.h` (a missing `typename`), so without clang the first GGUF op fails to build.
`FREETOKEN_GGUF_HOST_CXX` picks another compiler.
