"""``--spec-draft-head-quant``: a lower-precision copy of the lm_head for the MTP drafts.

Every drafted token scores the draft head's hidden state through the full vocabulary: on
Qwen3.8-Flash-Next that is a 248k x 2560 read per draft (fp8 0.63 GB, 1.9 ms on an RTX 3060),
K times per verify window. The drafts only decide what the target verifies -- a wrong draft
costs acceptance, never correctness -- so the draft-side head can be coarser than the
verifying one: Q4_0 halves the bytes (1.08 ms on the same card), Q8_0 keeps fp8's bytes with
more mantissa. The target's own lm_head is untouched, so the output does not change.

Heads this can read: bf16 / fp16, per-row fp8 (``--dense-quant fp8``) and Q8_0 rows
(``--dense-quant q8_0``). A head the checkpoint quantized itself (ModelOpt NVFP4, laid out for
its kernel) is left alone: the engine logs it and the drafts use the model's head.
"""

from __future__ import annotations

import torch

_KINDS = ("q4_0", "q8_0")


class UnsupportedHead(ValueError):
    pass


def pack_draft_head(lm_head, kind: str, *, chunk_rows: int = 8192) -> torch.Tensor:
    """Pack the head's weight into ggml ``kind`` rows on the weight's device: ``[vocab, row_bytes]`` uint8."""
    from freetoken.models.gguf.dequant import GGML_Q8_0, dequant_q8_0, quantize_q4_0, quantize_q8_0, row_bytes

    if kind not in _KINDS:
        raise ValueError(f"--spec-draft-head-quant {kind!r}: one of {_KINDS}")
    w = lm_head.weight
    V, K = int(lm_head.out_features), int(lm_head.in_features)
    scale = getattr(lm_head, "weight_scale", None)
    if w.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        if scale is None or tuple(w.shape) != (V, K):
            raise UnsupportedHead(f"fp8 head of shape {tuple(w.shape)} without a per-row scale")
        storage = "fp8"
    elif w.dtype == torch.uint8:
        if tuple(w.shape) != (V, row_bytes(K, GGML_Q8_0)):
            raise UnsupportedHead(f"a packed head of shape {tuple(w.shape)} (the checkpoint's own quantization)")
        storage = "q8_0"
    elif w.is_floating_point() and tuple(w.shape) == (V, K):
        storage = "float"
    else:
        raise UnsupportedHead(f"head weight {w.dtype} {tuple(w.shape)}")
    quant = quantize_q4_0 if kind == "q4_0" else quantize_q8_0
    out = torch.empty(V, row_bytes(K, 2 if kind == "q4_0" else GGML_Q8_0), dtype=torch.uint8, device=w.device)
    for r0 in range(0, V, chunk_rows):
        r1 = min(V, r0 + chunk_rows)
        if storage == "q8_0":
            rows = dequant_q8_0(w[r0:r1], torch.float32).reshape(r1 - r0, K)
        elif storage == "fp8":
            rows = w[r0:r1].to(torch.float32) * scale[r0:r1].to(torch.float32)[:, None]
        else:
            rows = w[r0:r1].to(torch.float32)
        out[r0:r1] = quant(rows)
    return out


def draft_logits(packed: torch.Tensor, kind: str, h: torch.Tensor, vocab: int) -> torch.Tensor:
    """``h [rows, K] @ head^T`` through the packed draft head (MMVQ for the few rows a draft has)."""
    from freetoken.kernel.gguf import ggml_mul_mat_vec_a8
    from freetoken.models.gguf.dequant import GGML_Q4_0, GGML_Q8_0

    qt = GGML_Q4_0 if kind == "q4_0" else GGML_Q8_0
    h2 = h.reshape(-1, h.shape[-1])
    if not h2.is_contiguous():
        h2 = h2.contiguous()
    return ggml_mul_mat_vec_a8(packed, h2, qt, vocab)
