"""ggml Q8_0 weight (blocks of 32 int8 with one fp16 scale, 8.5 bits per weight), served by the
vendored ggml kernels: the MMVQ GEMV for a single row, the dense MMQ (int8 dp4a over q8_1
activations) for the few rows of a verify window, and a dequantized cuBLAS GEMM for prefill.

Why this exists next to the fp8 W8A16 method (``--dense-quant fp8``): fp8's M>1 path has to
expand e4m3 to bf16 inside the kernel, which on Ampere and Turing (no fp8 tensor cores) runs at
half the memory bandwidth or worse; Q8_0's int8 codes go to dp4a as they are, so every M up
to the MMQ limit reads the weights once at bandwidth. Same bytes as fp8, more mantissa."""

from __future__ import annotations

import os
from typing import Any

import torch

from ..registry import LayerKind, register_method
from ..scheme import QuantKind
from .base import LinearConfig, LinearKernel, LinearMethod

Q8_0_BLOCK = 32
# The vendored dense MMQ walks a row in warp steps of 4 blocks (128 weights) and the MMVQ in
# steps of its own; a row whose length is not a multiple of 128 makes the last step read past
# the row -- into the next row (harmless: the activation padding there is zero) or, for the
# last row, past the buffer (a NaN scale poisons the output). Such shapes (the hyper-connection
# up projection: K = 320) take the dequantize + cuBLAS path at every M.
Q8_0_KERNEL_STEP = 128


def mmvq_max_rows() -> int:
    """Rows up to which the per-row GEMV serves (FREETOKEN_Q8_0_MMVQ_MAX_ROWS, default 1)."""
    return int(os.environ.get("FREETOKEN_Q8_0_MMVQ_MAX_ROWS", "1") or 1)


def mmq_max_rows() -> int:
    """Rows up to which the tiled int8 MMQ serves; above it the weight is dequantized once and
    cuBLAS runs the GEMM on tensor cores (FREETOKEN_Q8_0_MMQ_MAX_ROWS, default 64)."""
    return int(os.environ.get("FREETOKEN_Q8_0_MMQ_MAX_ROWS", "64") or 64)


class GgmlQ8_0LinearKernel(LinearKernel):
    name = "ggml"

    def unusable_reason(self, cfg: LinearConfig) -> str | None:
        if cfg.in_features % Q8_0_BLOCK:
            return f"in_features {cfg.in_features} is not a multiple of the Q8_0 block ({Q8_0_BLOCK})"
        if not torch.cuda.is_available():
            return "the ggml kernels need CUDA"
        return None

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.gguf import ggml_dequantize, ggml_mul_mat_a8, ggml_mul_mat_vec_a8
        from freetoken.models.gguf.dequant import GGML_Q8_0

        *lead, K = x.shape
        x2 = x.reshape(-1, K)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        rows, N = x2.shape[0], layer.out_features
        kernels_ok = K % Q8_0_KERNEL_STEP == 0
        if kernels_ok and rows <= mmvq_max_rows():
            out = ggml_mul_mat_vec_a8(layer.weight, x2, GGML_Q8_0, N)
        elif kernels_ok and rows <= mmq_max_rows():
            out = ggml_mul_mat_a8(layer.weight, x2, GGML_Q8_0, N)
        else:
            w = ggml_dequantize(layer.weight, GGML_Q8_0, N, K, x2.dtype)
            out = x2 @ w.t()
        if layer.bias is not None:
            out = out + layer.bias.to(out.dtype)
        return out.reshape(*lead, N)


class TorchQ8_0LinearKernel(LinearKernel):
    """Dequantize in torch and matmul: the reference, and what runs without CUDA."""

    name = "torch"

    def worth_it(self, cfg: LinearConfig) -> bool:
        return False

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        from freetoken.models.gguf.dequant import dequant_q8_0

        *lead, K = x.shape
        N = layer.out_features
        w = dequant_q8_0(layer.weight, x.dtype).reshape(N, K)
        out = x.reshape(-1, K) @ w.t()
        if layer.bias is not None:
            out = out + layer.bias.to(out.dtype)
        return out.reshape(*lead, N)


@register_method(QuantKind.Q8_0, LayerKind.LINEAR)
class Q8_0LinearMethod(LinearMethod):
    candidates = (GgmlQ8_0LinearKernel, TorchQ8_0LinearKernel)

    def create_weights(self, layer: Any) -> None:
        from freetoken.models.gguf.dequant import GGML_Q8_0, row_bytes

        g = self.cfg
        # the packed rows live under ``weight``: the loader quantizes the checkpoint's bf16
        # tensor into this shape (engine._quantize_at_load) and the merged loaders concatenate
        # parts along dim 0, which packed rows allow
        layer.weight = torch.empty(g.out_features, row_bytes(g.in_features, GGML_Q8_0), dtype=torch.uint8)
