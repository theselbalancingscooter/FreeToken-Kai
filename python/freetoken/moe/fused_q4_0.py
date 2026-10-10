"""Q4_0 entry point kept for its callers and tests; the kernel path lives in ``fused_gguf``."""

from __future__ import annotations

import torch

from freetoken.models.gguf.dequant import GGML_Q4_0

from .fused_gguf import fused_experts_gguf


def fused_experts_gguf_q4_0(
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,  # [num_slots, 2I, H//32*18] uint8
    down_q: torch.Tensor,  # [num_slots, H, I//32*18] uint8
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    return fused_experts_gguf(
        hidden_states, gate_up_q, down_q, topk_weights, topk_ids, activation, GGML_Q4_0
    )


__all__ = ["fused_experts_gguf_q4_0"]
