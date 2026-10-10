"""Grouped expert GEMM over native GGUF block banks (borrowed ggml MoE kernels).

Ports vLLM/sglang's ``_fused_moe_gguf`` MMVQ path onto FreeToken's offload-cache
interface: the experts are streamed to the GPU as packed block bytes and dequantized
*inside* ``ggml_moe_a8_vec`` -- no bf16 expert copy is materialized. We use the MMVQ
(vector) kernel for both prefill and decode: it consumes ``topk_ids`` directly (no
``moe_align_block_size`` needed) and on small batches it is the right choice anyway.
``topk_ids`` already index the streamed cache slots (decode) or the materialized layer
positions (prefill).

Formats: ``q4_0`` (gemma4 GGUF, 32-weight blocks of 18 bytes) and ``q2_0`` (Flash-Next
from the ISTA-DASLab GGUF, 64-weight blocks of 18 bytes); the ggml type picks the kernel.
"""

from __future__ import annotations

import torch

from freetoken.layers.activation import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul
from freetoken.models.gguf.dequant import GGML_Q2_0, GGML_Q4_0

_ACT = {"silu": silu_and_mul, "gelu": gelu_and_mul, "gelu_tanh": gelu_tanh_and_mul}

# bank format tag (OffloadMoeCache.quant_format) -> ggml type the kernels dispatch on
GGUF_FORMAT_TYPES = {"q4_0": GGML_Q4_0, "q2_0": GGML_Q2_0}
# the expert bank formats that carry no quant method: the cache's format tag dispatches
GGUF_EXPERT_FORMATS = frozenset(GGUF_FORMAT_TYPES)


def fused_experts_gguf(
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,  # [num_slots, 2I, row_bytes(H)] uint8
    down_q: torch.Tensor,  # [num_slots, H, row_bytes(I)] uint8
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    quant_type: int,
) -> torch.Tensor:
    from freetoken.kernel.gguf import ggml_moe_a8_vec

    act_fn = _ACT.get(activation)
    if act_fn is None:
        raise ValueError(f"unsupported MoE activation {activation!r}")

    num_tokens = hidden_states.shape[0]
    n2 = gate_up_q.shape[1]  # 2 * intermediate
    h = down_q.shape[1]  # hidden
    top_k = topk_ids.shape[1]
    qt = int(quant_type)

    # gate_up: [num_tokens*top_k, 2I] -> activation -> [num_tokens*top_k, I]
    gate_up = ggml_moe_a8_vec(hidden_states, gate_up_q, topk_ids, top_k, qt, n2, num_tokens)
    inter = act_fn(gate_up)
    # down: each of the num_tokens*top_k intermediate rows uses its own expert id.
    out = ggml_moe_a8_vec(inter, down_q, topk_ids, 1, qt, h, num_tokens * top_k)
    out = out.reshape(num_tokens, top_k, h) * topk_weights.reshape(num_tokens, top_k, 1).to(
        out.dtype
    )
    return out.sum(dim=1)


def fused_experts_gguf_format(
    fmt: str,
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,
    down_q: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    """``fused_experts_gguf`` keyed by the bank format tag."""
    return fused_experts_gguf(
        hidden_states, gate_up_q, down_q, topk_weights, topk_ids, activation, GGUF_FORMAT_TYPES[fmt]
    )


__all__ = ["GGUF_FORMAT_TYPES", "GGUF_EXPERT_FORMATS", "fused_experts_gguf", "fused_experts_gguf_format"]
