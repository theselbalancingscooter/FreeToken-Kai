"""--dense-quant: serve the projections a checkpoint left bf16 as fp8 or ggml Q8_0, quantized at load.

Every other QuantConfig answers what the checkpoint *stores*. These answer what the
deployment wants: on a card too small for the bf16 dense weights, a projection the checkpoint
never quantized is still worth halving. The weights themselves are converted in the loader
(``engine._quantize_at_load``), which sees an fp8 (or packed uint8) buffer where a bf16 tensor
arrived. ``q8_0`` keeps the same bytes as fp8 but feeds int8 dp4a kernels, so the few-row
GEMMs of a verify window read the weights at bandwidth on cards without fp8 tensor cores.
"""

from __future__ import annotations

import os
from typing import Any

from .configs.base import NoQuantConfig, QuantConfig, Stored
from .linear import LinearConfig
from .names import is_routed_expert, name_set
from .quant_backend import get_quant_backend
from .registry import LayerKind, method_class
from .scheme import QuantKind, QuantScheme, fp8_tensor_scheme, q8_0_scheme

# one fp32 scale per output row: a whole-tensor scale loses too much on a fused projection,
# and the W8A16 kernels read a per-row scale either way
AT_LOAD_FP8 = fp8_tensor_scheme("float", per_row=True)
AT_LOAD_Q8_0 = q8_0_scheme()

# Projections that stay bf16: the MoE router, the hyper-connection mixer's skinny GEMMs, the QSA
# indexer, the PLE table's key/value projections and the GatedDeltaNet b/a gates decide which
# experts run, how streams mix, which keys are scored, which n-gram rows are read and how much
# state decays. They are small, so they buy little VRAM, and they are the last place to spend
# accuracy. The vision tower stays too: its blocks are streamed from pinned host banks, so fp8
# would save host RAM, not VRAM, and every image feature the model sees comes out of it.
def _keep_bf16_patterns() -> tuple[str, ...]:
    pats = [
        "*.gate", "*.shared_expert_gate", "*hyper_connection*", "*.indexer", "*.ple",
        "*.in_proj_ba", "*.in_proj_b", "*.in_proj_a",
        "visual", "*.visual",
    ]
    # FREETOKEN_DENSE_QUANT_HC=1 quantizes the hyper-connection GEMMs too (bf16 cutlass, 3.5 ms
    # of a verify window per rank on an RTX 3060); FREETOKEN_DENSE_QUANT_ROUTER=1 the routers.
    # Both trade accuracy for bytes in the places that decide where tokens go -- A/B material.
    if os.environ.get("FREETOKEN_DENSE_QUANT_HC") == "1":
        pats.remove("*hyper_connection*")
    if os.environ.get("FREETOKEN_DENSE_QUANT_ROUTER") == "1":
        pats.remove("*.gate")
        pats.remove("*.shared_expert_gate")
    return tuple(pats)


KEEP_BF16 = name_set(_keep_bf16_patterns())


class LoadTimeQuantConfig(QuantConfig):
    """The checkpoint's config, with every unquantized dense projection reported as ``AT_LOAD``."""

    dialect = "dense-quant"
    AT_LOAD: QuantScheme = AT_LOAD_FP8
    STORAGE = {}  # see stored_tensors: the inner dialect's table, plus the at-load kind

    def __init__(self, inner: QuantConfig | None):
        inner = inner if inner is not None else NoQuantConfig()
        super().__init__(inner.name_map, ())
        self.inner = inner

    @classmethod
    def claims(cls, q: dict[str, Any]) -> bool:
        return False  # never selected by a checkpoint; --dense-quant wraps whatever was

    def scheme_for_name(self, name: str) -> QuantScheme | None:
        return self.inner.scheme_for_name(name)

    def stored_tensors(self, kind: QuantKind) -> dict[str, Stored]:
        # the loaders ask how the checkpoint stores a projection it DID quantize (a ModelOpt
        # export's NVFP4 attention under --dense-quant): that is the inner dialect's answer. The
        # at-load kind is stored as the bf16 ``weight`` the engine converts after the read -- but
        # the checkpoint may store some projections in that same kind itself (a ModelOpt export
        # keeps its attention in fp8, with an input_scale, under --dense-quant fp8), so the
        # answer for the at-load kind is the inner table with the at-load roles added, never
        # the at-load roles alone: ``storage()`` filters by each scheme's own roles.
        if kind is not self.AT_LOAD.kind:
            return self.inner.stored_tensors(kind)
        try:
            out = dict(self.inner.stored_tensors(kind))
        except (KeyError, AttributeError):
            out = {}
        for role in sorted(self.AT_LOAD.roles):
            out.setdefault(role, Stored(role))
        return out

    def layer_config(self, layer: Any, layer_kind: LayerKind, scheme: QuantScheme | None):
        return self.inner.layer_config(layer, layer_kind, scheme)

    def _quantize_at_load(self, prefix: str) -> bool:
        if self.inner.scheme_for(prefix) is not None:
            return False  # the checkpoint already quantized this one; keep its format
        names = self.name_map.to_checkpoint(prefix)
        # routed experts are read into banks by their own kernel, never converted here
        return not any(KEEP_BF16(n) or is_routed_expert(n) for n in names)

    def scheme_for(self, prefix: str) -> QuantScheme | None:
        # a model may ask what a projection's storage is before building it (the GatedDeltaNet
        # splits its fused in_proj only when the qkvz half is quantized), so answer as built
        inner = self.inner.scheme_for(prefix)
        if inner is not None:
            return inner
        return self.AT_LOAD if self._quantize_at_load(prefix) else None

    def get_quant_method(self, layer: Any, prefix: str):
        if layer.quant_layer_kind is not LayerKind.LINEAR or not self._quantize_at_load(prefix):
            return self.inner.get_quant_method(layer, prefix)
        kind = self.AT_LOAD.kind
        cls = method_class(kind, LayerKind.LINEAR)
        cfg = LinearConfig.from_layer(layer, self.AT_LOAD)
        return cls(cfg, get_quant_backend().select(LayerKind.LINEAR, kind))


class LoadTimeFp8Config(LoadTimeQuantConfig):
    """``--dense-quant fp8``: per-row fp8-e4m3, W8A16."""

    dialect = "dense-quant-fp8"
    AT_LOAD = AT_LOAD_FP8


class LoadTimeQ8Config(LoadTimeQuantConfig):
    """``--dense-quant q8_0``: ggml Q8_0 rows through the vendored int8 kernels."""

    dialect = "dense-quant-q8_0"
    AT_LOAD = AT_LOAD_Q8_0
