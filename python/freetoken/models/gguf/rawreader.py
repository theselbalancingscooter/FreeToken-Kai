"""A small GGUF reader that does not go through gguf-py.

``gguf.GGUFReader`` builds a ``GGMLQuantizationType`` for every tensor and raises on a type
it does not list; gguf-py 0.19.0 (PyPI, 2026-05) predates ggml's Q1_0 (41) and Q2_0 (42), so a
checkpoint that carries them cannot even be opened with it. This module parses the GGUF v2/v3
container itself -- header, KV metadata, tensor infos -- and hands out zero-copy ``[rows,
row_bytes]`` uint8 views of the packed tensor bytes from one read-only mmap of the file. Only the
types in ``dequant.BLOCK_SHAPE`` can be *viewed* (the row width needs the block geometry); every
tensor is still *listed*, whatever its type.

The split-file convention (``-00001-of-00002.gguf``) is not followed here: the caller names the
one shard it wants.
"""

from __future__ import annotations

import mmap
import os
import struct
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch

from .dequant import BLOCK_SHAPE, GGML_NAME

GGUF_MAGIC = b"GGUF"
_DEFAULT_ALIGNMENT = 32

# GGUF value types (gguf.h gguf_type)
_T_UINT8, _T_INT8, _T_UINT16, _T_INT16, _T_UINT32, _T_INT32, _T_FLOAT32, _T_BOOL = range(8)
_T_STRING, _T_ARRAY, _T_UINT64, _T_INT64, _T_FLOAT64 = range(8, 13)
_SCALAR_FMT = {
    _T_UINT8: "<B", _T_INT8: "<b", _T_UINT16: "<H", _T_INT16: "<h", _T_UINT32: "<I",
    _T_INT32: "<i", _T_FLOAT32: "<f", _T_BOOL: "<?", _T_UINT64: "<Q", _T_INT64: "<q",
    _T_FLOAT64: "<d",
}


@dataclass(frozen=True)
class RawGgufTensor:
    name: str
    ne: tuple[int, ...]  # ggml order, fastest dim first
    ggml_type: int
    offset: int  # bytes from the start of the data section

    @property
    def shape(self) -> tuple[int, ...]:
        """torch order (ggml dims reversed)."""
        return tuple(reversed(self.ne))

    @property
    def type_name(self) -> str:
        return GGML_NAME.get(self.ggml_type, f"type{self.ggml_type}")


class _Cursor:
    def __init__(self, buf: memoryview):
        self.buf = buf
        self.pos = 0

    def take(self, fmt: str):
        size = struct.calcsize(fmt)
        value = struct.unpack_from(fmt, self.buf, self.pos)[0]
        self.pos += size
        return value

    def string(self) -> str:
        n = self.take("<Q")
        raw = bytes(self.buf[self.pos:self.pos + n])
        self.pos += n
        return raw.decode("utf-8", errors="replace")

    def value(self, vtype: int):
        if vtype == _T_STRING:
            return self.string()
        if vtype == _T_ARRAY:
            etype = self.take("<I")
            n = self.take("<Q")
            if etype in _SCALAR_FMT and n:
                fmt = _SCALAR_FMT[etype]
                size = struct.calcsize(fmt)
                items = list(struct.unpack_from(f"<{n}{fmt[1]}", self.buf, self.pos))
                self.pos += size * n
                return items
            return [self.value(etype) for _ in range(n)]
        fmt = _SCALAR_FMT.get(vtype)
        if fmt is None:
            raise ValueError(f"GGUF: unknown KV value type {vtype} at byte {self.pos}")
        return self.take(fmt)


class RawGguf:
    """One GGUF file: ``metadata`` (KV dict), ``tensors`` (name -> RawGgufTensor, file order) and
    zero-copy views of the packed bytes. Keeps the file mapped for the object's lifetime."""

    def __init__(self, path: str):
        self.path = os.fspath(path)
        self._fd = os.open(self.path, os.O_RDONLY)
        try:
            self._mm = mmap.mmap(self._fd, 0, access=mmap.ACCESS_READ)
        except Exception:
            os.close(self._fd)
            raise
        cur = _Cursor(memoryview(self._mm))
        if bytes(cur.buf[:4]) != GGUF_MAGIC:
            raise ValueError(f"{self.path}: not a GGUF file")
        cur.pos = 4
        self.version = cur.take("<I")
        if self.version not in (2, 3):
            raise ValueError(f"{self.path}: GGUF version {self.version} is not supported (2 or 3)")
        n_tensors = cur.take("<Q")
        n_kv = cur.take("<Q")
        self.metadata: dict[str, Any] = {}
        for _ in range(n_kv):
            key = cur.string()
            vtype = cur.take("<I")
            self.metadata[key] = cur.value(vtype)
        self.tensors: dict[str, RawGgufTensor] = {}
        for _ in range(n_tensors):
            name = cur.string()
            n_dims = cur.take("<I")
            ne = tuple(cur.take("<Q") for _ in range(n_dims))
            ggml_type = cur.take("<I")
            offset = cur.take("<Q")
            self.tensors[name] = RawGgufTensor(name, ne, ggml_type, offset)
        align = int(self.metadata.get("general.alignment", _DEFAULT_ALIGNMENT))
        self.data_start = (cur.pos + align - 1) // align * align
        self.size = len(self._mm)

    def close(self) -> None:
        mm, self._mm = getattr(self, "_mm", None), None
        if mm is not None:
            mm.close()
            os.close(self._fd)

    def __del__(self):  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass

    @property
    def architecture(self) -> str | None:
        arch = self.metadata.get("general.architecture")
        return str(arch) if arch is not None else None

    def __iter__(self) -> Iterator[RawGgufTensor]:
        return iter(self.tensors.values())

    def row_layout(self, t: RawGgufTensor) -> tuple[int, int]:
        """``(rows, row_bytes)`` of ``t``: rows = product of all but the fastest ggml dim."""
        if t.ggml_type not in BLOCK_SHAPE:
            raise NotImplementedError(
                f"{t.name}: ggml type {t.type_name} has no block geometry here (BLOCK_SHAPE)"
            )
        block, type_size = BLOCK_SHAPE[t.ggml_type]
        n_fast = t.ne[0] if t.ne else 1
        if n_fast % block != 0:
            raise ValueError(f"{t.name}: fastest dim {n_fast} not a multiple of block {block} for {t.type_name}")
        rows = int(np.prod(t.ne[1:])) if len(t.ne) > 1 else 1
        return rows, n_fast // block * type_size

    def view(self, name: str) -> np.ndarray:
        """Zero-copy ``[rows, row_bytes]`` uint8 view of ``name``'s packed bytes."""
        t = self.tensors[name]
        rows, row_bytes = self.row_layout(t)
        start = self.data_start + t.offset
        end = start + rows * row_bytes
        if end > self.size:
            raise ValueError(f"{t.name}: data [{start}, {end}) runs past the end of {self.path} ({self.size} bytes)")
        return np.frombuffer(self._mm, dtype=np.uint8, count=rows * row_bytes, offset=start).reshape(rows, row_bytes)

    def packed(self, name: str) -> torch.Tensor:
        """``view`` as a torch uint8 tensor (shares the mmap; read-only)."""
        return torch.from_numpy(self.view(name))


def write_gguf(path: str, tensors: list[tuple[str, tuple[int, ...], int, bytes]],
               metadata: dict[str, Any] | None = None, alignment: int = _DEFAULT_ALIGNMENT) -> None:
    """Write a minimal GGUF v3 file: ``tensors`` are ``(name, ne, ggml_type, packed_bytes)`` with
    ``ne`` in ggml order. Metadata values may be str, int, float, bool or lists of those (ints
    become uint32 / int64 as they fit). For tests and tools; not a general-purpose writer."""
    metadata = dict(metadata or {})
    metadata.setdefault("general.alignment", alignment)
    out = bytearray()
    out += GGUF_MAGIC + struct.pack("<I", 3) + struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(metadata))

    def put_str(s: str) -> None:
        b = s.encode("utf-8")
        out.extend(struct.pack("<Q", len(b)) + b)

    def scalar_type(v) -> int:
        if isinstance(v, bool):
            return _T_BOOL
        if isinstance(v, int):
            return _T_UINT32 if 0 <= v < 2**32 else _T_INT64
        if isinstance(v, float):
            return _T_FLOAT32
        if isinstance(v, str):
            return _T_STRING
        raise TypeError(f"unsupported GGUF metadata value {v!r}")

    def put_value(vtype: int, v) -> None:
        if vtype == _T_STRING:
            put_str(v)
        else:
            out.extend(struct.pack(_SCALAR_FMT[vtype], v))

    for key, value in metadata.items():
        put_str(key)
        if isinstance(value, (list, tuple)):
            etype = scalar_type(value[0]) if value else _T_UINT32
            out.extend(struct.pack("<I", _T_ARRAY) + struct.pack("<I", etype) + struct.pack("<Q", len(value)))
            for item in value:
                put_value(etype, item)
        else:
            vtype = scalar_type(value)
            out.extend(struct.pack("<I", vtype))
            put_value(vtype, value)
    offset = 0
    offsets = []
    for name, ne, ggml_type, data in tensors:
        put_str(name)
        out.extend(struct.pack("<I", len(ne)))
        for d in ne:
            out.extend(struct.pack("<Q", d))
        out.extend(struct.pack("<I", ggml_type) + struct.pack("<Q", offset))
        offsets.append(offset)
        offset += (len(data) + alignment - 1) // alignment * alignment
    pad = (-len(out)) % alignment
    out.extend(b"\0" * pad)
    for (_name, _ne, _t, data), off in zip(tensors, offsets):
        assert len(out) - (len(out) - len(out)) >= 0
        out.extend(data)
        out.extend(b"\0" * ((-len(data)) % alignment))
    with open(path, "wb") as f:
        f.write(out)


__all__ = ["RawGguf", "RawGgufTensor", "write_gguf"]
