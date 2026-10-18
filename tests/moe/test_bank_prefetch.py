"""--moe-bank-prefetch: the CPU executor advises a task's cold rows before its workers read them.

What is worth pinning on a CPU-only box:

* the advice lands on exactly the rows the task routes to -- both passes' blocks, every row of
  them, and nothing of the rows it does not route to (reading neighbours is what the fault
  path's readahead window does wrong, and the reason this exists);
* ids the executor skips (-1 from a hybrid split, out of range) and resident rows cost nothing,
  and a row already in page cache is not advised again;
* the output is bit-identical with and without it -- it only moves when the pages arrive;
* it degrades to "do nothing" on memory that is not a file mapping, and on blocks the executor
  does not read.

The speed is not measurable here; guides/40 has the bench that chose the mechanism.
"""

from __future__ import annotations

import ctypes
import mmap
import os
import time
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

if not hasattr(os, "posix_fadvise"):
    pytest.skip("needs posix_fadvise / mincore (Linux)", allow_module_level=True)

_cpu_moe = pytest.importorskip("freetoken.kernel._cpu_moe")
if not getattr(_cpu_moe, "bank_prefetch_supported", lambda: False)():
    pytest.skip("this _cpu_moe build has no bank prefetch (stale .so or not Linux)",
                allow_module_level=True)

from freetoken.moe.bank_file import BankFile, layout_from_sample  # noqa: E402
from freetoken.moe.cpu_executor import CpuMoeExecutor  # noqa: E402
from freetoken.moe.mapped_bank import MappedBanks  # noqa: E402

E, HOT, H, I, TOPK, LAYERS = 40, 8, 512, 256, 4, 2
PAGE = mmap.PAGESIZE
_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]


def _present(addr: int, nbytes: int) -> float:
    """Share of the pages of [addr, addr+nbytes) in page cache (this process's mapping)."""
    lo = addr & ~(PAGE - 1)
    hi = (addr + nbytes + PAGE - 1) & ~(PAGE - 1)
    n = (hi - lo) // PAGE
    vec = ctypes.create_string_buffer(n)
    assert _libc.mincore(ctypes.c_void_p(lo), hi - lo, vec) == 0, ctypes.get_errno()
    return sum(b & 1 for b in vec.raw) / n


def _built(tmp_path):
    # bf16 banks big enough that both blocks are past _WHOLE_BLOCK_BYTES (4 MiB) and split:
    # gate_up rows are 512 KiB (20 MiB per layer), down rows 256 KiB (10 MiB).
    g = torch.Generator().manual_seed(0)
    src = {
        "gate_up": [(torch.randn((E, 2 * I, H), generator=g) * 0.05).to(torch.bfloat16)
                    for _ in range(LAYERS)],
        "down": [(torch.randn((E, H, I), generator=g) * 0.05).to(torch.bfloat16)
                 for _ in range(LAYERS)],
    }
    layers = list(range(LAYERS))
    lay = layout_from_sample({n: src[n][0] for n in src}, layers, E)
    path = str(tmp_path / "bank.ftmb")
    with BankFile.create(path, lay) as w:
        for layer in layers:
            w.write_layer(layer, {n: src[n][layer] for n in src}, list(range(E)))
    return src, path


def _evict(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def _executor(sources):
    cache = SimpleNamespace(num_layers=len(next(iter(sources.values()))), num_experts=E, quant_format="bf16",
                            bank_sources=sources)
    threads = torch.get_num_threads()
    try:
        return CpuMoeExecutor(cache, top_k=TOPK, activation="silu",
                              apply_router_weight_on_input=False, num_threads=2, max_tokens=4,
                              device=torch.device("cpu"), layer_formats=[("bf16", {})] * cache.num_layers)
    finally:
        torch.set_num_threads(threads)  # the executor clamps torch's pool; keep the test's


class _Task:
    """One executor task: its rows written into the executor's IO block (plain host tensors on a
    CPU-device executor), named by (layer, rows) as the doorbell names it."""

    def __init__(self, ex, layer, ids):
        ids = torch.as_tensor(ids, dtype=torch.int32).reshape(-1, TOPK)
        n = ids.shape[0]
        g = torch.Generator().manual_seed(1)
        self.ex, self.layer, self.n = ex, layer, n
        ex._io["x"][:n] = torch.randn((n, H), generator=g).to(torch.bfloat16)
        ex._io["ids"][:n] = ids
        ex._io["w"][:n] = 1.0 / TOPK

    def prefetch(self) -> None:
        self.ex._ext.prefetch_only(self.layer, self.n)

    def run(self) -> torch.Tensor:
        self.ex._ext.run(self.layer, self.n)
        return self.ex._io["y"][: self.n].clone()


def _wait_present(addr, nbytes, timeout=10.0):
    end = time.monotonic() + timeout
    while _present(addr, nbytes) < 1.0 and time.monotonic() < end:
        time.sleep(0.01)
    return _present(addr, nbytes)


def _rows(banks, name, layer):
    for bname, pos, addr, row_bytes, cold_from in banks.cold_blocks:
        if bname == name and pos == layer:
            return addr, row_bytes, cold_from
    raise KeyError((name, layer))


def test_cold_blocks_describe_the_file_backed_rows(tmp_path):
    _, path = _built(tmp_path)
    banks = MappedBanks(path, register=False, hot_per_layer=HOT)
    try:
        assert len(banks.cold_blocks) == 2 * LAYERS
        for name, pos, addr, row_bytes, cold_from in banks.cold_blocks:
            view = banks.sources[name][pos]
            assert addr == view.data_ptr()
            assert row_bytes == view[0].numel() * view.element_size()
            assert cold_from == HOT
    finally:
        banks.close()


def test_exactly_the_routed_cold_rows_are_advised(tmp_path):
    _, path = _built(tmp_path)
    banks = MappedBanks(path, register=False, hot_per_layer=HOT)
    try:
        ex = _executor(banks.sources)
        assert ex.enable_bank_prefetch(banks.cold_blocks) == 2 * LAYERS
        _evict(path)
        layer = 1
        routed = [20, 31]
        unrouted = [21, 39]
        blocks = [_rows(banks, n, layer) for n in ("gate_up", "down")]
        for addr, rb, _ in blocks:
            for e in routed + unrouted:
                if _present(addr + e * rb, rb) != 0.0:
                    pytest.skip("the page cache would not drop the test file")
        # two tokens: duplicates, a hybrid -1, a resident row and an out-of-range id
        t = _Task(ex, layer, [20, -1, 3, 31,
                              31, 20, E + 5, 2])
        t.prefetch()
        for addr, rb, _ in blocks:
            for e in routed:
                assert _wait_present(addr + e * rb, rb) == 1.0, (addr, e)
            for e in unrouted:
                assert _present(addr + e * rb, rb) == 0.0, (addr, e)
        s = ex.bank_prefetch_stats()
        assert s["tasks"] == 1 and s["rows"] == len(routed)
        assert s["ranges_advised"] == len(routed) * len(blocks)
        assert s["errno"] == 0
        # already in page cache: listed again, advised no more
        t.prefetch()
        s2 = ex.bank_prefetch_stats()
        assert s2["tasks"] == 2 and s2["ranges_advised"] == s["ranges_advised"]
    finally:
        banks.close()


def test_output_is_identical_with_and_without_prefetch(tmp_path):
    src, path = _built(tmp_path)
    banks = MappedBanks(path, register=False, hot_per_layer=HOT)
    try:
        mapped = _executor(banks.sources)
        assert mapped.enable_bank_prefetch(banks.cold_blocks) == 2 * LAYERS
        in_ram = _executor({n: [t.clone() for t in banks.sources[n]] for n in banks.sources})
        _evict(path)
        ids = [9, 33, 1, 17,
               33, -1, 25, 4,
               39, 8, 0, 12]
        for layer in range(LAYERS):
            a = _Task(mapped, layer, ids)
            b = _Task(in_ram, layer, ids)
            ya, yb = a.run(), b.run()
            assert torch.equal(ya, yb)
            assert ya.abs().sum() > 0
        # {8, 9, 12, 17, 25, 33, 39} per layer: rows at or past HOT, once each
        assert mapped.bank_prefetch_stats()["rows"] == 2 * 7
    finally:
        banks.close()


def test_nothing_is_advised_for_blocks_the_executor_does_not_read(tmp_path):
    _, path = _built(tmp_path)
    banks = MappedBanks(path, register=False, hot_per_layer=HOT)
    try:
        copy = _executor({n: [t.clone() for t in banks.sources[n]] for n in banks.sources})
        assert copy.enable_bank_prefetch(banks.cold_blocks) == 0
        t = _Task(copy, 0, [20, 21, 22, 23])
        t.prefetch()
        assert copy.bank_prefetch_stats()["tasks"] == 0
    finally:
        banks.close()


def test_resident_blocks_are_dropped_and_an_empty_list_turns_it_off(tmp_path):
    _, path = _built(tmp_path)
    banks = MappedBanks(path, register=False, hot_per_layer=HOT)
    try:
        ex = _executor(banks.sources)
        whole = [(n, p, a, rb, E) for n, p, a, rb, _ in banks.cold_blocks]
        assert ex.enable_bank_prefetch(whole) == 0
        assert ex.enable_bank_prefetch(banks.cold_blocks) == 2 * LAYERS
        assert ex.enable_bank_prefetch([]) == 0
        t = _Task(ex, 0, [20, 21, 22, 23])
        t.prefetch()
        assert ex.bank_prefetch_stats()["tasks"] == 0
    finally:
        banks.close()


def test_the_private_form_is_advised_too(tmp_path, monkeypatch):
    """FREETOKEN_BANK_MAP=private: resident rows are anonymous copies, the cold rows are still
    file-backed pages of a MAP_PRIVATE mapping, and MADV_WILLNEED reads them the same way."""
    import freetoken.moe.mapped_bank as mb

    class _Cudart:
        def cudaHostRegister(self, addr, nbytes, flags):
            return 0 if flags == 0 else 1

        def cudaHostUnregister(self, addr):
            return 0

    monkeypatch.setattr(torch.cuda, "cudart", lambda: _Cudart())
    monkeypatch.setattr(mb, "clear_cuda_error", lambda: None)
    monkeypatch.setenv("FREETOKEN_BANK_MAP", "private")
    _, path = _built(tmp_path)
    banks = MappedBanks(path, hot_per_layer=HOT)
    try:
        assert banks.private
        ex = _executor(banks.sources)
        assert ex.enable_bank_prefetch(banks.cold_blocks) == 2 * LAYERS
        _evict(path)
        addr, rb, _ = _rows(banks, "gate_up", 0)
        if _present(addr + 30 * rb, rb) != 0.0:
            pytest.skip("the page cache would not drop the test file")
        t = _Task(ex, 0, [30, -1, -1, -1])
        t.prefetch()
        assert _wait_present(addr + 30 * rb, rb) == 1.0
        assert ex.bank_prefetch_stats()["errno"] == 0
    finally:
        banks.close()


def test_a_failing_advice_stops_the_asking_for_that_block():
    """A range madvise refuses (here: no longer mapped, ENOMEM) is recorded once and the block
    is dropped, instead of failing a syscall per row per step."""
    row = 4 * PAGE
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                          ctypes.c_int, ctypes.c_long]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    base = libc.mmap(None, E * row, mmap.PROT_READ, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, -1, 0)
    assert base not in (None, ctypes.c_void_p(-1).value)
    assert libc.munmap(base, E * row) == 0  # the address range is a hole now
    gate_up = [torch.zeros((E, 2 * I, H), dtype=torch.bfloat16) for _ in range(LAYERS)]
    down = [torch.zeros((E, H, I), dtype=torch.bfloat16) for _ in range(LAYERS)]
    ex = _executor({"gate_up": gate_up, "down": down})
    assert ex._ext.set_bank_prefetch([0], [base], [row], [HOT], [1]) == 1
    t = _Task(ex, 0, [20, 21, 22, 23])
    t.prefetch()
    s = ex.bank_prefetch_stats()
    assert s["errno"] != 0 and s["ranges_advised"] == 0
    t.prefetch()  # the block is off now: rows listed, nothing advised
    s2 = ex.bank_prefetch_stats()
    assert s2["tasks"] == 2 and s2["ranges_advised"] == 0


def test_a_later_ranks_blocks_are_numbered_as_its_executor_numbers_them(tmp_path):
    """One file for every layer, and a rank maps only its own: the block's layer id is its position
    in the rank's layers (what the cache and the executor count), and its address is inside the
    rank's mapping. Numbered by the file's layers, rank 1's first layer would be 1, which its
    executor -- one layer, id 0 -- does not read, and the flag would silently do nothing."""
    _, path = _built(tmp_path)
    banks = MappedBanks(path, register=False, layers=[1], hot_per_layer=HOT)
    try:
        assert sorted(pos for _, pos, _, _, _ in banks.cold_blocks) == [0, 0]
        for name, pos, addr, _, _ in banks.cold_blocks:
            assert addr == banks.sources[name][pos].data_ptr()
        ex = _executor(banks.sources)
        assert ex.enable_bank_prefetch(banks.cold_blocks) == 2
        _evict(path)
        addr, rb, _ = _rows(banks, "gate_up", 0)
        if _present(addr + 30 * rb, rb) != 0.0:
            pytest.skip("the page cache would not drop the test file")
        t = _Task(ex, 0, [30, -1, -1, -1])
        t.prefetch()
        assert _wait_present(addr + 30 * rb, rb) == 1.0
    finally:
        banks.close()
