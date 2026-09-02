"""FTW-MLX: zero-copy mmapped expert store for the MLX backend.

The CUDA engine's FTW format repacks checkpoints so expert banks can be served
without per-request copies. This is its Apple-silicon analogue: the switch-GLU
expert tensors are repacked ONCE into a page-aligned flat file, and at serve
time each stacked tensor is memory-mapped and imported into MLX **zero-copy**
(``mx.from_dlpack`` on a ``np.memmap`` — unified memory makes the same pages
GPU-readable, the same trick llama.cpp's Metal backend uses for its weights).

The payoff, compared to the slot-cache offload path: expert serving becomes a
plain ``gather_qmm`` over the mapped store — no slot cache, no per-layer CPU
syncs, no speculation, no misses. Residency is managed by the OS page cache:
hot experts stay in RAM, cold ones fault in from SSD at first touch, and under
memory pressure clean file-backed pages are simply evicted (never swapped).
``mx.get_active_memory`` therefore only shows the dense weights; the experts'
real cost is reclaimable page cache.

Why a repack is needed at all: safetensors gives no alignment guarantees (the
JSON header makes every offset arbitrary), while Metal needs dtype-aligned
buffer offsets and ``np.memmap`` needs page-aligned mapping offsets. The repack
also stacks per-expert layouts (OLMoE-era conversions) into ``[E, ...]`` form.

Layout: ``experts.ftwm`` (page-aligned raw tensors) + ``manifest.json``:
{"version": 1, "source": {size, mtime}, "tensors": {name: {offset, shape, dtype}}}
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, List, Tuple

import numpy as np

from maxtoken.utils import init_logger

from .offload import _NP_DTYPES, SafetensorsIndex, _iter_modules, _is_switch_glu

logger = init_logger(__name__)

_PAGE = 16384  # Apple silicon VM page size; also satisfies mmap allocation granularity
_PROJS = ("gate_proj", "up_proj", "down_proj")
_PARTS = ("weight", "scales", "biases")


def _cache_dir(model_dir: str) -> str:
    src = os.stat(model_dir)
    tag = f"{os.path.basename(os.path.normpath(model_dir))}-{src.st_ino}"
    root = os.environ.get(
        "MAXTOKEN_MLX_FTW_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "maxtoken", "mlx-ftw"),
    )
    return os.path.join(root, tag)


def _source_stamp(index: SafetensorsIndex) -> Dict[str, int]:
    shards = sorted({e.shard for e in index.entries.values()})
    total = sum(os.path.getsize(s) for s in shards)
    mtime = max(int(os.path.getmtime(s)) for s in shards)
    return {"size": total, "mtime": mtime}


def _pack_enabled() -> bool:
    """Whether the store lays gate and up out interleaved per expert
    (``gate_up_proj``), so the two projections run as ONE gather (moe_pack.py).
    On by default; MAXTOKEN_MLX_FTW_PACK=0 keeps the classic per-proj layout."""
    return os.environ.get("MAXTOKEN_MLX_FTW_PACK", "1") != "0"


def _expert_slice(src, e: int, E: int, stacked: bool) -> Tuple[str, int, int]:
    """Byte range of expert ``e`` inside one source tensor."""
    if stacked:
        per = (src.end - src.start) // E
        return (src.shard, src.start + e * per, src.start + (e + 1) * per)
    s = src[e]
    return (s.shard, s.start, s.end)


def _switch_glu_tensor_names(index: SafetensorsIndex, glu_paths: List[str]):
    """Yield (store_name, byte_slices, shape, dtype) for every store tensor.

    Byte slices are (shard, start, end) ranges written in order. Stacked
    sources contribute one slice, per-expert sources one per expert (their
    concatenation IS the stacked layout, row-major). With packing enabled,
    each GLU's gate and up tensors are interleaved PER EXPERT into a single
    ``gate_up_proj`` tensor — affine groups run along the input axis, so the
    concatenation along output features is exact by construction."""

    def sources(path: str, proj: str, part: str):
        stacked = f"{path}.{proj}.{part}"
        if stacked in index:
            e = index[stacked]
            return e, int(e.shape[0]), True
        alt = path.rsplit(".", 1)[0] + ".experts"
        per = []
        i = 0
        while f"{alt}.{i}.{proj}.{part}" in index:
            per.append(index[f"{alt}.{i}.{proj}.{part}"])
            i += 1
        if per:
            return per, len(per), False
        return None, 0, False

    for path in glu_paths:
        pack = _pack_enabled()
        if pack:
            for part in _PARTS:
                g, Eg, _ = sources(path, "gate_proj", part)
                u, Eu, _ = sources(path, "up_proj", part)
                if g is None or u is None or Eg != Eu or Eg == 0:
                    pack = False
                    break
        if pack:
            for part in _PARTS:
                g, E, g_stacked = sources(path, "gate_proj", part)
                u, _, u_stacked = sources(path, "up_proj", part)
                slices: List[Tuple[str, int, int]] = []
                for e in range(E):
                    slices.append(_expert_slice(g, e, E, g_stacked))
                    slices.append(_expert_slice(u, e, E, u_stacked))
                first = g if g_stacked else g[0]
                if g_stacked:
                    shape = [E, 2 * int(first.shape[1]), *first.shape[2:]]
                else:
                    shape = [E, 2 * int(first.shape[0]), *first.shape[1:]]
                yield f"{path}.gate_up_proj.{part}", slices, shape, first.dtype
        for proj in (("down_proj",) if pack else _PROJS):
            for part in _PARTS:
                src, E, stacked = sources(path, proj, part)
                if src is None:
                    continue
                if stacked:
                    yield (
                        f"{path}.{proj}.{part}",
                        [(src.shard, src.start, src.end)],
                        list(src.shape),
                        src.dtype,
                    )
                else:
                    yield (
                        f"{path}.{proj}.{part}",
                        [(s.shard, s.start, s.end) for s in src],
                        [E, *src[0].shape],
                        src[0].dtype,
                    )


def repack_experts(model_dir: str, glu_paths: List[str]) -> str:
    """Write (or reuse) the page-aligned expert store for a checkpoint.

    Returns the cache directory containing ``experts.ftwm`` + ``manifest.json``.
    Idempotent: a manifest matching the source's size/mtime short-circuits.
    """
    index = SafetensorsIndex(model_dir)
    out_dir = _cache_dir(model_dir)
    manifest_path = os.path.join(out_dir, "manifest.json")
    store_path = os.path.join(out_dir, "experts.ftwm")
    stamp = {**_source_stamp(index), "packed": _pack_enabled()}
    if os.path.exists(manifest_path) and os.path.exists(store_path):
        try:
            manifest = json.load(open(manifest_path))
            if manifest.get("version") == 2 and manifest.get("source") == stamp:
                return out_dir
        except Exception:  # noqa: BLE001 -- corrupt cache: rebuild below
            pass

    os.makedirs(out_dir, exist_ok=True)
    tensors: Dict[str, Any] = {}
    tmp = store_path + ".tmp"
    total = 0
    handles: Dict[str, Any] = {}
    try:
        with open(tmp, "wb") as out:
            pos = 0
            for name, slices, shape, dtype in _switch_glu_tensor_names(
                index, glu_paths
            ):
                pad = (-pos) % _PAGE
                out.write(b"\0" * pad)
                pos += pad
                tensors[name] = {"offset": pos, "shape": shape, "dtype": dtype}
                for shard, start, end in slices:
                    src = handles.get(shard)
                    if src is None:
                        src = handles[shard] = open(shard, "rb")
                    src.seek(start)
                    remaining = end - start
                    while remaining:
                        chunk = src.read(min(remaining, 64 << 20))
                        out.write(chunk)
                        remaining -= len(chunk)
                        pos += len(chunk)
                total += 1
    finally:
        for fh in handles.values():
            fh.close()
    os.replace(tmp, store_path)
    with open(manifest_path, "w") as f:
        json.dump({"version": 2, "source": stamp, "tensors": tensors}, f)
    logger.info(
        f"FTW-MLX expert store: repacked {total} tensors "
        f"({os.path.getsize(store_path) / 2**30:.2f} GiB) into {out_dir}"
    )
    return out_dir


class MappedExpertStore:
    """Zero-copy mx views over the repacked expert store, one per tensor name.

    Holds the numpy memmap bases alive for the lifetime of the store — the mx
    arrays alias their pages (DLPack import), so dropping the bases would leave
    dangling device pointers.
    """

    def __init__(self, store_dir: str):
        import mlx.core as mx

        self._mx = mx
        self._bases: List[Any] = []
        manifest = json.load(open(os.path.join(store_dir, "manifest.json")))
        self.path = os.path.join(store_dir, "experts.ftwm")
        self.tensors: Dict[str, Any] = {}
        for name, meta in manifest["tensors"].items():
            np_dt = _NP_DTYPES[meta["dtype"]]
            base = np.memmap(
                self.path,
                dtype=np_dt,
                mode="c",  # copy-on-write: file-backed clean pages, OS-evictable
                offset=meta["offset"],
                shape=tuple(meta["shape"]),
            )
            self._bases.append(base)
            arr = mx.from_dlpack(base)
            if meta["dtype"] == "BF16":
                arr = arr.view(mx.bfloat16)
            self.tensors[name] = arr
        self._prefetch_lock = threading.Lock()
        self._prefetching = False
        self._advise_map: Any = None

    # -------------------------------------------------------------- prefetch

    def _advise_handle(self):
        """A parallel plain mmap of the store used only for madvise/page-in;
        it shares the page cache with the DLPack-imported memmaps."""
        import mmap as mmap_mod

        if self._advise_map is None:
            f = open(self.path, "rb")
            self._advise_map = mmap_mod.mmap(f.fileno(), 0, prot=mmap_mod.PROT_READ)
            self._advise_file = f  # keep the fd alive with the map
        return self._advise_map

    def start_prefetch(self) -> None:
        """Kick a background sweep that advises the store's pages into memory
        IN FILE (= layer) ORDER, so page-in runs ahead of the forward that
        consumes the layers in the same order. This is the unified-memory
        analogue of streaming expert uploads on a second stream: on a cold
        cache the prefill otherwise pays every page fault synchronously inside
        its evals. Idempotent while a sweep is running; near-free when warm."""
        import mmap as mmap_mod

        with self._prefetch_lock:
            if self._prefetching:
                return
            self._prefetching = True

        def sweep():
            import time as _time

            try:
                m = self._advise_handle()
                size = len(m)
                # Small chunks with an explicit yield: ``mmap.madvise`` does
                # NOT release the GIL, so a background sweep in 256 MB chunks
                # strangled the scheduler thread in 96 ms bites — measured as
                # 1.4 s prompt rounds and 0.7 s gaps during a concurrent
                # bench (the "B>=2 server tax" in large part). 32 MB keeps
                # each GIL hold near 10 ms and the sleep lets the scheduler
                # run between chunks; the sweep still finishes a 17 GiB
                # store in a few seconds.
                chunk = 32 << 20
                for off in range(0, size, chunk):
                    m.madvise(
                        mmap_mod.MADV_WILLNEED, off, min(chunk, size - off)
                    )
                    _time.sleep(0.002)
            except Exception as exc:  # noqa: BLE001 -- advisory only
                logger.warning(f"expert-store prefetch sweep failed: {exc!r}")
            finally:
                with self._prefetch_lock:
                    self._prefetching = False

        threading.Thread(target=sweep, name="ftw-mlx-prefetch", daemon=True).start()

    def mlock_all(self) -> bool:
        """Pin the whole store into memory (MAXTOKEN_MLX_MLOCK=1): the pressure
        counterpart of llama.cpp's host-register trick. Only sensible when the
        store fits in RAM with headroom — pinned pages cannot be reclaimed, so
        this trades system elasticity for immunity against expert-page eviction."""
        import ctypes

        try:
            libc = ctypes.CDLL(None, use_errno=True)
            total = 0
            for base in self._bases:
                addr = base.ctypes.data
                length = base.nbytes
                if libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(length)) != 0:
                    err = ctypes.get_errno()
                    logger.warning(
                        f"mlock failed after {total / 2**30:.2f} GiB (errno {err}); "
                        "raise the memlock rlimit or drop MAXTOKEN_MLX_MLOCK"
                    )
                    return False
                total += length
            logger.info(f"expert store pinned: {total / 2**30:.2f} GiB mlocked")
            return True
        except Exception as exc:  # noqa: BLE001 -- opt-in nicety
            logger.warning(f"mlock unavailable: {exc!r}")
            return False

    def glu_params(self, glu_path: str) -> Dict[str, Dict[str, Any]]:
        """{proj: {part: array}} for one switch-GLU; ``gate_up_proj`` when the
        store holds the packed interleaved layout."""
        out: Dict[str, Dict[str, Any]] = {}
        for proj in (*_PROJS, "gate_up_proj"):
            parts = {
                part: self.tensors[f"{glu_path}.{proj}.{part}"]
                for part in _PARTS
                if f"{glu_path}.{proj}.{part}" in self.tensors
            }
            if parts:
                out[proj] = parts
        return out


def _resolve_module(model, path: str):
    """Walk a dotted parameter path (list indices included) to its module."""
    obj = model
    for name in path.split("."):
        obj = obj[int(name)] if name.isdigit() else getattr(obj, name)
    return obj


def attach_mapped_experts(model, model_dir: str) -> int:
    """Serve every switch-GLU's experts from the zero-copy mapped store.

    Must run on a ``lazy=True``-loaded model. Unlike the slot-cache offload this
    keeps mlx-lm's own SwitchGLU modules and forward (including its sorted-gather
    prefill path) — only their weight/scales/biases parameters are replaced by
    the mapped views. Returns the number of GLUs rewired.
    """
    index = SafetensorsIndex(model_dir)
    glu_paths = []
    glu_mods = []
    for path, mod in _iter_modules(model):
        if not (path and _is_switch_glu(mod)):
            continue
        if f"{path}.{_PROJS[0]}.{_PARTS[0]}" in index or (
            f"{path.rsplit('.', 1)[0]}.experts.0.{_PROJS[0]}.{_PARTS[0]}" in index
        ):
            glu_paths.append(path)
            glu_mods.append(mod)
    if not glu_paths:
        raise ValueError("no switch-GLU expert tensors found for the mapped store")

    store_dir = repack_experts(model_dir, glu_paths)
    store = MappedExpertStore(store_dir)
    packed_count = 0
    for path, mod in zip(glu_paths, glu_mods, strict=True):
        params = store.glu_params(path)
        packed = params.get("gate_up_proj")
        if packed is not None:
            # Replace the whole SwitchGLU with the packed form BEFORE the
            # model evaluates its parameters: the dropped gate/up modules
            # still hold their lazy checkpoint arrays, and keeping them
            # reachable would materialize ~10 GB the mapped store exists to
            # avoid.
            from .moe_pack import classes

            PackedQuantizedProjection, PackedSwitchGLU, _ = classes()
            down = mod.down_proj
            dp = params["down_proj"]
            down.weight = dp["weight"]
            down.scales = dp["scales"]
            down.biases = dp["biases"]
            proj = PackedQuantizedProjection(
                packed["weight"],
                packed["scales"],
                packed.get("biases"),
                group_size=int(getattr(mod.gate_proj, "group_size", 64)),
                bits=int(getattr(mod.gate_proj, "bits", 4)),
                mode=str(getattr(mod.gate_proj, "mode", "affine")),
            )
            split_at = int(packed["weight"].shape[1]) // 2
            parent_path, attr = path.rsplit(".", 1)
            setattr(
                _resolve_module(model, parent_path),
                attr,
                PackedSwitchGLU(proj, down, mod.activation, split_at),
            )
            packed_count += 1
            continue
        for proj, parts in params.items():
            lin = getattr(mod, proj)
            lin.weight = parts["weight"]
            lin.scales = parts["scales"]
            lin.biases = parts["biases"]
    if packed_count:
        logger.info(
            f"MoE pack: {packed_count} switch-GLUs serve gate+up as one gather "
            "(interleaved mapped store)"
        )
    # The store must outlive the model; hang it off the model object.
    model._maxtoken_mapped_store = store
    logger.info(
        f"expert serving: zero-copy mmap store ({len(glu_paths)} MoE layers, "
        f"{os.path.getsize(store.path) / 2**30:.2f} GiB file-backed, OS-managed residency)"
    )
    if os.environ.get("MAXTOKEN_MLX_MLOCK") == "1":
        store.mlock_all()
    return len(glu_paths)
