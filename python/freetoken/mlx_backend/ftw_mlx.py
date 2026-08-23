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
from typing import Any, Dict, List, Tuple

import numpy as np

from freetoken.utils import init_logger

from .offload import _NP_DTYPES, SafetensorsIndex, _iter_modules, _is_switch_glu

logger = init_logger(__name__)

_PAGE = 16384  # Apple silicon VM page size; also satisfies mmap allocation granularity
_PROJS = ("gate_proj", "up_proj", "down_proj")
_PARTS = ("weight", "scales", "biases")


def _cache_dir(model_dir: str) -> str:
    src = os.stat(model_dir)
    tag = f"{os.path.basename(os.path.normpath(model_dir))}-{src.st_ino}"
    root = os.environ.get(
        "FREETOKEN_MLX_FTW_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "freetoken", "mlx-ftw"),
    )
    return os.path.join(root, tag)


def _source_stamp(index: SafetensorsIndex) -> Dict[str, int]:
    shards = sorted({e.shard for e in index.entries.values()})
    total = sum(os.path.getsize(s) for s in shards)
    mtime = max(int(os.path.getmtime(s)) for s in shards)
    return {"size": total, "mtime": mtime}


def _switch_glu_tensor_names(index: SafetensorsIndex, glu_paths: List[str]):
    """Yield (store_name, [source entries making up the stacked tensor]).

    Stacked sources contribute one entry; per-expert sources contribute E entries
    that the repack concatenates (which IS the stacked layout, row-major)."""
    for path in glu_paths:
        for proj in _PROJS:
            for part in _PARTS:
                stacked = f"{path}.{proj}.{part}"
                if stacked in index:
                    yield stacked, [index[stacked]], None
                    continue
                alt = path.rsplit(".", 1)[0] + ".experts"
                per = []
                e = 0
                while f"{alt}.{e}.{proj}.{part}" in index:
                    per.append(index[f"{alt}.{e}.{proj}.{part}"])
                    e += 1
                if per:
                    yield stacked, per, len(per)


def repack_experts(model_dir: str, glu_paths: List[str]) -> str:
    """Write (or reuse) the page-aligned expert store for a checkpoint.

    Returns the cache directory containing ``experts.ftwm`` + ``manifest.json``.
    Idempotent: a manifest matching the source's size/mtime short-circuits.
    """
    index = SafetensorsIndex(model_dir)
    out_dir = _cache_dir(model_dir)
    manifest_path = os.path.join(out_dir, "manifest.json")
    store_path = os.path.join(out_dir, "experts.ftwm")
    stamp = _source_stamp(index)
    if os.path.exists(manifest_path) and os.path.exists(store_path):
        try:
            manifest = json.load(open(manifest_path))
            if manifest.get("version") == 1 and manifest.get("source") == stamp:
                return out_dir
        except Exception:  # noqa: BLE001 -- corrupt cache: rebuild below
            pass

    os.makedirs(out_dir, exist_ok=True)
    tensors: Dict[str, Any] = {}
    tmp = store_path + ".tmp"
    total = 0
    with open(tmp, "wb") as out:
        pos = 0
        for name, entries, num_experts in _switch_glu_tensor_names(index, glu_paths):
            pad = (-pos) % _PAGE
            out.write(b"\0" * pad)
            pos += pad
            first = entries[0]
            if num_experts is None:
                shape = list(first.shape)
            else:
                shape = [num_experts, *first.shape]
            tensors[name] = {"offset": pos, "shape": shape, "dtype": first.dtype}
            for e in entries:
                with open(e.shard, "rb") as src:
                    src.seek(e.start)
                    remaining = e.end - e.start
                    while remaining:
                        chunk = src.read(min(remaining, 64 << 20))
                        out.write(chunk)
                        remaining -= len(chunk)
                        pos += len(chunk)
            total += 1
    os.replace(tmp, store_path)
    with open(manifest_path, "w") as f:
        json.dump({"version": 1, "source": stamp, "tensors": tensors}, f)
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

    def glu_params(self, glu_path: str) -> Dict[str, Dict[str, Any]]:
        """{proj: {part: array}} for one switch-GLU."""
        out: Dict[str, Dict[str, Any]] = {}
        for proj in _PROJS:
            out[proj] = {
                part: self.tensors[f"{glu_path}.{proj}.{part}"] for part in _PARTS
            }
        return out


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
    for path, mod in zip(glu_paths, glu_mods, strict=True):
        params = store.glu_params(path)
        for proj, parts in params.items():
            lin = getattr(mod, proj)
            lin.weight = parts["weight"]
            lin.scales = parts["scales"]
            lin.biases = parts["biases"]
    # The store must outlive the model; hang it off the model object.
    model._freetoken_mapped_store = store
    logger.info(
        f"expert serving: zero-copy mmap store ({len(glu_paths)} MoE layers, "
        f"{os.path.getsize(store.path) / 2**30:.2f} GiB file-backed, OS-managed residency)"
    )
    return len(glu_paths)
