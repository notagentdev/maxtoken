"""MoE expert offload for the MLX backend — FreeToken's core idea on Apple silicon.

The CUDA engine serves models whose experts do not fit in fast memory by keeping a
small GPU slot cache of hot experts (LRU) and streaming misses from host banks.
This module is the direct translation of that design to MLX:

- ``SafetensorsIndex`` / ``ExpertStore``: every (switch-GLU, expert) is a contiguous
  byte range inside a safetensors shard (the converted repos store experts stacked
  ``[E, ...]``, row-major). Experts are fetched with ``os.pread`` — no stacked tensor
  is ever materialized, so the resident footprint is the dense weights plus the cache.
- ``SlotCache``: per MoE layer a preallocated buffer of ``S << E`` expert slots
  (weight/scales/biases for gate/up/down). Decode runs ``mx.gather_qmm`` over the
  slot buffers with remapped indices — the same kernel stock mlx-lm uses over all
  ``E`` experts, so cache hits cost what a fully resident model costs. Misses are
  pread + one scatter write, evicting LRU slots. This mirrors the CUDA engine's
  ``slot_cache`` exactly (validated bit-identical against the stacked weights).
- Streamed prefill: a chunk that routes to more experts than the cache holds is
  served from a freshly materialized full layer (lazy ``mx.load`` -> eval -> drop),
  double-buffered by prefetching the next layer with ``mx.async_eval`` — FreeToken's
  full-layer prefill streaming. The slot cache is not polluted by prefill traffic.
- ``resize``: the slot count can be changed at runtime (elastic memory management);
  the most recently used experts survive a shrink/grow, everything else is evicted.

Attach with ``attach_expert_offload(model, model_dir, slots_per_layer)`` after an
``mlx_lm.load(..., lazy=True)``: the stacked expert arrays are dropped **unevaluated**
(they never touch memory) and every SwitchGLU is replaced by an ``OffloadSwitchGLU``.
Everything else about the model — and the mlx-lm generation loop — is unchanged.
"""

from __future__ import annotations

import glob
import json
import os
import struct
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from freetoken.utils import init_logger

logger = init_logger(__name__)

_NP_DTYPES = {"U32": np.uint32, "F16": np.float16, "BF16": np.uint16, "F32": np.float32}

_PROJS = ("gate_proj", "up_proj", "down_proj")
_PARTS = ("weight", "scales", "biases")


@dataclass(frozen=True)
class TensorEntry:
    shard: str
    dtype: str  # safetensors dtype string
    shape: Tuple[int, ...]
    start: int  # absolute byte offset of the tensor in the shard
    end: int


class SafetensorsIndex:
    """name -> TensorEntry across all shards of a checkpoint directory."""

    def __init__(self, model_dir: str):
        self.entries: Dict[str, TensorEntry] = {}
        shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        if not shards:
            raise FileNotFoundError(f"no .safetensors shards under {model_dir}")
        for shard in shards:
            with open(shard, "rb") as f:
                header_len = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_len))
            base = 8 + header_len
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                s, e = meta["data_offsets"]
                self.entries[name] = TensorEntry(
                    shard, meta["dtype"], tuple(meta["shape"]), base + s, base + e
                )

    def __contains__(self, name: str) -> bool:
        return name in self.entries

    def __getitem__(self, name: str) -> TensorEntry:
        return self.entries[name]


class LruTracker:
    """Pure LRU slot bookkeeping for one layer: expert id <-> slot id.

    Kept free of mx so it is unit-testable anywhere. ``touch`` order is recency;
    eviction picks the least recently used occupied slot.
    """

    def __init__(self, num_slots: int):
        self.num_slots = num_slots
        self.slot_of: "OrderedDict[int, int]" = OrderedDict()  # expert -> slot, LRU order
        self.free: List[int] = list(range(num_slots - 1, -1, -1))

    def lookup(self, experts: List[int]) -> Tuple[Dict[int, int], List[int]]:
        """(expert->slot for hits, ordered list of missing experts). Hits are touched."""
        hits: Dict[int, int] = {}
        misses: List[int] = []
        for e in experts:
            if e in self.slot_of:
                self.slot_of.move_to_end(e)
                hits[e] = self.slot_of[e]
            else:
                misses.append(e)
        return hits, misses

    def assign(self, expert: int) -> int:
        """Slot for a missing expert, evicting the LRU entry when full."""
        if self.free:
            slot = self.free.pop()
        else:
            _evicted, slot = self.slot_of.popitem(last=False)
        self.slot_of[expert] = slot
        return slot

    def most_recent(self, n: int) -> List[Tuple[int, int]]:
        """Up to n (expert, slot) pairs, most recently used first."""
        return [(e, s) for e, s in reversed(self.slot_of.items())][:n]

    def reset(self, num_slots: int) -> None:
        self.num_slots = num_slots
        self.slot_of.clear()
        self.free = list(range(num_slots - 1, -1, -1))


class ExpertStore:
    """pread-based fetch of single experts for one SwitchGLU (9 tensors).

    Converted MLX MoE repos come in two layouts:
    - stacked (qwen3_moe, qwen3_5_moe, ...): one ``[E, ...]`` tensor per
      (projection, part); expert e is a contiguous byte-range slice of it.
    - per-expert (olmoe-era conversions): ``...mlp.experts.{e}.{proj}.{part}``
      tensors; expert e is a whole tensor.
    Both reduce to "9 contiguous byte ranges per expert".
    """

    def __init__(self, index: SafetensorsIndex, weight_prefix: str, io_pool: ThreadPoolExecutor):
        import mlx.core as mx

        self._mx = mx
        self._pool = io_pool
        self._index = index
        stacked_names = [
            f"{weight_prefix}.{proj}.{part}" for proj in _PROJS for part in _PARTS
        ]
        if all(n in index for n in stacked_names):
            self.stacked = True
            self.names = stacked_names
            entries = [index[n] for n in stacked_names]
            self.num_experts = entries[0].shape[0]
            self.part_shapes = [e.shape[1:] for e in entries]
            self.part_dtypes = [e.dtype for e in entries]
            self.expert_nbytes = sum((e.end - e.start) // e.shape[0] for e in entries)
            self._stacked_entries = entries
        else:
            self._experts_prefix = (
                weight_prefix.rsplit(".", 1)[0] + ".experts"
                if "." in weight_prefix
                else "experts"
            )
            num = 0
            while f"{self._experts_prefix}.{num}.{_PROJS[0]}.{_PARTS[0]}" in index:
                num += 1
            if num == 0:
                raise KeyError(f"no expert weights on disk for {weight_prefix}")
            self.stacked = False
            self.num_experts = num
            first = [index[n] for n in self._per_expert_names(0)]
            self.part_shapes = [e.shape for e in first]
            self.part_dtypes = [e.dtype for e in first]
            self.expert_nbytes = sum(e.end - e.start for e in first)
        self._fds: Dict[str, int] = {}

    def _per_expert_names(self, expert: int) -> List[str]:
        return [
            f"{self._experts_prefix}.{expert}.{proj}.{part}"
            for proj in _PROJS
            for part in _PARTS
        ]

    def _ranges(self, expert: int) -> List[Tuple[str, int, int]]:
        """The 9 (shard, offset, nbytes) ranges of one expert."""
        if self.stacked:
            out = []
            for entry in self._stacked_entries:
                per = (entry.end - entry.start) // entry.shape[0]
                out.append((entry.shard, entry.start + expert * per, per))
            return out
        return [
            (e.shard, e.start, e.end - e.start)
            for e in (self._index[n] for n in self._per_expert_names(expert))
        ]

    def _fd(self, shard: str) -> int:
        fd = self._fds.get(shard)
        if fd is None:
            fd = os.open(shard, os.O_RDONLY)
            self._fds[shard] = fd
        return fd

    def _read_one(self, rng: Tuple[str, int, int]) -> bytes:
        shard, offset, nbytes = rng
        return os.pread(self._fd(shard), nbytes, offset)

    def fetch(self, experts: List[int]) -> List[List[Any]]:
        """For each expert: its 9 arrays in (proj x part) order. Reads run on the
        I/O pool (SSDs want queue depth); mx.array wrapping stays on the caller."""
        mx = self._mx
        jobs = [
            self._pool.submit(self._read_one, rng)
            for e in experts
            for rng in self._ranges(e)
        ]
        out: List[List[Any]] = []
        it = iter(jobs)
        for _e in experts:
            arrs = []
            for shape, dtype in zip(self.part_shapes, self.part_dtypes, strict=True):
                buf = next(it).result()
                np_arr = np.frombuffer(buf, dtype=_NP_DTYPES[dtype]).reshape(shape)
                arr = mx.array(np_arr)
                if dtype == "BF16":
                    arr = arr.view(mx.bfloat16)
                arrs.append(arr)
            out.append(arrs)
        return out

    def load_full_lazy(self) -> List[Any]:
        """The 9 stacked tensors as fresh lazy arrays (prefill streaming). Evaluating
        them materializes the full layer; dropping the references frees it."""
        mx = self._mx
        by_shard: Dict[str, Any] = {}

        def lazy(name: str) -> Any:
            shard = self._index[name].shard
            if shard not in by_shard:
                by_shard[shard] = mx.load(shard)
            return by_shard[shard][name]

        if self.stacked:
            return [lazy(name) for name in self.names]
        out = []
        for i in range(9):
            out.append(
                mx.stack(
                    [lazy(self._per_expert_names(e)[i]) for e in range(self.num_experts)]
                )
            )
        return out


class SlotCache:
    """Per-layer expert slot cache: 9 preallocated [S, ...] buffers + LRU.

    Besides the weight buffers it keeps two small device arrays so the decode
    forward never has to synchronize (the graph-compatible analogue of the CUDA
    engine's device-side slot cache):

    - ``lut``   [E] uint32: expert -> slot it *would* live in (stale allowed)
    - ``owner`` [S] int32:  slot -> expert it *actually* holds (-1 = empty)

    ``lut[inds]`` routes a lazy forward through the cache; ``owner[lut[inds]] ==
    inds`` is the lazy validity check evaluated once per token, after the fact.
    """

    def __init__(self, store: ExpertStore, num_slots: int):
        import mlx.core as mx

        self._mx = mx
        self.store = store
        self.lru = LruTracker(num_slots)
        self.hits = 0
        self.misses = 0
        self._alloc(num_slots)

    def _alloc(self, num_slots: int) -> None:
        mx = self._mx
        mx_dtypes = {
            "U32": mx.uint32, "F16": mx.float16, "BF16": mx.bfloat16, "F32": mx.float32
        }
        self.buffers = [
            mx.zeros((num_slots, *shape), dtype=mx_dtypes[dtype])
            for shape, dtype in zip(
                self.store.part_shapes, self.store.part_dtypes, strict=True
            )
        ]
        self.lut = mx.zeros((self.store.num_experts,), dtype=mx.uint32)
        self.owner = mx.full((num_slots,), -1, dtype=mx.int32)
        mx.eval(*self.buffers, self.lut, self.owner)

    @property
    def num_slots(self) -> int:
        return self.lru.num_slots

    def touch(self, experts: List[int]) -> None:
        """Recency update for an all-hit access (validated speculative token)."""
        hits, missing = self.lru.lookup(experts)
        assert not missing, "touch() on experts that are not resident"
        self.hits += len(hits)

    def acquire(self, experts: List[int]) -> Dict[int, int]:
        """expert -> slot for every requested expert, fetching misses from disk."""
        mx = self._mx
        hits, missing = self.lru.lookup(experts)
        self.hits += len(hits)
        self.misses += len(missing)
        if missing:
            fetched = self.store.fetch(missing)
            slots = []
            for e in missing:
                slot = self.lru.assign(e)
                slots.append(slot)
                hits[e] = slot
            # One scatter per buffer (not per expert): m single-slot writes each
            # rewrite the buffer's lazy graph; batching keeps the miss cost at the
            # I/O read plus one stacked copy.
            slot_idx = mx.array(slots, dtype=mx.uint32)
            for i, buf in enumerate(self.buffers):
                buf[slot_idx] = mx.stack([arrs[i] for arrs in fetched])
            expert_idx = mx.array(missing, dtype=mx.uint32)
            self.owner[slot_idx] = expert_idx.astype(mx.int32)
            self.lut[expert_idx] = slot_idx
        return hits

    def resize(self, num_slots: int) -> None:
        """Elastic resize; the most recently used experts survive."""
        mx = self._mx
        keep = self.lru.most_recent(num_slots)
        old_buffers = self.buffers
        self._alloc(num_slots)
        self.lru.reset(num_slots)
        for expert, old_slot in keep:
            new_slot = self.lru.assign(expert)
            for buf, old in zip(self.buffers, old_buffers, strict=True):
                buf[new_slot] = old[old_slot]
            self.owner[new_slot] = expert
            self.lut[expert] = new_slot
        mx.eval(*self.buffers, self.lut, self.owner)
        del old_buffers
        mx.clear_cache()


class OffloadSwitchGLU:
    """Drop-in replacement for mlx-lm's SwitchGLU, serving experts from a SlotCache.

    Not an nn.Module on purpose: it must own large mx buffers without them showing
    up in ``model.parameters()`` (they are state, not weights). mlx-lm's model code
    only ever calls it: ``y = self.switch_mlp(x, indices)``.
    """

    def __init__(self, store: ExpertStore, cache: SlotCache, activation, state: "OffloadState"):
        import mlx.core as mx

        self._mx = mx
        self.store = store
        self.cache = cache
        self.activation = activation
        self.state = state

    # -- the two serving paths ------------------------------------------------

    def _gather(self, x, w, s, b, indices, sorted_indices=False):
        return self._mx.gather_qmm(
            x, w, s, b,
            rhs_indices=indices,
            transpose=True,
            group_size=self.state.group_size,
            bits=self.state.bits,
            mode=self.state.mode,
            sorted_indices=sorted_indices,
        )

    def _run(self, x, idx, bufs, sorted_indices=False):
        (gw, gs_, gb), (uw, us, ub), (dw, db_, dbias) = bufs
        x = self._mx.expand_dims(x, (-2, -3))
        x_up = self._gather(x, uw, us, ub, idx, sorted_indices)
        x_gate = self._gather(x, gw, gs_, gb, idx, sorted_indices)
        x = self._gather(self.activation(x_up, x_gate), dw, db_, dbias, idx, sorted_indices)
        return x.squeeze(-2)

    def __call__(self, x, indices):
        mx = self._mx
        n_tokens = 1
        for d in indices.shape[:-1]:
            n_tokens *= d
        if self.state.speculating and n_tokens == 1:
            return self._forward_speculative(x, indices)

        np_inds = np.array(indices, copy=False)  # forces evaluation up to the router
        uniq = np.unique(np_inds)
        if len(uniq) > self.cache.num_slots:
            return self._forward_streamed(x, indices, np_inds)

        slot_map = self.cache.acquire([int(e) for e in uniq])
        lut = np.zeros(self.store.num_experts, dtype=np.uint32)
        for e, slot in slot_map.items():
            lut[e] = slot
        slot_inds = mx.array(lut[np_inds])
        b = self.cache.buffers
        bufs = ((b[0], b[1], b[2]), (b[3], b[4], b[5]), (b[6], b[7], b[8]))
        return self._run(x, slot_inds, bufs)

    def _forward_speculative(self, x, indices):
        """Decode path: fully lazy, zero CPU synchronization.

        Routes through the device lut; whether every routed expert was actually
        resident is itself a lazy value (``ok``), registered with the shared state
        and checked once per token by the generation loop. A miss makes this
        token's output garbage — the loop rolls the caches back, installs the
        missing experts and re-runs the step (FreeToken's device-side LRU turned
        into speculate-and-verify, which is what MLX's lazy graphs support)."""
        mx = self._mx
        cache = self.cache
        slot_inds = mx.take(cache.lut, indices)
        ok = mx.all(mx.take(cache.owner, slot_inds) == indices.astype(mx.int32))
        self.state.pending.append((self, indices, ok))
        b = cache.buffers
        bufs = ((b[0], b[1], b[2]), (b[3], b[4], b[5]), (b[6], b[7], b[8]))
        return self._run(x, slot_inds, bufs)

    def _forward_streamed(self, x, indices, np_inds):
        """Prefill: more experts than slots -> serve from a materialized full layer
        (prefetched by the previous layer where possible), never thrashing the
        cache with demand misses. The chunk's hottest experts ARE admitted, copied
        device-side from the already-materialized stack (free of disk I/O), so
        decode starts warm — FreeToken's prefill-warms-the-cache behavior."""
        mx = self._mx
        full = self.state.take_prefetch(self) or self.store.load_full_lazy()
        mx.eval(*full)
        self.state.start_prefetch_after(self)
        bufs = ((full[0], full[1], full[2]), (full[3], full[4], full[5]),
                (full[6], full[7], full[8]))
        if indices.dtype != mx.uint32:
            indices = indices.astype(mx.uint32)
        y = self._run(x, indices, bufs)
        self._admit_from_full(full, np_inds)
        mx.eval(y)  # the full layer must be droppable right now, not at graph eval
        del full, bufs
        return y

    def _admit_from_full(self, full, np_inds) -> None:
        mx = self._mx
        cache = self.cache
        counts = np.bincount(np_inds.reshape(-1), minlength=self.store.num_experts)
        hot = np.nonzero(counts)[0]
        # Ascending by frequency: the hottest expert is installed last and is
        # therefore the most recently used from the LRU's point of view.
        hot = hot[np.argsort(counts[hot], kind="stable")][-cache.num_slots:]
        hits, missing = cache.lru.lookup([int(e) for e in hot])
        _ = hits  # recency refresh for already-resident hot experts
        if not missing:
            return
        slots = [cache.lru.assign(e) for e in missing]
        slot_idx = mx.array(slots, dtype=mx.uint32)
        expert_idx = mx.array(missing, dtype=mx.uint32)
        for buf, stacked in zip(cache.buffers, full, strict=True):
            buf[slot_idx] = stacked[expert_idx]
        cache.owner[slot_idx] = expert_idx.astype(mx.int32)
        cache.lut[expert_idx] = slot_idx


class OffloadState:
    """Everything shared across the per-layer OffloadSwitchGLUs of one model."""

    def __init__(self, group_size: int, bits: int, mode: str = "affine"):
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self.glus: List[OffloadSwitchGLU] = []
        self._prefetch: Tuple[int, List[Any]] | None = None
        # (glu, lazy routed indices, lazy all-resident flag) per speculative layer
        # of the token currently being generated. See begin_token/commit_token.
        self.pending: List[Tuple[OffloadSwitchGLU, Any, Any]] = []
        # Zero-sync speculative serving is only sound when the generation loop
        # drives the speculate/verify/rollback protocol; it opts in per phase.
        self.speculating = False

    # -- speculate-and-verify decode -------------------------------------------

    def begin_token(self) -> None:
        self.pending.clear()

    def commit_token(self) -> bool:
        """After the step's eval: True if every routed expert was resident (the
        token stands; recency is updated). False if any layer missed — the caller
        must roll its KV/recurrent caches back, then the misses are installed
        here so the re-run is guaranteed all-hit."""
        all_ok = True
        for glu, indices, ok in self.pending:
            experts = [int(e) for e in np.unique(np.array(indices, copy=False))]
            if bool(ok.item()):
                glu.cache.touch(experts)
            else:
                all_ok = False
                glu.cache.acquire(experts)  # fetch + install + lut/owner update
        self.pending.clear()
        return all_ok

    def pending_oks(self) -> List[Any]:
        return [ok for _, _, ok in self.pending]

    # -- prefill double buffering ---------------------------------------------

    def take_prefetch(self, glu: OffloadSwitchGLU) -> List[Any] | None:
        i = self.glus.index(glu)
        if self._prefetch is not None and self._prefetch[0] == i:
            arrs = self._prefetch[1]
            self._prefetch = None
            return arrs
        self._prefetch = None
        return None

    def start_prefetch_after(self, glu: OffloadSwitchGLU) -> None:
        import mlx.core as mx

        i = self.glus.index(glu)
        if i + 1 < len(self.glus):
            arrs = self.glus[i + 1].store.load_full_lazy()
            mx.async_eval(*arrs)
            self._prefetch = (i + 1, arrs)

    # -- stats / elasticity ----------------------------------------------------

    def totals(self) -> Tuple[int, int, int]:
        h = sum(g.cache.hits for g in self.glus)
        m = sum(g.cache.misses for g in self.glus)
        return h, m, sum(g.cache.num_slots for g in self.glus)

    def cache_bytes(self) -> int:
        return sum(g.cache.num_slots * g.store.expert_nbytes for g in self.glus)

    def resize_total(self, total_slots: int) -> int:
        """Distribute a total slot budget evenly across layers; returns the total."""
        per_layer = max(1, total_slots // max(1, len(self.glus)))
        for g in self.glus:
            g.cache.resize(per_layer)
        return per_layer * len(self.glus)


def _iter_modules(module, path=""):
    """(dotted_path, module) over an mlx.nn tree, including list/dict children.

    Order matters below: mlx.nn.Module subclasses dict, so the Module check
    (duck-typed via .children to keep mlx imports lazy) must come first — a plain
    isinstance(dict) branch would swallow every submodule."""
    yield path, module
    children = module.children() if hasattr(module, "children") else {}
    for name, child in children.items():
        yield from _iter_child(child, f"{path}.{name}" if path else name)


def _iter_child(child, path):
    if hasattr(child, "children"):  # nn.Module (a dict subclass -- check first)
        yield from _iter_modules(child, path)
    elif isinstance(child, list):
        for i, c in enumerate(child):
            yield from _iter_child(c, f"{path}.{i}")
    elif isinstance(child, dict):
        for k, c in child.items():
            yield from _iter_child(c, f"{path}.{k}")


def _is_switch_glu(module) -> bool:
    return all(hasattr(module, p) for p in _PROJS) and hasattr(module, "activation")


def _set_by_path(root, dotted: str, value) -> None:
    parts = dotted.split(".")
    obj = root
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    last = parts[-1]
    if last.isdigit():
        obj[int(last)] = value
    else:
        setattr(obj, last, value)


def attach_expert_offload(model, model_dir: str, slots_per_layer: int,
                          io_threads: int = 8) -> OffloadState:
    """Replace every SwitchGLU in ``model`` with an offload-serving version.

    Must run on a ``lazy=True``-loaded model: the stacked expert arrays are still
    unevaluated graph nodes and are dropped here without ever being materialized.
    Returns the shared OffloadState (stats, elastic resize, prefetch chain).
    """
    quant = json.load(open(os.path.join(model_dir, "config.json"))).get("quantization")
    if not quant:
        raise ValueError("expert offload needs a quantized MLX checkpoint")
    index = SafetensorsIndex(model_dir)
    state = OffloadState(
        int(quant["group_size"]), int(quant["bits"]), str(quant.get("mode", "affine"))
    )
    pool = ThreadPoolExecutor(max_workers=io_threads)

    targets = [
        (path, mod) for path, mod in _iter_modules(model) if path and _is_switch_glu(mod)
    ]
    for path, mod in targets:
        try:
            store = ExpertStore(index, path, pool)
        except KeyError:  # module tree path has no weights on disk under either layout
            continue
        cache = SlotCache(store, max(1, slots_per_layer))
        glu = OffloadSwitchGLU(store, cache, mod.activation, state)
        state.glus.append(glu)
        _set_by_path(model, path, glu)
    if not state.glus:
        raise ValueError(
            "no offloadable SwitchGLU layers found (not an expert-parallel MoE "
            "checkpoint, or its weight names do not match the module tree)"
        )

    total = sum(g.cache.num_slots for g in state.glus)
    per_expert = state.glus[0].store.expert_nbytes
    logger.info(
        f"expert offload: {len(state.glus)} MoE layers, "
        f"{state.glus[0].store.num_experts} experts/layer, "
        f"cache {slots_per_layer} slots/layer ({total} total, "
        f"{total * per_expert / 2**30:.2f} GiB), {per_expert / 2**20:.2f} MiB/expert"
    )
    return state
