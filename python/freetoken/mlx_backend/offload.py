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

# Experimental: inline-serve first-offense decode misses instead of admitting
# them (see the call site in OffloadSwitchGLU.__call__ for the measured tradeoff).
_ADMIT_FILTER = os.environ.get("FREETOKEN_MLX_ADMIT_FILTER", "") == "1"

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

    def read_jobs(self, expert: int) -> List[Any]:
        """Submit the 9 preads of one expert to the I/O pool; returns the futures."""
        return [self._pool.submit(self._read_one, rng) for rng in self._ranges(expert)]

    def arrays_from(self, jobs: List[Any]) -> List[Any]:
        """The 9 mx arrays of one expert from its read futures."""
        mx = self._mx
        arrs = []
        for job, shape, dtype in zip(
            jobs, self.part_shapes, self.part_dtypes, strict=True
        ):
            np_arr = np.frombuffer(job.result(), dtype=_NP_DTYPES[dtype]).reshape(shape)
            arr = mx.array(np_arr)
            if dtype == "BF16":
                arr = arr.view(mx.bfloat16)
            arrs.append(arr)
        return arrs

    def fetch(self, experts: List[int]) -> List[List[Any]]:
        """For each expert: its 9 arrays in (proj x part) order. Reads run on the
        I/O pool (SSDs want queue depth); mx.array wrapping stays on the caller."""
        all_jobs = [self.read_jobs(e) for e in experts]
        return [self.arrays_from(jobs) for jobs in all_jobs]

    def fetch_stacked(
        self, experts: List[int], jobs_by_expert: Dict[int, List[Any]]
    ) -> List[Any]:
        """The 9 parts of ``experts`` stacked to [m, ...] — ONE mx array per part.

        Per-expert wrapping costs ~9 mx.array creations per expert per layer and
        dominated the miss path (Python, not I/O); stacking the raw bytes in numpy
        first turns that into 9 creations per *layer*."""
        mx = self._mx
        all_jobs = [
            jobs_by_expert.pop(e, None) or self.read_jobs(e) for e in experts
        ]
        out = []
        for i, (shape, dtype) in enumerate(
            zip(self.part_shapes, self.part_dtypes, strict=True)
        ):
            np_dt = _NP_DTYPES[dtype]
            stacked = np.stack(
                [
                    np.frombuffer(jobs[i].result(), dtype=np_dt).reshape(shape)
                    for jobs in all_jobs
                ]
            )
            arr = mx.array(stacked)
            if dtype == "BF16":
                arr = arr.view(mx.bfloat16)
            out.append(arr)
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
        # expert -> in-flight read futures, issued ahead of demand (predictive
        # prefetch keyed on the previous token's routing). Consumed by acquire.
        self._inflight: Dict[int, List[Any]] = {}
        # Admission filter (decode): a FIRST miss (none in the last _ADMIT_WINDOW
        # tokens) is served inline without a slot; a recurring miss earns one.
        # One-off tail experts would otherwise evict genuinely hot slots (cache
        # thrash) and pay an install copy for a single use.
        self._last_miss: Dict[int, int] = {}
        self._token_tick = 0
        self._alloc(num_slots)

    _ADMIT_WINDOW = 64  # tokens: "recurring" means a second miss within this many

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

    def prefetch(self, experts: List[int]) -> None:
        """Issue reads for experts predicted to be needed soon (not resident, not
        already in flight). By the time demand arrives the bytes are usually in."""
        for e in experts:
            if e not in self.lru.slot_of and e not in self._inflight:
                if len(self._inflight) > 4 * self.lru.num_slots:
                    self._inflight.pop(next(iter(self._inflight)))
                self._inflight[e] = self.store.read_jobs(e)

    def install(self, missing: List[int]) -> Dict[int, int]:
        """Fetch ``missing`` from disk and install them into slots (batched: one
        stacked read + one scatter per part). Returns expert -> slot."""
        mx = self._mx
        stacked = self.store.fetch_stacked(missing, self._inflight)
        assigned: Dict[int, int] = {}
        slots = []
        for e in missing:
            slot = self.lru.assign(e)
            slots.append(slot)
            assigned[e] = slot
        slot_idx = mx.array(slots, dtype=mx.uint32)
        for buf, part in zip(self.buffers, stacked, strict=True):
            buf[slot_idx] = part
        expert_idx = mx.array(missing, dtype=mx.uint32)
        self.owner[slot_idx] = expert_idx.astype(mx.int32)
        self.lut[expert_idx] = slot_idx
        return assigned

    def acquire(self, experts: List[int]) -> Dict[int, int]:
        """expert -> slot for every requested expert, fetching misses from disk."""
        hits, missing = self.lru.lookup(experts)
        self.hits += len(hits)
        self.misses += len(missing)
        if missing:
            hits.update(self.install(missing))
        return hits

    def split_admission(self, experts: List[int]) -> Tuple[List[int], List[int]]:
        """(experts to admit into slots, experts to serve inline this step).
        Callers must have looked residency up already: ``experts`` are misses."""
        admit: List[int] = []
        inline: List[int] = []
        now = self._token_tick
        for e in experts:
            last = self._last_miss.get(e)
            self._last_miss[e] = now
            (admit if last is not None and now - last <= self._ADMIT_WINDOW else inline).append(e)
        return admit, inline

    def decay_streaks(self) -> None:
        """Once per token: advance the admission clock; occasionally drop stale
        miss timestamps so the dict stays bounded."""
        self._token_tick += 1
        if self._token_tick % (8 * self._ADMIT_WINDOW) == 0:
            horizon = self._token_tick - 2 * self._ADMIT_WINDOW
            self._last_miss = {
                e: t for e, t in self._last_miss.items() if t >= horizon
            }

    def resize(self, num_slots: int) -> None:
        """Elastic resize; the most recently used experts survive."""
        mx = self._mx
        keep = self.lru.most_recent(num_slots)
        old_buffers = self.buffers
        self._inflight.clear()
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
        # Unique experts this layer routed to on the most recent token — the
        # predictor for the next token's prefetch (MoE routing is sticky).
        self.last_routed: Any = None

    # -- the two serving paths ------------------------------------------------

    def _qmm(self, x, w, s, b):
        """Single-expert quantized matmul (inline miss serving)."""
        return self._mx.quantized_matmul(
            x, w, s, b,
            transpose=True,
            group_size=self.state.group_size,
            bits=self.state.bits,
            mode=self.state.mode,
        )

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
        if self.state.speculating and n_tokens <= self.state.spec_window:
            return self._forward_speculative(x, indices)

        np_inds = np.array(indices, copy=False)  # forces evaluation up to the router
        uniq = np.unique(np_inds)
        if len(uniq) > self.cache.num_slots:
            return self._forward_streamed(x, indices, np_inds)

        self.last_routed = uniq
        cache = self.cache
        hits, missing = cache.lru.lookup([int(e) for e in uniq])
        cache.hits += len(hits)
        cache.misses += len(missing)
        inline: List[int] = []
        if missing:
            # Admission filter (opt-in, FREETOKEN_MLX_ADMIT_FILTER=1): serve a
            # first-offense miss inline and only give recurring misses a slot.
            # Measured: helps small expert pools with tight caches (OLMoE 64
            # experts @ 37% slots: +22%), hurts long-tail pools (Ornith 256
            # experts @ 35%: -30%, recurrence outruns the window) — so the
            # predictable admit-all is the default.
            if n_tokens == 1 and _ADMIT_FILTER:
                admit, inline = cache.split_admission(missing)
            else:
                admit = missing
            if admit:
                cache.install(admit)
        # Map expert -> slot through the device lut (install just refreshed it):
        # no per-layer host lut rebuild or upload.
        slot_inds = mx.take(cache.lut, indices)
        b = cache.buffers
        bufs = ((b[0], b[1], b[2]), (b[3], b[4], b[5]), (b[6], b[7], b[8]))
        if not inline:
            return self._run(x, slot_inds, bufs)
        # Inline-served misses (first offense, decode): the slot gather covers the
        # resident experts; positions routed to a non-resident expert are masked
        # out and replaced by a direct compute on weights read for this step only
        # (no slot eviction, no install copy) — the unified-memory analogue of
        # FreeToken's hybrid miss serving.
        mask = mx.take(cache.owner, slot_inds) == indices.astype(mx.int32)
        y = self._run(x, slot_inds, bufs) * mask[..., None].astype(x.dtype)
        flat = np_inds.reshape(-1)
        for e in inline:
            arrs = self.store.arrays_from(
                cache._inflight.pop(e, None) or self.store.read_jobs(e)
            )
            (gw, gs_, gb), (uw, us, ub), (dw, ds_, db_) = (
                arrs[0:3], arrs[3:6], arrs[6:9],
            )
            x_up = self._qmm(x, uw, us, ub)
            x_gate = self._qmm(x, gw, gs_, gb)
            y_e = self._qmm(self.activation(x_up, x_gate), dw, ds_, db_)
            pos = int(np.nonzero(flat == e)[0][0])
            y[..., pos, :] = y_e
        return y

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
        # The full layer must be droppable RIGHT NOW, not whenever the graph is
        # next evaluated. Evaluating y alone is not enough: the admission scatter
        # (buf[slots] = stacked[experts]) leaves the cache buffers holding a lazy
        # graph that still references this layer's whole stack, so every streamed
        # layer stays pinned — 40 GiB of them on a 512-expert model.
        cache = self.cache
        mx.eval(y, *cache.buffers, cache.owner, cache.lut)
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
        # Widest token window the lazy lut path may serve at once. 1 for plain
        # decode; k+1 when a draft model verifies k tokens per forward (the
        # whole window rolls back together on an expert miss).
        self.spec_window = 1

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
            uniq = np.unique(np.array(indices, copy=False))
            glu.last_routed = uniq
            experts = [int(e) for e in uniq]
            if bool(ok.item()):
                glu.cache.touch(experts)
            else:
                all_ok = False
                glu.cache.acquire(experts)  # fetch + install + lut/owner update
        self.pending.clear()
        return all_ok

    def pending_oks(self) -> List[Any]:
        return [ok for _, _, ok in self.pending]

    def prefetch_predicted(self) -> None:
        """Per-token housekeeping before the forward: age the admission streaks,
        and issue reads for each layer's previously routed, currently non-resident
        experts (inline-served misses in particular) — routing is sticky enough
        that many of the coming misses are already in flight when demanded."""
        for glu in self.glus:
            glu.cache.decay_streaks()
            if glu.last_routed is not None:
                glu.cache.prefetch([int(e) for e in glu.last_routed])

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
