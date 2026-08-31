"""The prefix cache's second tier: boundary snapshots kept on the SSD.

The in-memory prefix store (prefix_cache.py) makes turn 2+ of a conversation
cheap, and it dies with the process. Whatever prompt repeats across server
runs -- an agent's fixed system prompt is the case that matters: ~9k tokens,
~75 s of compute-bound prefill on the 27B, paid again after every restart and
every reload -- is prefilled once and read back from disk after that.

What is stored, and why in this shape. Prompts are cut into blocks of
BOUNDARY_TOKENS tokens, each block addressed by a hash chained from its
predecessor's (so a hash names the whole token prefix up to its block, and
two prompts sharing a system prompt share every block of it). Per block:

  * the attention KV of the block's tokens (``<hash>.kv``: 16 MB on the 27B
    -- 16 attention layers x K,V x 4 heads x 256 dims x 256 tokens, bf16),
    shared by every prompt that starts with the same tokens;
  * optionally the recurrent state at the block's end (``<hash>.rec``:
    151 MB on the 27B -- 48 GDN layers x 48 heads x 128 x 128 fp32). A
    hybrid model can only resume at a position whose recurrent state exists,
    and that state is a snapshot, not a slice, so keeping one at every block
    of every prompt would cost ten times the KV. They are written where a
    resume is likely: every REC_EVERY-th block regardless, the last full
    block of a prompt (where the next session's user message will diverge
    from this one's), any block a request resumed at, and the block where a
    prompt was found to diverge from what the store already held.

A lookup walks the chain and resumes at the deepest block that has what the
model needs (KV, plus recurrent state on hybrid models); the prefill then
covers the rest. The prefill of the speculative path also ends a chunk at the
positions the tier asks for (``cut_points``), so those snapshots exist.

Measured on the M1 Max SSD: a 9k-token restore reads ~0.75 GB in well under
a second; writes go through a background thread at ~1 GB/s, so the prefill
never waits for the disk. Snapshots are converted to host memory in the
scheduler thread (bf16 through a uint16 view -- numpy has no bf16) so the
writer thread never touches MLX; the copies are pageable, not wired.

Files are safetensors, so ``mx.load`` reads them lazily and only the tensors
a restore needs are read. A manifest carries sizes and LRU stamps; eviction
is oldest-first, which drops a chain's tail before its head (a hit touches
the whole chain, so a head is never older than its tail). A store belongs to
one model (weights + config hashed into the directory name) and one process
at a time (advisory lock); a second server on the same model runs without
the tier rather than racing on the manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from maxtoken.utils import init_logger

logger = init_logger(__name__)

MANIFEST_VERSION = 1
DEFAULT_DIR = "~/.maxtoken/prefix-cache"
DEFAULT_BUDGET_GB = 20.0
# Prompts shorter than this are not worth a disk entry: their prefill costs
# seconds, and each block of the ladder carries a 100+ MB recurrent snapshot.
MIN_PROMPT_TOKENS = 512
# A recurrent snapshot every this many blocks (1024 tokens), whatever else
# the policy decides, so a prompt that diverges anywhere resumes within a
# few seconds of prefill of the divergence.
REC_EVERY = 4
# The writer's backlog cap: past this, new snapshots are skipped (logged),
# never queued into unbounded host memory.
MAX_PENDING_BYTES = 3 << 30
# Recurrent snapshots kept in host memory until a prompt's prefill ends,
# for the case where the last block's snapshot was not written eagerly.
WINDOW_SLOTS = 4

_ST_DTYPES = {
    "bfloat16": "BF16",
    "float16": "F16",
    "float32": "F32",
    "int32": "I32",
    "uint32": "U32",
    "int64": "I64",
    "uint16": "U16",
    "uint8": "U8",
    "int8": "I8",
    "int16": "I16",
    "bool_": "BOOL",
}


# ----------------------------------------------------------------------------
# hashing, keys, files


def chain_hashes(tokens: Sequence[int], block: int, n_blocks: Optional[int] = None) -> List[str]:
    """Hashes of the first ``n_blocks`` full blocks, each chained from the
    previous one so hash ``i`` names ``tokens[:i*block]`` exactly."""
    toks = np.asarray(tokens, dtype=np.int32)
    n = len(toks) // block if n_blocks is None else n_blocks
    out: List[str] = []
    h = b""
    for i in range(n):
        h = hashlib.sha256(h + toks[i * block : (i + 1) * block].tobytes()).digest()
        out.append(h.hex()[:32])
    return out


def restore_blocks(prompt_len: int, block: int) -> int:
    """Blocks a prompt of this length can resume at: at least one token must
    remain for the model to process."""
    return max(0, (prompt_len - 1) // block)


def model_key(name: str, model_dir: Optional[str]) -> str:
    """Directory name for one model's store: the served name plus a digest
    of config.json and the weight files' names and sizes, so a requantized
    checkpoint under the same name gets a store of its own."""
    h = hashlib.sha256()
    if model_dir and os.path.isdir(model_dir):
        cfg = os.path.join(model_dir, "config.json")
        if os.path.isfile(cfg):
            with open(cfg, "rb") as f:
                h.update(f.read())
        for entry in sorted(os.listdir(model_dir)):
            if entry.endswith(".safetensors"):
                try:
                    h.update(f"{entry}:{os.path.getsize(os.path.join(model_dir, entry))}".encode())
                except OSError:
                    pass
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip("/").replace("/", "--"))[:80]
    return f"{safe}-{h.hexdigest()[:12]}"


def resolve_model_dir(model_path: str) -> Optional[str]:
    """The checkpoint's local directory (hub ids through the cache mlx_lm.load
    already populated); None when it is not on this machine."""
    if os.path.isdir(model_path):
        return model_path
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(
            model_path, local_files_only=True, allow_patterns=["*.safetensors", "*.json"]
        )
    except Exception:  # noqa: BLE001 -- not cached locally, or no hub library
        return None


def store_dir(root: str, model_path: str) -> str:
    return os.path.join(
        os.path.expanduser(root), model_key(model_path, resolve_model_dir(model_path))
    )


def _to_host(a) -> Tuple[str, np.ndarray]:
    """(safetensors dtype, numpy array of the raw bytes) for an mx array."""
    import mlx.core as mx

    if a.dtype == mx.bfloat16:
        return "BF16", np.ascontiguousarray(np.array(mx.view(a, mx.uint16)))
    name = _ST_DTYPES.get(str(a.dtype).split(".")[-1])
    if name is None:
        raise TypeError(f"prefix disk: no safetensors dtype for {a.dtype}")
    return name, np.ascontiguousarray(np.array(a))


def write_safetensors(path: str, tensors: Dict[str, Tuple[str, np.ndarray]], metadata: Dict[str, str]) -> int:
    """Write ``tensors`` ({name: (dtype tag, array)}) atomically; returns the
    file size. The format is small enough to write directly: an 8-byte
    header length, a JSON header, the raw buffers in header order."""
    header: Dict[str, Any] = {"__metadata__": {k: str(v) for k, v in metadata.items()}}
    offset = 0
    order: List[np.ndarray] = []
    for name, (dtype, arr) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + arr.nbytes],
        }
        offset += arr.nbytes
        order.append(arr)
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for arr in order:
            f.write(memoryview(arr).cast("B"))
    os.replace(tmp, path)
    return 8 + len(raw) + offset


def cache_layout(cache: Sequence[Any]) -> Optional[List[str]]:
    """Per layer, what the tier stores: ``kv`` for a trimmable (keys, values)
    cache, ``rec`` for a recurrent one whose state is a list of arrays. Any
    other cache type (rotating windows, quantized KV) leaves the tier off."""
    kinds: List[str] = []
    for c in cache:
        if type(c).__name__ in _UNSTORABLE_CACHES:
            return None
        kinds.append("kv" if c.is_trimmable() else "rec")
    return kinds


_UNSTORABLE_CACHES = frozenset({
    "RotatingKVCache", "QuantizedKVCache", "BatchKVCache", "BatchRotatingKVCache",
    "ChunkedKVCache", "CacheList", "MambaCache",
})


# ----------------------------------------------------------------------------
# the store


@dataclass
class _Block:
    index: int
    kv_bytes: int = 0
    rec_bytes: int = 0
    used: float = 0.0
    kv_ready: bool = False
    rec_ready: bool = False
    kv_pending: bool = False
    rec_pending: bool = False

    @property
    def kv_known(self) -> bool:
        return self.kv_ready or self.kv_pending

    @property
    def rec_known(self) -> bool:
        return self.rec_ready or self.rec_pending


class DiskPrefixStore:
    """Block store for one model under ``root/<model key>``. All public
    methods are called from the scheduler thread; the writer thread only
    writes files and the manifest."""

    def __init__(
        self,
        root: str,
        max_bytes: int,
        key: str,
        layout: Sequence[str],
        *,
        block_tokens: int = 256,
        rec_every: int = REC_EVERY,
        min_prompt_tokens: int = MIN_PROMPT_TOKENS,
        clock: Callable[[], float] = time.time,
    ):
        self.dir = os.path.join(os.path.expanduser(root), key)
        os.makedirs(self.dir, exist_ok=True)
        self.max_bytes = int(max_bytes)
        self.block = int(block_tokens)
        self.layout = list(layout)
        self.needs_rec = "rec" in self.layout
        self.rec_every = max(1, int(rec_every))
        self.min_prompt_tokens = int(min_prompt_tokens)
        self._clock = clock
        self.blocks: Dict[str, _Block] = {}
        self._lock = threading.Lock()
        self._queue: "queue.Queue[Optional[tuple]]" = queue.Queue()
        self._pending_bytes = 0
        self._window: Dict[str, Tuple[int, Dict[str, Tuple[str, np.ndarray]]]] = {}
        self._window_order: List[str] = []
        self.hits = 0
        self.misses = 0
        self.restored_tokens = 0
        self.written_bytes = 0
        self.skipped_writes = 0
        self.disabled: Optional[str] = None
        self._lock_fd: Optional[int] = None
        if not self._acquire_lock():
            self.disabled = "another process holds this store"
            logger.warning(f"prefix disk: {self.dir} is in use by another process; tier off")
            return
        self._load_manifest()
        self._writer = threading.Thread(target=self._write_loop, name="prefix-disk", daemon=True)
        self._writer.start()

    # ------------------------------------------------------------ lifecycle

    def _acquire_lock(self) -> bool:
        try:
            import fcntl

            fd = os.open(os.path.join(self.dir, "lock"), os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd)
                return False
            self._lock_fd = fd
            return True
        except Exception:  # noqa: BLE001 -- no flock: run unlocked rather than not at all
            return True

    def _manifest_path(self) -> str:
        return os.path.join(self.dir, "manifest.json")

    def _kv_path(self, h: str) -> str:
        return os.path.join(self.dir, f"{h}.kv")

    def _rec_path(self, h: str) -> str:
        return os.path.join(self.dir, f"{h}.rec")

    def _load_manifest(self) -> None:
        try:
            with open(self._manifest_path(), "r", encoding="utf-8") as f:
                m = json.load(f)
        except FileNotFoundError:
            return
        except Exception as exc:  # noqa: BLE001 -- unreadable manifest: start over
            logger.warning(f"prefix disk: manifest unreadable ({exc}); starting empty")
            self._wipe()
            return
        if (
            m.get("version") != MANIFEST_VERSION
            or m.get("block_tokens") != self.block
            or m.get("layout") != ",".join(self.layout)
        ):
            logger.info("prefix disk: store layout changed; discarding old snapshots")
            self._wipe()
            return
        for h, rec in m.get("blocks", {}).items():
            b = _Block(index=int(rec.get("i", 0)), used=float(rec.get("used", 0.0)))
            kv = int(rec.get("kv", 0))
            if kv and self._file_is(self._kv_path(h), kv):
                b.kv_bytes, b.kv_ready = kv, True
            rc = int(rec.get("rec", 0))
            if rc and self._file_is(self._rec_path(h), rc):
                b.rec_bytes, b.rec_ready = rc, True
            if b.kv_ready:
                self.blocks[h] = b
            elif b.rec_ready:
                self._unlink(self._rec_path(h))
        logger.info(
            f"prefix disk: {len(self.blocks)} blocks, {self.total_bytes() / 2**30:.2f} GiB "
            f"in {self.dir} (budget {self.max_bytes / 2**30:.0f} GiB)"
        )

    @staticmethod
    def _file_is(path: str, size: int) -> bool:
        try:
            return os.path.getsize(path) == size
        except OSError:
            return False

    @staticmethod
    def _unlink(path: str) -> None:
        try:
            os.unlink(path)
        except OSError:
            pass

    def _wipe(self) -> None:
        for entry in os.listdir(self.dir):
            if entry.endswith((".kv", ".rec", ".tmp")) or entry in ("manifest.json", "stats.json"):
                self._unlink(os.path.join(self.dir, entry))
        self.blocks.clear()

    def close(self, timeout: float = 30.0) -> None:
        if self.disabled:
            return
        self._queue.put(None)
        self._writer.join(timeout)
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None

    def flush(self, timeout: float = 60.0) -> None:
        """Wait for queued writes (tests and shutdown)."""
        if self.disabled:
            return
        # The newest boundary's snapshot may still be dispatched, not queued.
        self._flush_snapshot()
        done = threading.Event()
        self._queue.put(("sync", done))
        done.wait(timeout)

    def _disable(self, reason: str) -> None:
        if not self.disabled:
            self.disabled = reason
            logger.warning(f"prefix disk: tier off -- {reason}")

    # ------------------------------------------------------------ accounting

    def total_bytes(self) -> int:
        return sum(b.kv_bytes + b.rec_bytes for b in self.blocks.values())

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return self._stats_locked()

    def _stats_locked(self) -> Dict[str, Any]:
        n_rec = sum(1 for b in self.blocks.values() if b.rec_ready)
        return {
            "dir": self.dir,
            "blocks": len(self.blocks),
            "rec_blocks": n_rec,
            "bytes": self.total_bytes(),
            "budget_bytes": self.max_bytes,
            "block_tokens": self.block,
            "hits": self.hits,
            "misses": self.misses,
            "restored_tokens": self.restored_tokens,
            "written_bytes": self.written_bytes,
            "skipped_writes": self.skipped_writes,
            "pending_bytes": self._pending_bytes,
            "disabled": self.disabled,
        }

    def _touch(self, hashes: Sequence[str]) -> None:
        now = self._clock()
        for h in hashes:
            b = self.blocks.get(h)
            if b is not None:
                b.used = now

    def _make_room(self, nbytes: int, keep: Sequence[str]) -> bool:
        """Evict oldest-first until ``nbytes`` more fit; False if they cannot
        (the writer backlog is full, or everything left is in use)."""
        if self._pending_bytes + nbytes > MAX_PENDING_BYTES:
            return False
        protected = set(keep)
        while self.total_bytes() + self._pending_bytes + nbytes > self.max_bytes:
            # Oldest first; among equals the deepest block, so a chain loses
            # its tail (still resumable earlier) before its head (orphaning
            # everything behind it).
            victims = [
                (b.used, -b.index, h)
                for h, b in self.blocks.items()
                if h not in protected and not (b.kv_pending or b.rec_pending)
            ]
            if not victims:
                return False
            self._drop(min(victims)[2])
        return True

    def _drop(self, h: str) -> None:
        b = self.blocks.pop(h, None)
        if b is None:
            return
        self._unlink(self._kv_path(h))
        self._unlink(self._rec_path(h))

    # ------------------------------------------------------------ lookup

    def chain(self, tokens: Sequence[int], n_blocks: Optional[int] = None) -> List[str]:
        return chain_hashes(tokens, self.block, n_blocks)

    def known_blocks(self, hashes: Sequence[str]) -> int:
        """Length of the leading run of hashes whose KV is on disk."""
        n = 0
        for h in hashes:
            b = self.blocks.get(h)
            if b is None or not b.kv_ready:
                break
            n += 1
        return n

    def lookup(self, tokens: Sequence[int]) -> Optional[Tuple[int, List[str]]]:
        """(block index to resume at, the chain up to it) or None."""
        if self.disabled:
            return None
        n_full = restore_blocks(len(tokens), self.block)
        if n_full == 0:
            return None
        hashes = self.chain(tokens, n_full)
        best = 0
        with self._lock:
            for i, h in enumerate(hashes, 1):
                b = self.blocks.get(h)
                if b is None or not b.kv_ready:
                    break
                if not self.needs_rec or b.rec_ready:
                    best = i
        if best == 0:
            self.misses += 1
            return None
        return best, hashes[:best]

    def cut_points(self, tokens: Sequence[int], start: int) -> List[int]:
        """Token positions past ``start`` where a prefill should end a chunk
        so a snapshot exists there: the prompt's last full block (the next
        session diverges right after it) and the end of what the store
        already knows (where this prompt diverged from an earlier one)."""
        if self.disabled or len(tokens) < self.min_prompt_tokens:
            return []
        n_full = restore_blocks(len(tokens), self.block)
        if n_full == 0:
            return []
        hashes = self.chain(tokens, n_full)
        with self._lock:
            known = self.known_blocks(hashes)
        points = {n_full * self.block, known * self.block}
        return sorted(p for p in points if p > start)

    def restore(self, make_cache: Callable[[], List[Any]], tokens: Sequence[int], i: int) -> Optional[List[Any]]:
        """Fresh cache objects positioned at ``i * block`` tokens of ``tokens``,
        or None (and the offending blocks dropped) if the files failed."""
        import mlx.core as mx

        hashes = self.chain(tokens, i)
        t0 = time.perf_counter()
        try:
            cache = make_cache()
            if len(cache) != len(self.layout):
                raise ValueError(f"cache has {len(cache)} layers, store {len(self.layout)}")
            files = [mx.load(self._kv_path(h), format="safetensors") for h in hashes]
            rec = (
                mx.load(self._rec_path(hashes[-1]), format="safetensors") if self.needs_rec else {}
            )
            arrays: List[Any] = []
            for layer, (c, kind) in enumerate(zip(cache, self.layout)):
                if kind == "kv":
                    ks = [f[f"k{layer}"] for f in files]
                    vs = [f[f"v{layer}"] for f in files]
                    k = ks[0] if len(ks) == 1 else mx.concatenate(ks, axis=2)
                    v = vs[0] if len(vs) == 1 else mx.concatenate(vs, axis=2)
                    c.state = (k, v)
                    arrays += [k, v]
                else:
                    st = list(c.state)
                    for slot in range(len(st)):
                        key = f"r{layer}.{slot}"
                        if key in rec:
                            st[slot] = rec[key]
                            arrays.append(rec[key])
                    c.state = st
            mx.eval(*arrays)
        except Exception as exc:  # noqa: BLE001 -- a bad file must not fail the request
            logger.warning(f"prefix disk: restore at block {i} failed ({exc}); dropping the chain")
            with self._lock:
                for h in hashes:
                    self._drop(h)
            self._queue.put(("manifest",))
            return None
        with self._lock:
            self._touch(hashes)
            self.hits += 1
            self.restored_tokens += i * self.block
            nbytes = sum(self.blocks[h].kv_bytes for h in hashes if h in self.blocks)
            nbytes += self.blocks[hashes[-1]].rec_bytes if hashes[-1] in self.blocks else 0
        self._queue.put(("manifest",))
        logger.info(
            f"prefix disk: restored {i * self.block} tokens ({i} blocks, {nbytes / 2**20:.0f} MB) "
            f"in {1e3 * (time.perf_counter() - t0):.0f} ms"
        )
        return cache

    # ------------------------------------------------------------ insert

    def observe(
        self,
        tokens: Sequence[int],
        cache: Sequence[Any],
        *,
        prompt_len: Optional[int] = None,
        restore_point: bool = False,
    ) -> None:
        """A cache positioned exactly at ``len(tokens)`` -- a chunk end during
        prefill, or a just-restored prefix. Writes whatever the store lacks
        for this chain: the KV of every missing block, and the recurrent
        snapshot at this block when the policy wants one here."""
        if self.disabled:
            return
        n = len(tokens)
        if n == 0 or n % self.block or (prompt_len or n) < self.min_prompt_tokens:
            return
        i = n // self.block
        try:
            self._observe(self.chain(tokens, i), i, cache, prompt_len, restore_point)
        except Exception as exc:  # noqa: BLE001 -- never take the scheduler down
            self._disable(f"snapshot at block {i} failed: {type(exc).__name__}: {exc}")

    def _observe(self, hashes: List[str], i: int, cache, prompt_len, restore_point: bool) -> None:
        import mlx.core as mx

        h = hashes[-1]
        n_full = restore_blocks(prompt_len, self.block) if prompt_len else None
        with self._lock:
            pre_existing = h in self.blocks and self.blocks[h].kv_known
            missing = [j for j in range(1, i + 1) if not (
                hashes[j - 1] in self.blocks and self.blocks[hashes[j - 1]].kv_known
            )]
            have_rec = h in self.blocks and self.blocks[h].rec_known
        rec_wanted = (
            restore_point
            or pre_existing
            or i % self.rec_every == 0
            or (n_full is not None and i == n_full)
        )
        kv_layers = [(l, c) for l, (c, kind) in enumerate(zip(cache, self.layout)) if kind == "kv"]
        rec_layers = [(l, c) for l, (c, kind) in enumerate(zip(cache, self.layout)) if kind == "rec"]

        # Everything this call will copy, dispatched asynchronously. The old
        # synchronous eval stalled the SCHEDULER at every 256-token boundary
        # for the GPU flush plus the host copies -- measured ~30% of served
        # prefill throughput on the 27B (79-86 vs 113-118 tok/s without the
        # tier). Now the previous boundary's snapshot is harvested here (its
        # arrays have long finished on the GPU, so eval + copy are cheap) and
        # this boundary's is dispatched with async_eval; at most one snapshot
        # is in flight, bounding the extra memory to one boundary's views.
        self._flush_snapshot()
        views: List[Any] = []
        kv_states = {}
        for l, c in kv_layers:
            k, v = c.state
            if k.ndim != 4 or k.shape[0] != 1 or k.shape[2] != i * self.block or getattr(c, "meta_state", ""):
                raise ValueError(f"layer {l}: unexpected KV state {k.shape}")
            kv_states[l] = (k, v)
        need_rec = self.needs_rec and not have_rec
        rec_states = {}
        if need_rec:
            for l, c in rec_layers:
                st = c.state
                if not isinstance(st, list):
                    raise ValueError(f"layer {l}: recurrent state is {type(st).__name__}")
                rec_states[l] = st
                views += [a for a in st if a is not None]
        for j in missing:
            for l, (k, v) in kv_states.items():
                lo, hi = (j - 1) * self.block, j * self.block
                views += [k[..., lo:hi, :], v[..., lo:hi, :]]

        def finish():
            jobs: List[tuple] = []
            for j in missing:
                tensors: Dict[str, Tuple[str, np.ndarray]] = {}
                for l, (k, v) in kv_states.items():
                    lo, hi = (j - 1) * self.block, j * self.block
                    tensors[f"k{l}"] = _to_host(k[..., lo:hi, :])
                    tensors[f"v{l}"] = _to_host(v[..., lo:hi, :])
                jobs.append(("kv", hashes[j - 1], j, tensors))
            rec_tensors: Optional[Dict[str, Tuple[str, np.ndarray]]] = None
            if need_rec:
                rec_tensors = {}
                for l, st in rec_states.items():
                    for slot, a in enumerate(st):
                        if a is not None:
                            rec_tensors[f"r{l}.{slot}"] = _to_host(a)
                if rec_wanted:
                    jobs.append(("rec", h, i, rec_tensors))
                else:
                    self._window_put(h, i, rec_tensors)

            with self._lock:
                self._touch([x for x in hashes if x in self.blocks])
                for job in jobs:
                    kind, bh, idx, tensors = job
                    nbytes = sum(arr.nbytes for _, arr in tensors.values())
                    if not self._make_room(nbytes, hashes):
                        self.skipped_writes += 1
                        continue
                    b = self.blocks.setdefault(bh, _Block(index=idx))
                    b.used = self._clock()
                    if kind == "kv":
                        b.kv_pending = True
                    else:
                        b.rec_pending = True
                    self._pending_bytes += nbytes
                    self._queue.put(job)

        if views:
            mx.async_eval(*views)
        self._snapshot_pending = (views, finish)

    def _flush_snapshot(self) -> None:
        """Harvest the previously dispatched snapshot, if any."""
        pending = getattr(self, "_snapshot_pending", None)
        if pending is None:
            return
        self._snapshot_pending = None
        views, finish = pending
        import mlx.core as mx

        try:
            if views:
                mx.eval(*views)
            finish()
        except Exception as exc:  # noqa: BLE001 -- never take the scheduler down
            self._disable(f"snapshot flush failed: {type(exc).__name__}: {exc}")

    def _window_put(self, h: str, i: int, tensors) -> None:
        if h in self._window:
            return
        self._window[h] = (i, tensors)
        self._window_order.append(h)
        while len(self._window_order) > WINDOW_SLOTS:
            old = self._window_order.pop(0)
            self._window.pop(old, None)

    def done(self, prompt_tokens: Sequence[int]) -> None:
        """The prompt's prefill is over: persist the deepest windowed
        recurrent snapshot on its chain (the batched path's last full block
        is a segment end the policy could not anticipate), forget the rest."""
        if self.disabled:
            return
        # The prompt's last boundary snapshot may still be in flight.
        self._flush_snapshot()
        if not self._window:
            return
        n_full = restore_blocks(len(prompt_tokens), self.block)
        hashes = self.chain(prompt_tokens, n_full)
        flushed = False
        for i in range(n_full, 0, -1):
            h = hashes[i - 1]
            entry = self._window.pop(h, None)
            if entry is None:
                continue
            self._window_order.remove(h)
            if flushed:
                continue
            idx, tensors = entry
            with self._lock:
                b = self.blocks.get(h)
                if b is None or not b.kv_known or b.rec_known:
                    continue
                nbytes = sum(arr.nbytes for _, arr in tensors.values())
                if not self._make_room(nbytes, hashes):
                    self.skipped_writes += 1
                    continue
                b.rec_pending = True
                self._pending_bytes += nbytes
                self._queue.put(("rec", h, idx, tensors))
            flushed = True

    # ------------------------------------------------------------ writer

    def _write_loop(self) -> None:
        dirty = False
        last_persist = 0.0
        while True:
            try:
                job = self._queue.get(timeout=2.0)
            except queue.Empty:
                job = ("idle",)
            if job is None:
                if dirty:
                    self._persist()
                return
            kind = job[0]
            if kind == "sync":
                if dirty:
                    self._persist()
                    dirty = False
                job[1].set()
                continue
            if kind == "manifest":
                dirty = True
            elif kind in ("kv", "rec"):
                self._write_job(job)
                dirty = True
            if dirty and (self._queue.empty() or time.monotonic() - last_persist > 5.0):
                self._persist()
                dirty = False
                last_persist = time.monotonic()

    def _write_job(self, job: tuple) -> None:
        kind, h, idx, tensors = job
        nbytes = sum(arr.nbytes for _, arr in tensors.values())
        path = self._kv_path(h) if kind == "kv" else self._rec_path(h)
        size = 0
        try:
            size = write_safetensors(
                path, tensors, {"block": str(idx), "tokens": str(idx * self.block), "kind": kind}
            )
        except Exception as exc:  # noqa: BLE001 -- disk full, permissions: skip this block
            logger.warning(f"prefix disk: writing {os.path.basename(path)} failed: {exc}")
            self._unlink(f"{path}.tmp")
        with self._lock:
            self._pending_bytes -= nbytes
            b = self.blocks.get(h)
            if b is None:
                self._unlink(path)
                return
            if kind == "kv":
                b.kv_pending = False
                if size:
                    b.kv_bytes, b.kv_ready = size, True
                    self.written_bytes += size
                else:
                    self._drop(h)
            else:
                b.rec_pending = False
                if size:
                    b.rec_bytes, b.rec_ready = size, True
                    self.written_bytes += size

    def _persist(self) -> None:
        with self._lock:
            blocks = {
                h: {"i": b.index, "kv": b.kv_bytes, "rec": b.rec_bytes, "used": round(b.used, 3)}
                for h, b in self.blocks.items()
                if b.kv_ready
            }
            stats = self._stats_locked()
        manifest = {
            "version": MANIFEST_VERSION,
            "block_tokens": self.block,
            "layout": ",".join(self.layout),
            "blocks": blocks,
        }
        for name, payload in (("manifest.json", manifest), ("stats.json", stats)):
            path = os.path.join(self.dir, name)
            try:
                with open(f"{path}.tmp", "w", encoding="utf-8") as f:
                    json.dump(payload, f)
                os.replace(f"{path}.tmp", path)
            except Exception as exc:  # noqa: BLE001 -- best effort
                logger.warning(f"prefix disk: writing {name} failed: {exc}")


# ----------------------------------------------------------------------------
# wiring


def budget_bytes(config: Any) -> int:
    raw = os.environ.get("MAXTOKEN_PREFIX_CACHE_DISK_GB")
    gb = float(raw) if raw not in (None, "") else float(
        getattr(config, "prefix_cache_disk_gb", DEFAULT_BUDGET_GB) or 0.0
    )
    return int(max(0.0, gb) * 2**30)


def root_dir(config: Any) -> str:
    return os.environ.get("MAXTOKEN_PREFIX_CACHE_DIR") or str(
        getattr(config, "prefix_cache_dir", None) or DEFAULT_DIR
    )


def open_for(config: Any, model: Any, model_dir: Optional[str]) -> Optional[DiskPrefixStore]:
    """The tier for this server, or None: budget 0, or a cache layout the
    tier does not store."""
    budget = budget_bytes(config)
    if budget <= 0:
        logger.info("prefix disk: off (--prefix-cache-disk-gb 0)")
        return None
    from mlx_lm.models.cache import make_prompt_cache

    layout = cache_layout(make_prompt_cache(model))
    if layout is None:
        logger.info("prefix disk: off (cache layout not storable)")
        return None
    key = model_key(str(getattr(config, "model_path", "model")), model_dir)
    try:
        store = DiskPrefixStore(root_dir(config), budget, key, layout)
    except Exception as exc:  # noqa: BLE001 -- unwritable directory: run without
        logger.warning(f"prefix disk: off ({exc})")
        return None
    if store.disabled:
        return None
    logger.info(
        f"prefix disk: on ({store.dir}, budget {budget / 2**30:.0f} GiB, "
        f"{'KV + recurrent' if store.needs_rec else 'KV'} snapshots)"
    )
    return store


def read_stats(root: Optional[str], model_path: str) -> Optional[Dict[str, Any]]:
    """The stats the worker's tier last persisted, for /admin/cache/status."""
    try:
        path = os.path.join(store_dir(root or DEFAULT_DIR, model_path), "stats.json")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001 -- no tier, no stats
        return None
