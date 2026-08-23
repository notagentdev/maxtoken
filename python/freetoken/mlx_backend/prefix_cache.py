"""Cross-request prefix cache for the MLX backend.

A request whose token prompt shares a prefix with earlier work skips recomputing
that prefix: the worker snapshots generation caches (KV *and* recurrent state) at
token boundaries and restores the longest matching snapshot for a new request,
prefilling only the remainder. This is FreeToken's prefix reuse on MLX — and it
covers the case that is broken upstream (mlx-lm#980): hybrid models (GDN/SSM,
sliding window), whose recurrent states cannot be trimmed to arbitrary token
positions. The fix is the same idea LM Studio ships for its engine: keep
snapshots at *fixed boundaries* and restore exactly there; for purely trimmable
caches (full attention) any prefix length works via trim-on-restore.

Memory discipline: snapshots store *references* to the live cache's arrays.
mx arrays are immutable, so the first in-place-style write after a snapshot
copies (copy-on-write) and the store's view stays intact — a snapshot costs one
deferred KV-buffer copy, not an eager clone. Recurrent-state lists are shallow-
copied on both insert and restore because their entries are replaced in place.

The store is plain Python over the caches' ``state``/``meta_state`` accessors
(the same surface ``save_prompt_cache`` uses), so every mlx-lm cache type works.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np

from freetoken.utils import init_logger

logger = init_logger(__name__)

# A hit below this many tokens is not worth the restore bookkeeping.
MIN_MATCH_TOKENS = 32
# During prefill, snapshots are taken each time the processed-token count
# crosses a multiple of this (hybrid models can only restore AT a snapshot).
# 256 matches LM Studio's engine: fine enough that reasoning models — whose
# chat templates strip/rewrite previous turns, so end-of-request snapshots
# rarely match — still get boundary hits from re-rendered conversations.
BOUNDARY_TOKENS = 256


def _copy_state(state: Any) -> Any:
    """Isolate mutable containers; the arrays themselves are shared (COW)."""
    if isinstance(state, list):
        return list(state)
    return state


def _state_nbytes(state: Any) -> int:
    total = 0
    stack = [state]
    while stack:
        s = stack.pop()
        if isinstance(s, (list, tuple)):
            stack.extend(s)
        elif hasattr(s, "nbytes"):
            total += s.nbytes
    return total


def _common_prefix_len(a: np.ndarray, b: np.ndarray) -> int:
    m = min(len(a), len(b))
    if m == 0:
        return 0
    neq = a[:m] != b[:m]
    return int(np.argmax(neq)) if neq.any() else m


@dataclass
class _Entry:
    tokens: np.ndarray  # int32, the exact tokens the states cover
    states: List[Tuple[Any, Any]]  # per layer: (state, meta_state)
    trimmable: bool
    nbytes: int
    last_used: float = field(default_factory=time.monotonic)


class PrefixStore:
    """LRU over cache snapshots, matched by longest shared token prefix."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self.entries: List[_Entry] = []
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------------ insert

    def snapshot(self, cache: List[Any]) -> List[Tuple[Any, Any]]:
        return [(_copy_state(c.state), c.meta_state) for c in cache]

    def insert(self, tokens: List[int], cache: List[Any]) -> None:
        if len(tokens) < MIN_MATCH_TOKENS:
            return
        toks = np.asarray(tokens, dtype=np.int32)
        states = self.snapshot(cache)
        entry = _Entry(
            tokens=toks,
            states=states,
            trimmable=all(c.is_trimmable() for c in cache),
            nbytes=sum(_state_nbytes(s) for s, _ in states),
        )
        # An entry for the exact same tokens is superseded, not duplicated.
        self.entries = [e for e in self.entries if not np.array_equal(e.tokens, toks)]
        self.entries.append(entry)
        self._evict()

    def _evict(self) -> None:
        total = sum(e.nbytes for e in self.entries)
        while total > self.max_bytes and len(self.entries) > 1:
            victim = min(self.entries[:-1], key=lambda e: e.last_used)
            self.entries.remove(victim)
            total -= victim.nbytes

    # ------------------------------------------------------------------ lookup

    def lookup(self, tokens: List[int]) -> Optional[Tuple[_Entry, int]]:
        """(entry, usable_prefix_tokens) for the best match, or None.

        At least one prompt token must remain to process, so the usable length
        is capped at len(tokens) - 1. Non-trimmable entries are only usable when
        their WHOLE token sequence is a prefix of the prompt (restore happens
        exactly at the snapshot); trimmable ones can serve any shared prefix.
        """
        toks = np.asarray(tokens, dtype=np.int32)
        cap = len(toks) - 1
        best: Tuple[int, Optional[_Entry]] = (0, None)
        for e in self.entries:
            common = _common_prefix_len(e.tokens, toks)
            if common == len(e.tokens):
                usable = min(common, cap) if e.trimmable else (
                    common if common <= cap else 0
                )
            elif e.trimmable:
                usable = min(common, cap)
            else:
                usable = 0
            if usable > best[0]:
                best = (usable, e)
        usable, entry = best
        if entry is None or usable < MIN_MATCH_TOKENS:
            self.misses += 1
            return None
        entry.last_used = time.monotonic()
        self.hits += 1
        return entry, usable

    def restore(self, model, entry: _Entry, n_tokens: int) -> List[Any]:
        """Fresh cache objects positioned at ``n_tokens`` of the entry's tokens."""
        from mlx_lm.models.cache import make_prompt_cache

        cache = make_prompt_cache(model)
        for c, (state, meta) in zip(cache, entry.states, strict=True):
            c.state = _copy_state(state)
            if meta:
                c.meta_state = meta
        overshoot = len(entry.tokens) - n_tokens
        if overshoot > 0:
            for c in cache:
                c.trim(overshoot)
        return cache
