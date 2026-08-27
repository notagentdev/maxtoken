"""PrefixStore logic against fake cache objects (no mlx needed): matching rules
for trimmable vs boundary-only entries, COW isolation of stored state, LRU
eviction, and the exact-restore contract."""

from typing import Any, List

import numpy as np
import pytest

from maxtoken.mlx_backend.prefix_cache import (
    MIN_MATCH_TOKENS,
    PrefixStore,
    _common_prefix_len,
)


class FakeKV:
    """Trimmable cache: state is (keys, values) 'arrays' (np here)."""

    def __init__(self):
        self.keys = np.zeros((1, 0))
        self.offset = 0

    @property
    def state(self):
        return (self.keys, self.keys)

    @state.setter
    def state(self, v):
        self.keys = v[0]
        self.offset = self.keys.shape[-1]

    @property
    def meta_state(self):
        return ""

    def is_trimmable(self):
        return True

    def trim(self, n):
        self.offset -= n


class FakeRecurrent:
    """Non-trimmable cache: state is a LIST whose slots get replaced in place."""

    def __init__(self):
        self.cache: List[Any] = [np.array([0.0]), None]

    @property
    def state(self):
        return self.cache

    @state.setter
    def state(self, v):
        self.cache = v

    @property
    def meta_state(self):
        return ""

    def is_trimmable(self):
        return False

    def trim(self, n):  # pragma: no cover -- must never be called for these
        raise AssertionError("trim on non-trimmable cache")


def toks(n, offset=0):
    return list(range(offset, offset + n))


def kv_at(n):
    c = FakeKV()
    c.keys = np.zeros((1, n))
    c.offset = n
    return c


def test_trimmable_entry_serves_any_shared_prefix():
    store = PrefixStore(max_bytes=1 << 30)
    store.insert(toks(100), [kv_at(100)])
    hit = store.lookup(toks(60) + [999])
    assert hit is not None
    entry, n = hit
    assert n == 60


def test_non_trimmable_entry_only_matches_at_its_boundary():
    store = PrefixStore(max_bytes=1 << 30)
    store.insert(toks(64), [FakeRecurrent()])
    # prompt shares only 50 tokens -> no usable restore point
    assert store.lookup(toks(50) + [999]) is None
    # prompt fully contains the entry -> restore exactly at 64
    hit = store.lookup(toks(100))
    assert hit is not None and hit[1] == 64


def test_full_prompt_hit_is_capped_for_trimmable_only():
    store = PrefixStore(max_bytes=1 << 30)
    store.insert(toks(100), [kv_at(100)])
    hit = store.lookup(toks(100))  # identical prompt: one token must remain
    assert hit is not None and hit[1] == 99

    store2 = PrefixStore(max_bytes=1 << 30)
    store2.insert(toks(100), [FakeRecurrent()])
    assert store2.lookup(toks(100)) is None  # can't trim the overshoot


def test_restore_trims_overshoot_and_isolates_lists():
    pytest.importorskip("mlx_lm", reason="restore goes through make_prompt_cache")
    store = PrefixStore(max_bytes=1 << 30)
    rec = FakeRecurrent()
    rec.cache[0] = np.array([7.0])
    stored_list_marker = rec.cache
    store.insert(toks(64), [rec])
    entry, n = store.lookup(toks(64) + [999])
    assert n == 64

    fresh = [FakeRecurrent()]

    class Model:
        def make_cache(self):  # make_prompt_cache(model) delegates to this
            return fresh

    cache = store.restore(Model(), entry, 64)
    assert cache is fresh
    assert cache[0].cache[0][0] == 7.0  # state arrived
    # restored cache got a COPY of the stored list: replacing its slots must not
    # corrupt the stored snapshot
    cache[0].cache[0] = np.array([123.0])
    assert entry.states[0][0][0][0] == 7.0
    assert entry.states[0][0] is not stored_list_marker  # insert copied too


def test_lru_eviction_by_bytes_keeps_newest():
    store = PrefixStore(max_bytes=3 * 100 * 8)  # room for ~3 entries of kv_at(100)
    for i in range(5):
        store.insert(toks(100, offset=i * 1000), [kv_at(100)])
    assert len(store.entries) <= 4
    # the newest insert always survives
    assert any(
        np.array_equal(e.tokens, np.asarray(toks(100, offset=4000), dtype=np.int32))
        for e in store.entries
    )


def test_min_match_threshold():
    store = PrefixStore(max_bytes=1 << 30)
    store.insert(toks(100), [kv_at(100)])
    short = toks(MIN_MATCH_TOKENS - 1) + [999, 998]
    assert store.lookup(short) is None


def test_common_prefix_len():
    a = np.array([1, 2, 3, 4], dtype=np.int32)
    assert _common_prefix_len(a, np.array([1, 2, 9], dtype=np.int32)) == 2
    assert _common_prefix_len(a, a) == 4
    assert _common_prefix_len(a, np.array([], dtype=np.int32)) == 0


def test_same_tokens_supersede():
    store = PrefixStore(max_bytes=1 << 30)
    store.insert(toks(100), [kv_at(100)])
    store.insert(toks(100), [kv_at(100)])
    assert len(store.entries) == 1
