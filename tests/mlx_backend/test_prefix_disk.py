"""The prefix cache's disk tier: what survives a restart, and what it costs.

Blocks are addressed by a hash chain, so the tests check the chain names
prefixes exactly; a store reopened on the same directory resumes a prompt at
the deepest block that has KV *and* (on a hybrid layout) the recurrent state;
the snapshot policy writes the recurrent state where a resume is likely and
nowhere else; eviction drops a chain's tail before its head; and a damaged
file drops its chain instead of failing the request.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="the tier stores mx arrays")

from maxtoken.mlx_backend import prefix_disk as pd  # noqa: E402
from maxtoken.mlx_backend.prefix_disk import (  # noqa: E402
    DiskPrefixStore,
    chain_hashes,
    restore_blocks,
)

BLOCK = 8  # small blocks keep the arrays tiny; the policy is size-agnostic
HEADS, DIM = 2, 4


class FakeKV:
    """Trimmable (keys, values) cache with mlx-lm's KVCache state contract."""

    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    @property
    def state(self):
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    @state.setter
    def state(self, v):
        self.keys, self.values = v
        self.offset = self.keys.shape[2]

    @property
    def meta_state(self):
        return ""

    def is_trimmable(self):
        return True

    def trim(self, n):
        self.offset -= n


class FakeRec:
    """Non-trimmable recurrent cache: state is a list replaced slot by slot."""

    def __init__(self):
        self.cache = [None, None]

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


def kv_at(n, dtype=mx.float32):
    """Keys/values whose entry at position t is t (+ head, dim offsets)."""
    t = mx.arange(n, dtype=mx.float32)[None, None, :, None]
    h = mx.arange(HEADS, dtype=mx.float32)[None, :, None, None] * 0.25
    d = mx.arange(DIM, dtype=mx.float32)[None, None, None, :] * 0.0625
    k = mx.broadcast_to(t + h + d, (1, HEADS, n, DIM)).astype(dtype)
    return k, (k + 0.5).astype(dtype)


def rec_at(n):
    return [mx.full((1, 3), float(n)), mx.full((1, 2, 2), float(-n))]


def cache_at(n, layout=("kv", "rec"), dtype=mx.float32):
    cache = []
    for kind in layout:
        if kind == "kv":
            c = FakeKV()
            c.state = kv_at(n, dtype)
        else:
            c = FakeRec()
            c.state = rec_at(n)
        cache.append(c)
    return cache


def fresh(layout=("kv", "rec")):
    return [FakeKV() if k == "kv" else FakeRec() for k in layout]


def store(root, layout=("kv", "rec"), budget=1 << 30, clock=None, **kw):
    kw.setdefault("min_prompt_tokens", 16)
    return DiskPrefixStore(
        str(root), budget, "model", list(layout), block_tokens=BLOCK,
        clock=clock or (lambda: 1.0), **kw
    )


def toks(n, seed=1):
    return [seed * 1000 + i for i in range(n)]


def assert_positioned(cache, n, layout=("kv", "rec")):
    for c, kind in zip(cache, layout):
        if kind == "kv":
            k, v = c.state
            assert k.shape[2] == n == c.offset
            ek, ev = kv_at(n, k.dtype)
            assert mx.array_equal(k, ek) and mx.array_equal(v, ev)
        else:
            e = rec_at(n)
            assert all(mx.array_equal(a, b) for a, b in zip(c.state, e))


# ---------------------------------------------------------------- hashing


def test_chain_hashes_name_token_prefixes_exactly():
    t = toks(40)
    full = chain_hashes(t, BLOCK)
    assert len(full) == 5
    assert chain_hashes(t[:24], BLOCK) == full[:3]
    other = list(t)
    other[20] += 1  # block 3 differs -> hashes 3.. differ, 1-2 stay
    assert chain_hashes(other, BLOCK)[:2] == full[:2]
    assert chain_hashes(other, BLOCK)[2:] != full[2:]


def test_restore_blocks_leaves_one_token_to_process():
    assert restore_blocks(BLOCK * 3, BLOCK) == 2
    assert restore_blocks(BLOCK * 3 + 1, BLOCK) == 3
    assert restore_blocks(3, BLOCK) == 0


# ---------------------------------------------------------------- round trip


def test_a_reopened_store_resumes_at_the_deepest_complete_block(tmp_path):
    prompt = toks(41)  # 5 full blocks, one token to spare
    s = store(tmp_path)
    s.observe(prompt[:16], cache_at(16), prompt_len=len(prompt))  # windowed
    s.observe(prompt[:32], cache_at(32), prompt_len=len(prompt))  # 4th block: eager
    s.observe(prompt[:40], cache_at(40), prompt_len=len(prompt))  # last full block: eager
    s.flush()
    s.close()

    again = store(tmp_path)
    assert again.stats()["blocks"] == 5
    other = prompt[:40] + [7, 8, 9]  # shares all five blocks
    hit = again.lookup(other)
    assert hit is not None and hit[0] == 5
    cache = again.restore(fresh, other, 5)
    assert_positioned(cache, 40)
    assert again.stats()["hits"] == 1 and again.stats()["restored_tokens"] == 40

    # A prompt sharing four blocks resumes at the fourth (eager snapshot),
    # not the fifth; one sharing only two blocks finds KV but no recurrent
    # state there and is not served.
    hit = again.lookup(prompt[:32] + [1, 2, 3])
    assert hit is not None and hit[0] == 4
    assert again.lookup(prompt[:16] + [1, 2, 3]) is None
    again.close()


def test_bf16_kv_survives_the_uint16_detour(tmp_path):
    prompt = toks(17)
    s = store(tmp_path, layout=("kv",))
    s.observe(prompt[:16], cache_at(16, ("kv",), mx.bfloat16), prompt_len=len(prompt))
    s.flush()
    cache = s.restore(lambda: fresh(("kv",)), prompt, 2)
    k, _ = cache[0].state
    assert k.dtype == mx.bfloat16
    assert_positioned(cache, 16, ("kv",))
    s.close()


def test_kv_only_layouts_need_no_recurrent_snapshot(tmp_path):
    prompt = toks(100)
    s = store(tmp_path, layout=("kv",))
    s.observe(prompt[:16], cache_at(16, ("kv",)), prompt_len=len(prompt))
    s.flush()
    assert s.lookup(prompt)[0] == 2
    s.close()


# ---------------------------------------------------------------- policy


def test_the_last_windowed_snapshot_is_written_when_the_prompt_is_done(tmp_path):
    prompt = toks(100)
    s = store(tmp_path)
    s.observe(prompt[:8], cache_at(8), prompt_len=len(prompt))
    s.observe(prompt[:16], cache_at(16), prompt_len=len(prompt))
    s.flush()
    assert s.lookup(prompt) is None, "KV alone cannot resume a hybrid cache"
    s.done(prompt)
    s.flush()
    assert s.lookup(prompt)[0] == 2, "the deepest windowed block got its snapshot"
    assert s.stats()["rec_blocks"] == 1, "and only that one"
    s.close()


def test_a_prompt_diverging_from_a_stored_one_marks_the_fork(tmp_path):
    a = toks(25)  # 3 full blocks
    s = store(tmp_path)
    for n in (8, 16, 24):
        s.observe(a[:n], cache_at(n), prompt_len=len(a))
    s.flush()
    b = a[:16] + toks(25, seed=2)  # shares two blocks, then differs
    assert s.cut_points(b, 0) == [16, 40]
    assert s.cut_points(b, 16) == [40]
    # Passing through block 2 (already stored) is the fork: snapshot there.
    s.observe(b[:16], cache_at(16), prompt_len=len(b))
    s.flush()
    assert s.lookup(a[:16] + [5, 6]) == (2, chain_hashes(a[:16], BLOCK))
    s.close()


def test_a_resume_point_is_kept(tmp_path):
    prompt = toks(100)
    s = store(tmp_path)
    s.observe(prompt[:8], cache_at(8), prompt_len=len(prompt), restore_point=True)
    s.flush()
    assert s.lookup(prompt)[0] == 1
    s.close()


def test_short_prompts_stay_in_ram(tmp_path):
    s = store(tmp_path, min_prompt_tokens=64)
    s.observe(toks(16), cache_at(16), prompt_len=20)
    s.flush()
    assert s.stats()["blocks"] == 0 and s.stats()["bytes"] == 0
    s.close()


# ---------------------------------------------------------------- budget


def test_eviction_drops_a_chains_tail_before_its_head(tmp_path):
    now = [1.0]
    s = store(tmp_path, layout=("kv",), budget=1 << 30, clock=lambda: now[0])
    a, b = toks(17, seed=1), toks(17, seed=2)
    s.observe(a[:16], cache_at(16, ("kv",)), prompt_len=len(a))
    s.flush()
    per_block = s.stats()["bytes"] // 2
    s.max_bytes = 3 * per_block  # room for three blocks in total
    now[0] = 2.0
    s.observe(b[:16], cache_at(16, ("kv",)), prompt_len=len(b))
    s.flush()
    assert s.stats()["blocks"] == 3
    assert s.lookup(a)[0] == 1, "a lost its tail, still resumes at its head"
    assert s.lookup(b)[0] == 2
    assert not os.path.exists(s._kv_path(chain_hashes(a, BLOCK)[1]))
    s.close()


def test_a_chain_larger_than_the_budget_is_skipped_not_half_written(tmp_path):
    s = store(tmp_path, layout=("kv",), budget=1)
    s.observe(toks(16), cache_at(16, ("kv",)), prompt_len=17)
    s.flush()
    assert s.stats()["blocks"] == 0 and s.stats()["skipped_writes"] == 2
    s.close()


# ---------------------------------------------------------------- damage


def test_a_damaged_file_drops_its_chain_instead_of_failing(tmp_path):
    prompt = toks(17)
    s = store(tmp_path, layout=("kv",))
    s.observe(prompt[:16], cache_at(16, ("kv",)), prompt_len=len(prompt))
    s.flush()
    h = chain_hashes(prompt, BLOCK)[1]
    with open(s._kv_path(h), "wb") as f:
        f.write(b"garbage")
    assert s.restore(lambda: fresh(("kv",)), prompt, 2) is None
    assert s.lookup(prompt) is None
    assert s.stats()["blocks"] == 0
    s.close()


def test_a_layout_change_discards_the_old_store(tmp_path):
    s = store(tmp_path)
    s.observe(toks(16), cache_at(16), prompt_len=17)
    s.flush()
    s.close()
    again = store(tmp_path, layout=("kv",))
    assert again.stats()["blocks"] == 0
    assert not [f for f in os.listdir(again.dir) if f.endswith((".kv", ".rec"))]
    again.close()


def test_a_second_process_on_the_same_store_runs_without_it(tmp_path):
    first = store(tmp_path)
    second = store(tmp_path)
    assert second.disabled and not first.disabled
    assert second.lookup(toks(40)) is None
    first.close()


def test_stats_are_published_for_the_server(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "x"}')
    key = pd.model_key(str(model_dir), str(model_dir))
    s = DiskPrefixStore(str(tmp_path / "root"), 1 << 30, key, ["kv"], block_tokens=BLOCK,
                        min_prompt_tokens=16)
    s.observe(toks(16), cache_at(16, ("kv",)), prompt_len=17)
    s.flush()
    published = pd.read_stats(str(tmp_path / "root"), str(model_dir))
    assert published["blocks"] == 2 and published["budget_bytes"] == 1 << 30
    assert pd.read_stats(str(tmp_path / "elsewhere"), str(model_dir)) is None
    s.close()


def test_model_key_tracks_the_weights():
    assert pd.model_key("a/b", None) == pd.model_key("a/b", None)
    assert pd.model_key("a/b", None) != pd.model_key("a/c", None)
    assert pd.model_key("a/b", None).startswith("a--b-")


# ---------------------------------------------------------------- wiring


def test_the_scheduler_takes_the_deeper_tier(tmp_path):
    from maxtoken.mlx_backend.prefix_cache import PrefixStore
    from maxtoken.mlx_backend.worker import MlxScheduler

    sched = MlxScheduler.__new__(MlxScheduler)
    sched.model = SimpleNamespace(make_cache=lambda: fresh(("kv",)))
    sched.prefix_store = PrefixStore(1 << 30)
    restored = {}
    sched.prefix_disk = SimpleNamespace(
        block=BLOCK,
        lookup=lambda ids: (5, None),
        restore=lambda make, ids, i: restored.setdefault("cache", cache_at(40, ("kv",))),
        observe=lambda *a, **k: None,
    )
    prompt = toks(80)
    cache, n = sched._lookup_prefix(prompt)
    assert n == 40 and cache is restored["cache"]
    assert sched.prefix_store.lookup(prompt)[1] == 40, "RAM learned the disk's entry"

    # RAM deeper than disk: RAM wins, and the disk is told about the resume.
    seen = {}
    sched.prefix_store = PrefixStore(1 << 30)
    sched.prefix_store.insert(prompt[:64], cache_at(64, ("kv",)))
    sched.prefix_disk.observe = lambda ids, c, **kw: seen.update(n=len(ids), **kw)
    cache, n = sched._lookup_prefix(prompt)
    assert n == 64 and seen == {"n": 64, "prompt_len": 80, "restore_point": True}


def test_the_prefill_ends_a_chunk_where_the_tier_asks(monkeypatch):
    from maxtoken.mlx_backend.worker import MlxScheduler

    widths = []

    class Model:
        def __call__(self, tokens, cache=None):
            widths.append(int(tokens.shape[1]))
            return mx.zeros((1, tokens.shape[1], 4))

    sched = MlxScheduler.__new__(MlxScheduler)
    sched._mx = mx
    sched.model = Model()
    sched.prefix_store = None
    observed = []
    sched.prefix_disk = SimpleNamespace(
        cut_points=lambda ids, start: [768],
        observe=lambda ids, c, **kw: observed.append(len(ids)),
    )
    ids = list(range(1200))
    for _ in sched._prefill_chunks(None, ids, 0):
        pass
    assert widths == [768, 431], "one cut at 768, then the rest"
    assert observed == [768], "and the tier saw the state there"
