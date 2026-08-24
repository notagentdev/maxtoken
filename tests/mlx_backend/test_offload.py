"""Expert-offload building blocks: LRU bookkeeping and safetensors byte-range
reads. The LRU and index tests run anywhere; ExpertStore fetch tests need mlx
(Apple silicon) and a crafted two-layout checkpoint written into tmp_path."""

import json
import struct

import numpy as np
import pytest

from freetoken.mlx_backend.offload import LruTracker, SafetensorsIndex


# ---------------------------------------------------------------- LruTracker

def test_lru_hits_and_misses():
    lru = LruTracker(2)
    hits, misses = lru.lookup([3, 5])
    assert hits == {} and misses == [3, 5]
    assert lru.assign(3) != lru.assign(5)
    hits, misses = lru.lookup([3, 5])
    assert set(hits) == {3, 5} and misses == []


def test_lru_evicts_least_recently_used():
    lru = LruTracker(2)
    s3, s5 = lru.assign(3), lru.assign(5)
    lru.lookup([3])  # 3 is now more recent than 5
    s7 = lru.assign(7)  # evicts 5, not 3
    assert s7 == s5
    hits, misses = lru.lookup([3, 5, 7])
    assert set(hits) == {3, 7} and misses == [5]


def test_lru_most_recent_order_and_reset():
    lru = LruTracker(3)
    for e in (1, 2, 3):
        lru.assign(e)
    lru.lookup([1])
    assert [e for e, _ in lru.most_recent(2)] == [1, 3]
    lru.reset(1)
    assert lru.lookup([1]) == ({}, [1])
    assert lru.num_slots == 1


# ---------------------------------------------------------- safetensors index

def write_safetensors(path, tensors):
    """Minimal safetensors writer: {name: np.ndarray} with C-contiguous data."""
    dtype_names = {np.dtype(np.uint32): "U32", np.dtype(np.float16): "F16"}
    header = {}
    offset = 0
    payload = b""
    for name, arr in tensors.items():
        data = arr.tobytes()
        header[name] = {
            "dtype": dtype_names[arr.dtype],
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
        payload += data
    blob = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(payload)


def test_index_reports_absolute_offsets(tmp_path):
    a = np.arange(12, dtype=np.uint32).reshape(3, 4)
    b = np.arange(6, dtype=np.float16).reshape(2, 3)
    write_safetensors(tmp_path / "m.safetensors", {"a": a, "b": b})
    ix = SafetensorsIndex(str(tmp_path))
    assert ix["a"].shape == (3, 4) and ix["a"].dtype == "U32"
    with open(ix["a"].shard, "rb") as f:
        f.seek(ix["b"].start)
        raw = f.read(ix["b"].end - ix["b"].start)
    assert np.frombuffer(raw, dtype=np.float16).reshape(2, 3).tolist() == b.tolist()


# ------------------------------------------------------- ExpertStore (mlx only)

mlx = pytest.importorskip("mlx.core", reason="ExpertStore fetch needs mlx")


def _fake_glu_tensors(rng, num_experts, stacked, prefix):
    """The 9 (proj x part) tensors of one switch-GLU, in either disk layout."""
    shapes = {"weight": (4, 2), "scales": (4, 1), "biases": (4, 1)}
    dtypes = {"weight": np.uint32, "scales": np.float16, "biases": np.float16}
    tensors = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for part, shape in shapes.items():
            full = (rng.random((num_experts, *shape)) * 100).astype(dtypes[part])
            if stacked:
                tensors[f"{prefix}.switch_mlp.{proj}.{part}"] = full
            else:
                for e in range(num_experts):
                    tensors[f"{prefix}.experts.{e}.{proj}.{part}"] = full[e]
    return tensors


@pytest.mark.parametrize("stacked", [True, False], ids=["stacked", "per-expert"])
def test_expert_store_fetch_matches_disk(tmp_path, stacked):
    from concurrent.futures import ThreadPoolExecutor

    from freetoken.mlx_backend.offload import ExpertStore

    rng = np.random.default_rng(0)
    tensors = _fake_glu_tensors(rng, num_experts=5, stacked=stacked, prefix="model.mlp")
    write_safetensors(tmp_path / "m.safetensors", tensors)
    ix = SafetensorsIndex(str(tmp_path))
    store = ExpertStore(ix, "model.mlp.switch_mlp", ThreadPoolExecutor(2))
    assert store.stacked is stacked
    assert store.num_experts == 5

    fetched = store.fetch([3, 1])
    key = (
        "model.mlp.switch_mlp.gate_proj.weight"
        if stacked
        else "model.mlp.experts.3.gate_proj.weight"
    )
    want = tensors[key][3] if stacked else tensors[key]
    got = np.array(fetched[0][0])
    assert got.tolist() == want.tolist()

    full = store.load_full_lazy()
    mlx.eval(*full)
    assert full[0].shape == (5, 4, 2)
    got_full = np.array(full[0][3])
    assert got_full.tolist() == want.tolist()


# ------------------------------------------------ banked short-chunk serving


def _quantized_glu_dir(tmp_path, num_experts, d=32, h=32, group=32):
    """A real 4-bit quantized switch-GLU checkpoint dir gather_qmm can serve."""
    import json as _json

    mlx.random.seed(7)
    tensors = {}
    for proj, shape in (
        ("gate_proj", (h, d)),
        ("up_proj", (h, d)),
        ("down_proj", (d, h)),
    ):
        ws, ss, bs = [], [], []
        for _ in range(num_experts):
            w = mlx.random.normal(shape).astype(mlx.float16)
            wq, sc, bi = mlx.quantize(w, group_size=group, bits=4)
            ws.append(wq)
            ss.append(sc)
            bs.append(bi)
        tensors[f"model.mlp.switch_mlp.{proj}.weight"] = np.array(
            mlx.stack(ws), copy=False
        )
        tensors[f"model.mlp.switch_mlp.{proj}.scales"] = np.array(
            mlx.stack(ss).astype(mlx.float16), copy=False
        )
        tensors[f"model.mlp.switch_mlp.{proj}.biases"] = np.array(
            mlx.stack(bs).astype(mlx.float16), copy=False
        )
    write_safetensors(tmp_path / "m.safetensors", tensors)
    (tmp_path / "config.json").write_text(
        _json.dumps({"quantization": {"group_size": group, "bits": 4}})
    )
    return d


def _build_glu(tmp_path, num_experts=8, slots=3):
    from concurrent.futures import ThreadPoolExecutor

    from freetoken.mlx_backend.offload import (
        ExpertStore,
        OffloadState,
        OffloadSwitchGLU,
        SlotCache,
    )

    d = _quantized_glu_dir(tmp_path, num_experts)
    ix = SafetensorsIndex(str(tmp_path))
    store = ExpertStore(ix, "model.mlp.switch_mlp", ThreadPoolExecutor(2))
    state = OffloadState(32, 4)
    cache = SlotCache(store, slots)
    activation = lambda up, gate: up * mlx.sigmoid(gate)  # noqa: E731
    glu = OffloadSwitchGLU(store, cache, activation, state)
    state.glus.append(glu)
    return glu, cache, d


def test_banked_serving_matches_full_bank(tmp_path):
    """A short chunk routing more experts than the cache has slots must produce
    exactly what serving from the fully materialized layer produces."""
    glu, cache, d = _build_glu(tmp_path, num_experts=8, slots=3)
    x = mlx.random.normal((1, 4, d)).astype(mlx.float16)
    # 4 tokens x top-2, 7 unique experts > 3 slots -> banked path
    inds = mlx.array([[[0, 5], [3, 6], [1, 5], [7, 2]]], dtype=mlx.uint32)
    got = glu(x, inds)

    full = glu.store.load_full_lazy()
    want = glu._run(
        x, inds,
        ((full[0], full[1], full[2]), (full[3], full[4], full[5]),
         (full[6], full[7], full[8])),
    )
    assert np.array_equal(np.array(got), np.array(want))


def test_banked_serving_admits_hottest_and_reuses_slots(tmp_path):
    glu, cache, d = _build_glu(tmp_path, num_experts=8, slots=3)
    x = mlx.random.normal((1, 4, d)).astype(mlx.float16)
    # expert 5 is routed twice -> hottest -> must be admitted
    inds = mlx.array([[[0, 5], [3, 6], [1, 5], [7, 2]]], dtype=mlx.uint32)
    glu(x, inds)
    assert 5 in cache.lru.slot_of
    misses_first = cache.misses
    # a second identical chunk serves the admitted experts from slots
    glu(x, inds)
    assert cache.misses < misses_first * 2
    assert cache.hits > 0


def test_long_chunk_still_streams(tmp_path, monkeypatch):
    """Chunks beyond the bank-token gate keep the full-layer streaming path."""
    import freetoken.mlx_backend.offload as off

    glu, cache, d = _build_glu(tmp_path, num_experts=8, slots=3)
    monkeypatch.setattr(off, "_BANK_TOKENS", 2)
    called = {}
    orig = glu._forward_streamed

    def spy(x, indices, np_inds):
        called["streamed"] = True
        return orig(x, indices, np_inds)

    glu._forward_streamed = spy
    x = mlx.random.normal((1, 4, d)).astype(mlx.float16)
    inds = mlx.array([[[0, 5], [3, 6], [1, 5], [7, 2]]], dtype=mlx.uint32)
    glu(x, inds)
    assert called.get("streamed")


# --------------------------------------- rebalance + cross-layer read-ahead


def _build_state(tmp_path, num_layers=3, num_experts=8, slots=4):
    from concurrent.futures import ThreadPoolExecutor

    from freetoken.mlx_backend.offload import (
        ExpertStore,
        OffloadState,
        OffloadSwitchGLU,
        SlotCache,
    )

    d = _quantized_glu_dir(tmp_path, num_experts)
    ix = SafetensorsIndex(str(tmp_path))
    state = OffloadState(32, 4)
    pool = ThreadPoolExecutor(2)
    activation = lambda up, gate: up * mlx.sigmoid(gate)  # noqa: E731
    for _ in range(num_layers):
        store = ExpertStore(ix, "model.mlp.switch_mlp", pool)
        glu = OffloadSwitchGLU(store, SlotCache(store, slots), activation, state)
        state.glus.append(glu)
    return state, d


def test_rebalance_moves_slots_to_miss_pressure(tmp_path):
    state, _ = _build_state(tmp_path, num_layers=3, num_experts=8, slots=4)
    assert state.rebalance(floor=2) is False  # first call only sets the baseline
    state.glus[0].cache.misses += 10
    state.glus[1].cache.misses += 0
    state.glus[2].cache.misses += 2
    assert state.rebalance(floor=2) is True
    slots = [g.cache.num_slots for g in state.glus]
    assert sum(slots) == 12  # total budget preserved
    assert min(slots) >= 2  # floor respected
    assert slots[0] > slots[1]  # pressure got the spread
    # quiet window -> no further movement
    assert state.rebalance(floor=2) is False


def test_rebalance_caps_at_num_experts(tmp_path):
    state, _ = _build_state(tmp_path, num_layers=2, num_experts=8, slots=7)
    state.rebalance(floor=2)
    state.glus[0].cache.misses += 100
    state.rebalance(floor=2)
    assert all(g.cache.num_slots <= 8 for g in state.glus)


def test_read_ahead_prefetches_next_layer(tmp_path):
    state, d = _build_state(tmp_path, num_layers=2, num_experts=8, slots=4)
    g0, g1 = state.glus
    g0.next_glu = g1

    def fake_gate(x):
        # favor experts 6 and 7 for the next layer
        scores = np.zeros((1, 1, 8), dtype=np.float16)
        scores[..., 6] = 5.0
        scores[..., 7] = 4.0
        return mlx.array(scores)

    g0.next_gate = fake_gate
    x = mlx.random.normal((1, 3, d)).astype(mlx.float16)
    # 6 uniq > 4 slots, 3 tokens <= bank gate: the banked path fires read-ahead
    inds = mlx.array([[[0, 1], [2, 3], [4, 5]]], dtype=mlx.uint32)
    g0(x, inds)
    inflight = set(g1.cache._inflight)
    assert 6 in inflight and 7 in inflight
    # ... and the prefetched bytes satisfy the next layer's install
    misses_before = g1.cache.misses
    g1(x, mlx.array([[[6, 7]]], dtype=mlx.uint32))
    assert g1.cache.misses == misses_before + 2
    assert 6 in g1.cache.lru.slot_of and 7 in g1.cache.lru.slot_of
