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
