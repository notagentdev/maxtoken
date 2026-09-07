"""Load a Niwaki qwen4_exp checkpoint (mlx-vlm >= 0.7.0rc0 + the checkpoint's
own ``niwaki_flash_load.py``) with MaxToken's stores attached BEFORE anything
is evaluated:

  * the PLE n-gram table stays on disk — a numpy memmap over the safetensors
    PLE shards; rows are gathered on the host (the stock ShardedEmbedding
    already syncs there) and dequantized on the device: 16 rows of 180 B per
    token, most of the 13 GiB table never gets touched;
  * the routed experts (24 layers x 512 x 1.44 MiB = 17.3 GiB) come from the
    FTW-MLX zero-copy mapped store, or from the expert slot cache under a hard
    budget when ``cache_rate`` is given.

Measured 2026-09-07 on a 32 GB M1 Max (mapped store, narrow command buffers):
decode 16.7-17.5 tok/s, prefill 277 tok/s warm, ~3.5 GiB truly resident.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_NP = {"U32": np.uint32, "F16": np.float16, "BF16": np.uint16, "F32": np.float32}


class MemmapPairedRows:
    """One PLE shard in the paired-row 2-bit layout, read straight from the shard file."""

    def __init__(self, index, prefix, *, group_size, bits, pair, dims):
        self.group_size, self.bits, self.pair, self.dims = group_size, bits, pair, dims
        self.parts = {}
        for part in ("weight", "scales", "biases"):
            e = index[f"{prefix}.{part}"]
            mm = np.memmap(e.shard, dtype=_NP[e.dtype], mode="r", offset=e.start, shape=e.shape)
            self.parts[part] = (mm, e.dtype)
        self.rows = self.parts["weight"][0].shape[0] * pair

    def gather(self, local: np.ndarray):
        import mlx.core as mx

        packed, piece = local // self.pair, local % self.pair
        uniq, inv = np.unique(packed, return_inverse=True)
        arrs = []
        for part in ("weight", "scales", "biases"):
            mm, dt = self.parts[part]
            a = mx.array(np.ascontiguousarray(mm[uniq]))
            if dt == "BF16":
                a = a.view(mx.bfloat16)
            arrs.append(a)
        deq = mx.dequantize(arrs[0], arrs[1], arrs[2], group_size=self.group_size, bits=self.bits)
        rows = deq[mx.array(inv.astype(np.int32))]
        cols = mx.array(piece.astype(np.int32))[:, None] * self.dims + mx.arange(self.dims)[None, :]
        return mx.take_along_axis(rows, cols, axis=1)


class MemmapShardedEmbedding:
    """Drop-in for mlx-vlm's ShardedEmbedding: the same host sync, rows from disk."""

    def __init__(self, shards, offsets, dims):
        self.shards, self.offsets, self.dims = shards, np.asarray(offsets), dims
        self.lookups = 0

    def __call__(self, indices):
        import mlx.core as mx

        flat = indices.reshape(-1)
        mx.eval(flat)
        host = np.array(flat.tolist(), dtype=np.int64)
        self.lookups += host.size
        if host.size == 0:
            return mx.zeros((*indices.shape, self.dims), dtype=mx.bfloat16)
        shard_ids = np.searchsorted(self.offsets, host, side="right") - 1
        pieces = []
        for sid in np.unique(shard_ids):
            pos = np.nonzero(shard_ids == sid)[0]
            pieces.append((pos, self.shards[int(sid)].gather(host[pos] - self.offsets[sid])))
        result = mx.zeros((host.size, self.dims), dtype=pieces[0][1].dtype)
        for pos, vals in pieces:
            result = result.at[mx.array(pos.astype(np.int32))].add(vals)
        return result.reshape(*indices.shape, self.dims)


def install_memmap_ple(model, model_dir: str, index, log=print) -> int:
    from maxtoken.mlx_backend.offload import _iter_modules

    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    tc = cfg.get("text_config", cfg)
    pair = int(tc.get("niwaki_ple_pair", 1))
    q = tc["niwaki_ple_quant"]
    n = 0
    for path, mod in list(_iter_modules(model)):
        emb = getattr(mod, "ngram_embedding", None)
        if emb is None or not hasattr(emb, "shards"):
            continue
        dims = emb.dims
        shards = [
            MemmapPairedRows(index, f"{path}.ngram_embedding.shards.{i}",
                             group_size=int(q["group_size"]), bits=int(q["bits"]), pair=pair, dims=dims)
            for i in range(len(emb.shards))
        ]
        mod.ngram_embedding = MemmapShardedEmbedding(shards, emb.shard_offsets, dims)
        n += len(shards)
        log(f"PLE on disk: {path}: {len(shards)} shards, {sum(s.rows for s in shards)/1e6:.1f} M rows x {dims} dims, "
            f"{q['bits']}-bit g{q['group_size']} pair {pair}")
    return n


def load_niwaki_with_stores(model_dir: str, *, cache_rate: float | None = None, log=print):
    """(model, processor) with the PLE on disk and the experts on MaxToken's store."""
    import mlx.core as mx

    from maxtoken.mlx_backend.offload import SafetensorsIndex

    model_dir = os.path.abspath(model_dir)
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)  # niwaki_flash_load.py ships in the checkpoint
    from niwaki_flash_load import load

    t0 = time.perf_counter()
    model, processor = load(model_dir, lazy=True)  # nothing evaluated yet
    index = SafetensorsIndex(model_dir)
    if install_memmap_ple(model, model_dir, index, log) == 0:
        raise RuntimeError("no PLE shards found in the checkpoint")
    if cache_rate:
        from maxtoken.mlx_backend.offload import attach_expert_offload

        state = attach_expert_offload(model, model_dir, 1)
        mx.eval(model.parameters())
        per_layer = max(1, int(float(cache_rate) * state.glus[0].store.num_experts))
        total = state.resize_total(per_layer * len(state.glus))
        log(f"slot cache: {total} slots over {len(state.glus)} layers ({state.cache_bytes()/2**30:.2f} GiB)")
        model._maxtoken_offload_state = state
    else:
        from maxtoken.mlx_backend.ftw_mlx import attach_mapped_experts

        attach_mapped_experts(model, model_dir)
        mx.eval(model.parameters())
    log(f"loaded with stores in {time.perf_counter()-t0:.1f}s; active {mx.get_active_memory()/2**30:.2f} GiB "
        "(includes the file-backed store)")
    return model, processor
