"""FTW-MLX zero-copy expert store: repack roundtrip for both disk layouts.

Needs mlx (the store imports mapped tensors via DLPack); the repack itself is
exercised through a crafted checkpoint in tmp_path, with MAXTOKEN_MLX_FTW_DIR
pointing the cache there too.
"""

import json
import os

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core", reason="mapped store needs mlx")

from maxtoken.mlx_backend.ftw_mlx import (  # noqa: E402 -- after importorskip
    MappedExpertStore,
    _PAGE,
    repack_experts,
)

# tests/ is not a package; load the sibling module's helpers by path.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_offload_helpers", os.path.join(os.path.dirname(__file__), "test_offload.py")
)
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
_fake_glu_tensors = _helpers._fake_glu_tensors
write_safetensors = _helpers.write_safetensors


@pytest.mark.parametrize("packed", [True, False], ids=["packed", "classic"])
@pytest.mark.parametrize("stacked", [True, False], ids=["stacked", "per-expert"])
def test_repack_and_mapped_views_roundtrip(tmp_path, monkeypatch, stacked, packed):
    monkeypatch.setenv("MAXTOKEN_MLX_FTW_DIR", str(tmp_path / "ftw-cache"))
    monkeypatch.setenv("MAXTOKEN_MLX_FTW_PACK", "1" if packed else "0")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    rng = np.random.default_rng(1)
    tensors = _fake_glu_tensors(rng, num_experts=5, stacked=stacked, prefix="model.mlp")
    write_safetensors(model_dir / "m.safetensors", tensors)

    out_dir = repack_experts(str(model_dir), ["model.mlp.switch_mlp"])
    manifest = json.load(open(os.path.join(out_dir, "manifest.json")))
    for meta in manifest["tensors"].values():
        assert meta["offset"] % _PAGE == 0  # the whole point of the repack

    store = MappedExpertStore(out_dir)
    params = store.glu_params("model.mlp.switch_mlp")

    def stacked_source(proj, part):
        if stacked:
            return tensors[f"model.mlp.switch_mlp.{proj}.{part}"]
        return np.stack(
            [tensors[f"model.mlp.experts.{e}.{proj}.{part}"] for e in range(5)]
        )

    if packed:
        # gate and up interleaved PER EXPERT into one tensor; the classic
        # per-proj entries for gate/up must be gone.
        assert "gate_proj" not in params and "up_proj" not in params
        got = np.array(params["gate_up_proj"]["weight"])
        g = stacked_source("gate_proj", "weight")
        u = stacked_source("up_proj", "weight")
        want = np.stack(
            [np.concatenate([g[e], u[e]], axis=0) for e in range(5)]
        )
        assert got.shape == want.shape
        assert got.tolist() == want.tolist()
    else:
        got = np.array(params["gate_proj"]["weight"])
        want = stacked_source("gate_proj", "weight")
        assert got.shape == want.shape
        assert got.tolist() == want.tolist()
    down = np.array(params["down_proj"]["weight"])
    assert down.tolist() == stacked_source("down_proj", "weight").tolist()

    # idempotent: a second call must reuse the manifest, not rewrite the store
    mtime = os.path.getmtime(os.path.join(out_dir, "experts.ftwm"))
    assert repack_experts(str(model_dir), ["model.mlp.switch_mlp"]) == out_dir
    assert os.path.getmtime(os.path.join(out_dir, "experts.ftwm")) == mtime
