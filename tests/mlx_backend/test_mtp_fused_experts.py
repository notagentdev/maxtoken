"""Replacement MTP heads ship transformers-5 fused expert stacks
(`experts.gate_up_proj` [E, 2*inter, hidden], `experts.down_proj`
[E, hidden, inter]); the drafter must unfuse them into mlx-lm's switch_mlp
layout with the gate rows first, like mlx-lm's own qwen3_5_moe sanitize."""
import numpy as np

from maxtoken.mlx_backend.mtp_draft import _split_fused_experts


def _fused(E=4, inter=8, hidden=16):
    gate_up = np.zeros((E, 2 * inter, hidden), dtype=np.float32)
    gate_up[:, :inter, :] = 1.0  # gate half
    gate_up[:, inter:, :] = 2.0  # up half
    return {
        "layers.0.mlp.experts.gate_up_proj": gate_up,
        "layers.0.mlp.experts.down_proj": np.full((E, hidden, inter), 3.0, dtype=np.float32),
        "layers.0.mlp.gate.weight": np.zeros((E, hidden), dtype=np.float32),
    }


def test_gate_rows_come_first():
    out = _split_fused_experts(_fused())
    assert "layers.0.mlp.experts.gate_up_proj" not in out
    assert "layers.0.mlp.experts.down_proj" not in out
    g = out["layers.0.mlp.switch_mlp.gate_proj.weight"]
    u = out["layers.0.mlp.switch_mlp.up_proj.weight"]
    d = out["layers.0.mlp.switch_mlp.down_proj.weight"]
    assert g.shape == (4, 8, 16) and (g == 1.0).all()
    assert u.shape == (4, 8, 16) and (u == 2.0).all()
    assert d.shape == (4, 16, 8) and (d == 3.0).all()
    assert "layers.0.mlp.gate.weight" in out  # router passes through


def test_half_mapped_block_is_left_alone():
    w = _fused()
    del w["layers.0.mlp.experts.down_proj"]
    out = _split_fused_experts(w)
    assert "layers.0.mlp.experts.gate_up_proj" in out
    assert "layers.0.mlp.switch_mlp.gate_proj.weight" not in out


def test_per_expert_layout_passes_through():
    w = {"layers.0.mlp.switch_mlp.gate_proj.weight": np.zeros((4, 8, 16), dtype=np.float32)}
    assert _split_fused_experts(w) == w


def test_norm_leaf_and_candidates():
    from maxtoken.mlx_backend.mtp_draft import _norm_candidates, _norm_leaf

    assert _norm_leaf("layers.0.post_attention_layernorm.weight") == "post_attention_layernorm"
    assert _norm_leaf("norm.weight") == "norm"
    assert _norm_leaf("layers.0.self_attn.q_norm.weight") == "q_norm"
    assert _norm_leaf("pre_fc_norm_embedding.weight") == "pre_fc_norm_embedding"
    assert _norm_leaf("layers.0.mlp.gate.weight") is None
    assert _norm_leaf("fc.weight") is None
    cands = list(_norm_candidates())
    assert cands[0] == frozenset()
    assert len(cands) == 1 + 15 + 1
    assert frozenset({"post_attention_layernorm"}) in cands  # the shipped Ornith-1.5 case
    assert len(cands[-1]) == 7  # the fully raw export (shisa-ai)


def test_offset_norms_only_and_all():
    from maxtoken.mlx_backend.mtp_draft import _offset_norms

    w = {"layers.0.post_attention_layernorm.weight": np.zeros(4), "norm.weight": np.zeros(4),
         "fc.weight": np.zeros((2, 4))}
    only = _offset_norms(w, "only:post_attention_layernorm")
    assert (only["layers.0.post_attention_layernorm.weight"] == 1).all()
    assert (only["norm.weight"] == 0).all() and (only["fc.weight"] == 0).all()
    every = _offset_norms(w, "all")
    assert (every["norm.weight"] == 1).all() and (every["fc.weight"] == 0).all()
    assert _offset_norms(w, "") is w
