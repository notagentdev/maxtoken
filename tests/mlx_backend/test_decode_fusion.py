"""decode_fusion must change the launch count, not the answer.

A tiny qwen3_5_moe model (two gated-delta layers, one attention layer, eight
experts) stands in for Ornith: after install, a decode step must agree with the
stock forward, a prefill must be bit-identical to it (the fused paths are
decode-only by design), a padded batch must still take the stock recurrence,
and uninstall must leave the classes as it found them.
"""

import pytest

mx = pytest.importorskip("mlx.core", reason="decode fusion needs MLX")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("mlx_lm.models.qwen3_5_moe", reason="needs mlx-lm's qwen3_5_moe")

from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

from maxtoken.mlx_backend import decode_fusion  # noqa: E402


def _tiny_model():
    from mlx_lm.models.qwen3_5_moe import Model, ModelArgs

    cfg = {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 256,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 4096,
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
            "full_attention_interval": 3,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32,
            "norm_topk_prob": True,
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
        },
    }
    mx.random.seed(0)
    model = Model(ModelArgs.from_dict(cfg))
    nn.quantize(model, group_size=32, bits=4)
    mx.eval(model.parameters())
    return model


def _greedy(model, ids, n=6):
    cache = make_prompt_cache(model)
    logits = model(mx.array([ids]), cache=cache)
    mx.eval(logits)
    out = []
    y = mx.argmax(logits[:, -1, :], axis=-1)
    for _ in range(n):
        out.append(int(y.item()))
        y = mx.argmax(model(y[:, None], cache=cache)[:, -1, :], axis=-1)
    return logits, out


@pytest.fixture
def model():
    m = _tiny_model()
    yield m
    decode_fusion.uninstall()


def test_install_reports_the_fused_and_compiled_blocks(model):
    report = decode_fusion.install(model)
    assert report == {"gdn_fused": 2, "moe_compiled": 3}


def test_decode_agrees_and_prefill_is_untouched(model):
    ids = [1, 7, 3, 9, 12, 5, 8]
    logits_stock, toks_stock = _greedy(model, ids)
    decode_fusion.install(model)
    logits_fused, toks_fused = _greedy(model, ids)
    # Prefill runs the stock code path under the patch: bit-identical.
    assert mx.array_equal(logits_stock, logits_fused)
    assert toks_stock == toks_fused


def test_decode_step_logits_match_closely(model):
    ids = [4, 4, 2, 11]
    cache_a, cache_b = make_prompt_cache(model), make_prompt_cache(model)
    mx.eval(model(mx.array([ids]), cache=cache_a), model(mx.array([ids]), cache=cache_b))
    y = mx.array([[9]])
    stock = model(y, cache=cache_a)
    decode_fusion.install(model)
    fused = model(y, cache=cache_b)
    mx.eval(stock, fused)
    diff = float(mx.abs(stock.astype(mx.float32) - fused.astype(mx.float32)).max())
    scale = float(mx.abs(stock.astype(mx.float32)).max())
    assert diff <= 1e-2 * max(scale, 1.0), f"decode step drifted by {diff} (scale {scale})"


def test_batched_decode_takes_the_compiled_path(model):
    """Two rows decoding together (continuous batching's shape): the compiled
    MoE forward is traced for B=2 and agrees with the stock forward."""
    prompts = mx.array([[4, 4, 2, 11], [6, 1, 9, 2]])
    cache_a, cache_b = make_prompt_cache(model), make_prompt_cache(model)
    mx.eval(model(prompts, cache=cache_a), model(prompts, cache=cache_b))
    y = mx.array([[9], [3]])
    stock = model(y, cache=cache_a)
    decode_fusion.install(model)
    fused = model(y, cache=cache_b)
    mx.eval(stock, fused)
    assert fused.shape == (2, 1, 256)
    assert float(mx.abs(stock.astype(mx.float32) - fused.astype(mx.float32)).max()) <= 1e-2
    assert len(decode_fusion._COMPILED) == 3


def test_uninstall_restores_the_classes(model):
    from mlx_lm.models.qwen3_5 import GatedDeltaNet
    from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

    gdn_call, moe_call = GatedDeltaNet.__call__, Qwen3NextSparseMoeBlock.__call__
    decode_fusion.install(model)
    assert GatedDeltaNet.__call__ is not gdn_call
    assert Qwen3NextSparseMoeBlock.__call__ is not moe_call
    decode_fusion.uninstall()
    assert GatedDeltaNet.__call__ is gdn_call
    assert Qwen3NextSparseMoeBlock.__call__ is moe_call
    assert not decode_fusion._COMPILED


def test_dispatch_defaults_do_not_override_the_user(monkeypatch):
    from maxtoken.mlx_backend import metal_env

    monkeypatch.delenv("MLX_MAX_OPS_PER_BUFFER", raising=False)
    monkeypatch.setenv("MLX_MAX_MB_PER_BUFFER", "77")
    got = metal_env.apply_dispatch_defaults()
    assert got["MLX_MAX_OPS_PER_BUFFER"] == metal_env.DEFAULTS["MLX_MAX_OPS_PER_BUFFER"]
    assert got["MLX_MAX_MB_PER_BUFFER"] == "77"


def test_prefix_store_budget_comes_from_headroom():
    from maxtoken.mlx_backend.worker import prefix_store_budget

    gib = 1 << 30
    # 32 GB machine, 18 GB model: a quarter of (32 - 18 - 6) = 2 GB, not 4.8.
    assert prefix_store_budget(32 * gib, 18 * gib) == 2 * gib
    # A small model still gets no more than the old 15% of RAM.
    assert prefix_store_budget(32 * gib, 4 * gib) == int(0.15 * 32 * gib)
    # A model that fills the machine keeps the floor.
    assert prefix_store_budget(32 * gib, 30 * gib) == 256 << 20


def test_paced_prefill_matches_the_stock_forward(model):
    """A prefill evaluated every two layers must produce the same logits and
    the same caches as the unpaced forward; a single-position forward takes
    the original path untouched."""
    from maxtoken.mlx_backend import prefill_pacing

    ids = mx.array([[1, 7, 3, 9, 12, 5, 8, 2]])
    cache_a, cache_b = make_prompt_cache(model), make_prompt_cache(model)
    stock = model(ids, cache=cache_a)
    mx.eval(stock)
    assert prefill_pacing.install(model, pace=2) == 2
    try:
        paced = model(ids, cache=cache_b)
        mx.eval(paced)
        assert mx.array_equal(stock, paced)
        for a, b in zip(cache_a, cache_b):
            for x, y in zip(a.state, b.state):
                if x is not None:
                    assert mx.array_equal(x, y)
        y = mx.array([[4]])
        s1, s2 = model(y, cache=cache_a), model(y, cache=cache_b)
        mx.eval(s1, s2)
        assert mx.array_equal(s1, s2)
    finally:
        prefill_pacing.uninstall()
