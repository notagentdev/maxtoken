"""Prefix-cache exactness against a real checkpoint (needs_weights).

Run with MAXTOKEN_TEST_MLX_MODEL pointing at a local MLX checkpoint dir (any
mlx-lm model; a small one like OLMoE-1B-7B-4bit keeps it fast):

    MAXTOKEN_TEST_MLX_MODEL=~/models/olmoe pytest tests/mlx_backend -m needs_weights

The contract proven here is the strongest a prefix cache can give: restoring a
snapshot and continuing is BIT-IDENTICAL to having kept the original cache
alive. (A cold recompute may differ in bf16 rounding — chunked prefill and
decode accumulate in different orders; that is inherent to every prefix cache.)
"""

import os

import pytest

pytestmark = pytest.mark.needs_weights

MODEL_DIR = os.path.expanduser(os.environ.get("MAXTOKEN_TEST_MLX_MODEL", ""))
if not MODEL_DIR or not os.path.isdir(MODEL_DIR):
    pytest.skip("set MAXTOKEN_TEST_MLX_MODEL to a local MLX checkpoint dir",
                allow_module_level=True)
mx = pytest.importorskip("mlx.core", reason="needs mlx")


def _load():
    from mlx_lm import load

    return load(MODEL_DIR)


def _generate(model, cache, prompt_tail, n):
    from mlx_lm.generate import generate_step

    toks = []
    for tok, _ in generate_step(
        mx.array(prompt_tail), model, prompt_cache=cache, max_tokens=n
    ):
        toks.append(int(tok))
    return toks


def test_restore_equals_keeping_the_cache_alive():
    from mlx_lm.models.cache import make_prompt_cache

    from maxtoken.mlx_backend.prefix_cache import PrefixStore

    model, tokenizer = _load()
    p1 = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Name three colors and say one fact about each."}],
        add_generation_prompt=True,
    )
    cache1 = make_prompt_cache(model)
    mx.eval(model(mx.array(p1[:-1])[None], cache=cache1))
    out1 = _generate(model, cache1, p1[-1:], 48)
    conv = p1 + out1[:-1]

    store = PrefixStore(4 << 30)
    store.insert(conv, cache1)

    tail = tokenizer.encode(" Continue the list, please.", add_special_tokens=False)
    p2 = conv + tail

    def continue_with(cache, start):
        if start < len(p2) - 1:
            mx.eval(model(mx.array(p2[start:len(p2) - 1])[None], cache=cache))
        return _generate(model, cache, p2[-1:], 48)

    gold = continue_with(cache1, len(conv))  # conversation never left memory

    hit = store.lookup(p2)
    assert hit is not None
    entry, n = hit
    restored = store.restore(model, entry, n)
    assert continue_with(restored, n) == gold
