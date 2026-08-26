"""Committing a captured prefix must be all-or-nothing, and land on the right row.

A wrong index here binds a recurrent layer to a state the model never reached,
and nothing downstream would notice — the output just quietly drifts. A partial
commit is worse still: some layers on the accepted prefix, others past it, with
no snapshot left to undo either.
"""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core", reason="capture binds mx arrays")

from freetoken.mlx_backend import gdn_capture  # noqa: E402


class _ArraysCache:
    """Stands in for mlx-lm's ArraysCache: two state slots, no offset."""

    def __init__(self):
        self.cache = [None, None]

    def __setitem__(self, i, v):
        self.cache[i] = v

    def __getitem__(self, i):
        return self.cache[i]


class _KVCache:
    def __init__(self):
        self.trimmed = 0

    def trim(self, n):
        self.trimmed += n
        return n


def _model(n_linear=2, n_attn=1):
    layers = []
    for _ in range(n_linear):
        layers.append(SimpleNamespace(is_linear=True, linear_attn=SimpleNamespace()))
    for _ in range(n_attn):
        layers.append(SimpleNamespace(is_linear=False))
    inner = SimpleNamespace(layers=layers)
    return SimpleNamespace(model=inner), layers


def _captures(layers, positions=4):
    caps = {}
    for layer in layers:
        if not layer.is_linear:
            continue
        conv = [mx.full((1, 3, 4), float(i)) for i in range(positions)]
        state = [mx.full((1, 2, 2, 2), float(i)) for i in range(positions)]
        caps[id(layer.linear_attn)] = (conv, state)
    return caps


def test_commit_binds_the_kept_row_and_trims_attention():
    model, layers = _model()
    caps = _captures(layers)
    cache = [_ArraysCache(), _ArraysCache(), _KVCache()]

    assert gdn_capture.commit_prefix(model, cache, caps, keep=2, verified=4)

    for entry in cache[:2]:
        # keep=2 means the state recorded after the second position, index 1.
        assert float(entry[0].reshape(-1)[0]) == 1.0
        assert float(entry[1].reshape(-1)[0]) == 1.0
    assert cache[2].trimmed == 2


def test_commit_is_refused_whole_when_a_layer_is_missing():
    model, layers = _model()
    caps = _captures(layers)
    caps.pop(id(layers[1].linear_attn))
    cache = [_ArraysCache(), _ArraysCache(), _KVCache()]

    assert not gdn_capture.commit_prefix(model, cache, caps, keep=2, verified=4)
    # Nothing may have moved: the caller still needs its snapshot to be valid.
    assert cache[0][0] is None and cache[1][0] is None
    assert cache[2].trimmed == 0


@pytest.mark.parametrize("keep,verified", [(0, 4), (5, 4), (-1, 4)])
def test_commit_rejects_impossible_windows(keep, verified):
    model, layers = _model()
    cache = [_ArraysCache(), _ArraysCache(), _KVCache()]
    assert not gdn_capture.commit_prefix(
        model, cache, _captures(layers), keep=keep, verified=verified
    )


def test_full_window_commit_trims_nothing():
    model, layers = _model()
    cache = [_ArraysCache(), _ArraysCache(), _KVCache()]
    assert gdn_capture.commit_prefix(
        model, cache, _captures(layers), keep=4, verified=4
    )
    assert cache[2].trimmed == 0
    assert float(cache[0][1].reshape(-1)[0]) == 3.0


def test_unsupported_model_declines_capture():
    model, _ = _model()
    assert not gdn_capture.supported(model)
    with gdn_capture.capture(model) as caps:
        assert caps == {}
