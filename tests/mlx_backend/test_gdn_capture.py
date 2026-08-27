"""Committing a captured prefix must be all-or-nothing, and replay exactly the
accepted rows.

A wrong length here binds a recurrent layer to a state the model never reached,
and nothing downstream would notice — the output just quietly drifts. A partial
commit is worse still: some layers on the accepted prefix, others past it, with
no snapshot left to undo either.

The arithmetic of the replay is checked end-to-end instead: greedy generation
through the capture path is token-for-token identical to the rollback path.
"""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core", reason="capture binds mx arrays")

import mlx_lm.models.gated_delta as gated_delta  # noqa: E402

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
    layers = [
        SimpleNamespace(is_linear=True, linear_attn=SimpleNamespace(
            A_log=mx.zeros((2,)), dt_bias=mx.zeros((2,)), training=False))
        for _ in range(n_linear)
    ]
    layers += [SimpleNamespace(is_linear=False) for _ in range(n_attn)]
    return SimpleNamespace(model=SimpleNamespace(layers=layers)), layers


def _captures(layers, window=4):
    caps = {}
    for layer in layers:
        if not layer.is_linear:
            continue
        caps[id(layer.linear_attn)] = {
            "gdn": layer.linear_attn,
            "conv_input": mx.arange(3 + window).reshape(1, 3 + window, 1),
            "conv_keep": 3,
            "q": mx.zeros((1, window, 1, 1)),
            "k": mx.zeros((1, window, 1, 1)),
            "v": mx.zeros((1, window, 1, 1)),
            "a": mx.zeros((1, window, 1)),
            "b": mx.zeros((1, window, 1)),
            "state_in": mx.zeros((1, 1, 1, 1)),
            "mask": None,
        }
    return caps


@pytest.fixture
def replay(monkeypatch):
    """Record what the recurrence replay is asked to run."""
    seen = []

    def fake(q, k, v, a, b, A_log, dt_bias, state, mask=None, use_kernel=True):
        seen.append(int(q.shape[1]))
        return None, mx.full((1, 1, 1, 1), float(q.shape[1]))

    monkeypatch.setattr(gated_delta, "gated_delta_update", fake)
    return seen


def test_replays_exactly_the_accepted_rows(replay):
    model, layers = _model()
    cache = [_ArraysCache(), _ArraysCache(), _KVCache()]

    assert gdn_capture.commit_prefix(model, cache, _captures(layers),
                                     keep=2, verified=4)

    assert replay == [2, 2], "each recurrent layer replays the accepted prefix"
    for entry in cache[:2]:
        # conv state after position keep-1 is conv_input[:, keep:keep+3].
        assert [int(x) for x in entry[0].reshape(-1)] == [2, 3, 4]
        assert float(entry[1].reshape(-1)[0]) == 2.0
    assert cache[2].trimmed == 2


def test_a_whole_window_costs_no_replay(replay):
    model, layers = _model()
    cache = [_ArraysCache(), _ArraysCache(), _KVCache()]

    assert gdn_capture.commit_prefix(model, cache, _captures(layers),
                                     keep=4, verified=4)

    assert replay == [], "nothing was rejected, so nothing is recomputed"
    assert cache[2].trimmed == 0
    assert cache[0][0] is None, "the forward already left the caches correct"


def test_commit_is_refused_whole_when_a_layer_is_missing(replay):
    model, layers = _model()
    caps = _captures(layers)
    caps.pop(id(layers[1].linear_attn))
    cache = [_ArraysCache(), _ArraysCache(), _KVCache()]

    assert not gdn_capture.commit_prefix(model, cache, caps, keep=2, verified=4)
    # Nothing may have moved: the caller still needs its snapshot to be valid.
    assert replay == []
    assert cache[0][0] is None and cache[1][0] is None
    assert cache[2].trimmed == 0


@pytest.mark.parametrize("keep,verified", [(0, 4), (5, 4), (-1, 4)])
def test_commit_rejects_impossible_windows(keep, verified, replay):
    model, layers = _model()
    cache = [_ArraysCache(), _ArraysCache(), _KVCache()]
    assert not gdn_capture.commit_prefix(
        model, cache, _captures(layers), keep=keep, verified=verified
    )
    assert replay == []


def test_unsupported_model_declines_capture():
    model, _ = _model()
    assert not gdn_capture.supported(model)
    with gdn_capture.capture(model) as caps:
        assert caps == {}
