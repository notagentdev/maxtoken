"""The sparse sampling path must produce exactly what the dense one did.

``spec_sample`` changes only the representation of the target's and drafter's
distributions — a top-k support instead of a vocabulary-wide row. Everything
that follows from them (which tokens can be drawn, with what probability, and
which drafts survive) has to agree with the dense ``_shaped_dist`` /
``_accept_speculative`` pair to the last renormalization, because the promise
of speculative decoding is that the committed tokens are the target's own.
"""

from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="sparse sampling runs on mx arrays")

from maxtoken.mlx_backend import spec_sample  # noqa: E402
from maxtoken.mlx_backend.worker import MlxScheduler  # noqa: E402


def _sp(temperature=0.7, top_k=40, top_p=0.95, greedy=False):
    return SimpleNamespace(
        is_greedy=greedy, temperature=temperature, top_k=top_k, top_p=top_p
    )


def _dense(sp):
    s = MlxScheduler.__new__(MlxScheduler)
    s._mx = mx
    return s._shaped_dist(sp)


def _dense_row(shape, row):
    probs = np.asarray(shape(row), dtype=np.float64)
    ids = np.nonzero(probs > 0)[0]
    return ids, probs[ids]


def _sparse_as_dict(row):
    ids, probs = row
    return {int(t): float(p) for t, p in zip(ids, probs) if p > 0}


def _numpy_shape(row, temperature, top_k, top_p):
    """The rule both paths implement, written plainly: temperature softmax,
    keep the top_k, keep a token while the mass sorted before it is under
    top_p, renormalize."""
    p = np.exp(row / temperature - np.max(row / temperature))
    p /= p.sum()
    if top_k:
        cut = np.argsort(-p)[top_k:]
        p[cut] = 0.0
    if top_p:
        order = np.argsort(-p)
        before = np.cumsum(p[order]) - p[order]
        p[order[before >= top_p]] = 0.0
    return p / p.sum()


@pytest.mark.parametrize("top_p", [0.95, 0.6, 0.0])
@pytest.mark.parametrize("top_k", [40, 5])
def test_dense_shape_matches_numpy(top_k, top_p):
    """The dense path, against a plain numpy statement of the rule. This is
    the regression test for the reversed-index scatter that used to collapse
    every top-p distribution to a single token."""
    shape = _dense(_sp(top_k=top_k, top_p=top_p))
    mx.random.seed(11)
    # Flat enough that no row is a point mass at top_p=0.6 (the collapse
    # would then be invisible), peaked enough that the 0.6 cutoff bites.
    rows = mx.random.normal((3, 1000)) * 1.5
    for i in range(3):
        want = _numpy_shape(np.asarray(rows[i], dtype=np.float64), 0.7, top_k, top_p)
        got = np.asarray(shape(rows[i]), dtype=np.float64)
        assert np.count_nonzero(want) > 1, "a test row must not be a point mass"
        assert np.array_equal(want > 0, got > 0)
        assert np.abs(want - got).max() < 1e-5


@pytest.mark.parametrize("top_p", [0.95, 0.6, 0.0])
@pytest.mark.parametrize("top_k", [40, 5])
def test_support_and_shape_match_the_dense_distribution(top_k, top_p):
    """Same support, same probabilities, on random logits with no ties."""
    sp = _sp(top_k=top_k, top_p=top_p)
    spec = spec_sample.sampler_spec(sp)
    shape = _dense(sp)
    mx.random.seed(11)
    rows = mx.random.normal((3, 1000)) * 3.0
    ids, probs = spec_sample.support(rows, spec)
    sparse = spec_sample.shape_rows(ids, probs, spec)
    for i in range(3):
        want_ids, want_p = _dense_row(shape, rows[i])
        got = _sparse_as_dict(sparse[i])
        assert sorted(got) == sorted(int(t) for t in want_ids)
        for t, p in zip(want_ids, want_p):
            assert abs(got[int(t)] - p) < 1e-5


def test_sample_row_draws_from_the_dense_distribution():
    """The device-side draft sampler: histogram over many draws against the
    dense shaped distribution, and the support it reports is that distribution."""
    sp = _sp(temperature=1.0, top_k=8, top_p=0.9)
    spec = spec_sample.sampler_spec(sp)
    shape = _dense(sp)
    mx.random.seed(5)
    row = mx.random.normal((50,)) * 2.0
    want_ids, want_p = _dense_row(shape, row)
    want = dict(zip(want_ids.tolist(), want_p.tolist()))

    _, ids, probs = spec_sample.sample_row(row, spec)
    got = _sparse_as_dict((np.asarray(ids), np.asarray(probs)))
    assert sorted(got) == sorted(want)
    for t, p in want.items():
        assert abs(got[t] - p) < 1e-5

    counts = {}
    rounds = 3000
    for _ in range(rounds):
        tok = int(spec_sample.sample_row(row, spec)[0].item())
        counts[tok] = counts.get(tok, 0) + 1
    assert set(counts) <= set(want)
    for t, p in want.items():
        assert abs(counts.get(t, 0) / rounds - p) < 0.03


def test_sampler_spec_declines_what_the_sparse_path_cannot_represent():
    assert spec_sample.sampler_spec(_sp(greedy=True)) is None
    assert spec_sample.sampler_spec(_sp(temperature=0.0)) is None
    assert spec_sample.sampler_spec(_sp(top_k=0)) is None
    assert spec_sample.sampler_spec(_sp(top_k=-1)) is None
    assert spec_sample.sampler_spec(_sp(top_k=spec_sample.MAX_SPARSE_TOP_K + 1)) is None
    spec = spec_sample.sampler_spec(_sp(top_k=20, top_p=1.0))
    assert spec == spec_sample.SamplerSpec(0.7, 20, 0.0)


def _row(ids, probs):
    return np.asarray(ids, dtype=np.int64), np.asarray(probs, dtype=np.float64)


def test_committed_token_follows_the_target_distribution():
    """Same statistical contract as the dense test: p is the truth, q a
    deliberately different proposal, and the emitted token must be
    distributed as p."""
    p = _row([0, 1, 2, 3], [0.5, 0.3, 0.15, 0.05])
    q = _row([0, 1, 2, 3], [0.1, 0.1, 0.4, 0.4])
    rng = np.random.default_rng(0)
    counts = [0, 0, 0, 0]
    rounds = 4000
    for _ in range(rounds):
        d = int(rng.choice(4, p=q[1]))
        accepted, tok = spec_sample.accept([d], [p, p], [q], rng)
        counts[d if accepted else tok] += 1
    for i, want in enumerate([0.5, 0.3, 0.15, 0.05]):
        assert abs(counts[i] / rounds - want) < 0.03


def test_supports_need_not_overlap():
    """A residual over a target support the proposal never touched is the
    target itself; a draft outside the target's support is always rejected."""
    p = _row([7, 8], [0.7, 0.3])
    q = _row([1, 2], [0.5, 0.5])
    rng = np.random.default_rng(2)
    for _ in range(50):
        accepted, tok = spec_sample.accept([2], [p, p], [q], rng)
        assert accepted == 0
        assert tok in (7, 8)


def test_acceptance_stops_at_the_first_rejection():
    good = _row([0], [1.0])
    p_list = [good, _row([1], [1.0]), good]
    q_list = [good, _row([2], [1.0])]
    accepted, tok = spec_sample.accept([0, 2], p_list, q_list, np.random.default_rng(3))
    assert accepted == 1
    assert tok == 1


def test_no_drafts_draws_the_bonus_from_the_single_position():
    p = _row([4, 5], [0.999, 0.001])
    accepted, tok = spec_sample.accept([], [p], [], np.random.default_rng(4))
    assert accepted == 0
    assert tok == 4
