"""Rejection sampling with residual correction must not change what the model says.

The whole point of speculative decoding is that it is free of quality cost: the
committed tokens have to follow the TARGET's distribution exactly, whatever the
drafter proposes. That is a statistical property, so it is tested statistically
against a known distribution — a wrong acceptance rule (accepting too eagerly,
or drawing the correction from p instead of the residual) shifts the histogram
and fails here.
"""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core", reason="acceptance runs on mx arrays")

from maxtoken.mlx_backend.worker import MlxScheduler  # noqa: E402


def _sched():
    s = MlxScheduler.__new__(MlxScheduler)
    s._mx = mx
    return s


def test_committed_token_follows_the_target_distribution():
    """p is the truth, q is a deliberately different proposal. Over many rounds
    the emitted token must be distributed as p, not as q and not as a blend."""
    s = _sched()
    p = mx.array([0.5, 0.3, 0.15, 0.05])
    q = mx.array([0.1, 0.1, 0.4, 0.4])  # very different from p
    mx.random.seed(0)

    counts = [0, 0, 0, 0]
    rounds = 4000
    for _ in range(rounds):
        d = int(mx.random.categorical(mx.log(q)).item())
        accepted, tok = s._accept_speculative([d], [p, p], [q])
        counts[d if accepted else tok] += 1

    got = [c / rounds for c in counts]
    for i, want in enumerate([0.5, 0.3, 0.15, 0.05]):
        assert abs(got[i] - want) < 0.03, f"token {i}: {got[i]:.3f} vs {want}"


def test_a_perfect_drafter_is_always_accepted():
    """q == p: every draft survives (min(1, p/q) == 1), so a round commits the
    whole window plus its bonus."""
    s = _sched()
    p = mx.array([0.6, 0.25, 0.1, 0.05])
    mx.random.seed(1)
    for _ in range(50):
        d = int(mx.random.categorical(mx.log(p)).item())
        accepted, _ = s._accept_speculative([d], [p, p], [p])
        assert accepted == 1


def test_a_draft_the_target_rules_out_is_rejected():
    """p(d) == 0: the draft cannot survive, and the correction must come from
    the residual — never from the impossible token."""
    s = _sched()
    p = mx.array([0.7, 0.3, 0.0, 0.0])
    q = mx.array([0.0, 0.0, 0.5, 0.5])
    mx.random.seed(2)
    for _ in range(50):
        accepted, tok = s._accept_speculative([2], [p, p], [q])
        assert accepted == 0
        assert tok in (0, 1)


def test_acceptance_stops_at_the_first_rejection():
    """Positions after a rejection are not committed: their drafts were built on
    a token the target did not take."""
    s = _sched()
    good = mx.array([1.0, 0.0, 0.0, 0.0])
    p_list = [good, mx.array([0.0, 1.0, 0.0, 0.0]), good]
    q_list = [good, mx.array([0.0, 0.0, 1.0, 0.0])]
    mx.random.seed(3)
    accepted, tok = s._accept_speculative([0, 2], p_list, q_list)
    assert accepted == 1          # first draft fine, second impossible under p
    assert tok == 1               # correction drawn from position 1's residual
