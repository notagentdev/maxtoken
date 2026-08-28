"""Sparse sampling for the speculative verify loop.

A verify round needs the target's sampling distribution at every window
position and the drafter's at every draft position, and then reads a handful
of entries from them: p(d) and q(d) for the acceptance test, and a residual to
draw the correction from. Built densely, each distribution is a chain of
vocabulary-wide kernels — softmax, argpartition, a full argsort for top-p,
scatters, a renormalization — over 248 320 columns on this checkpoint, and the
acceptance test then reads its scalars back one ``.item()`` at a time, each a
GPU sync. Measured on Qwen3.8-27B that cost 7.2 ms a round on the target side
and another 4 ms inside the drafter, against a 55 ms plain decode step.

The shaped distribution is truncated to ``top_k`` entries by construction, so
everything here works on that support alone. One ``argpartition`` finds it,
one gather reads its logits, one ``logsumexp`` normalizes them against the
FULL vocabulary — so the top-p cutoff sees exactly the probabilities the dense
path would — and the rest is arithmetic over a few dozen numbers. On the
target side that arithmetic runs on the host after a single transfer; on the
draft side it runs on the device, so a draft chain never stops for a host
round trip and the whole round has one synchronization point.

Only the representation changes. The committed tokens follow exactly the
distribution the dense path produced — ``tests/mlx_backend/test_spec_sample.py``
checks the support, the probabilities and the acceptance histogram against it.
A sampler without a usable ``top_k`` (off, or wider than ``MAX_SPARSE_TOP_K``)
has no small support to work on and stays on the dense path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

import numpy as np

# Wider than this and the support is no longer "a few dozen numbers"; the dense
# path handles it, as it does a sampler with top_k off entirely.
MAX_SPARSE_TOP_K = 256

Row = Tuple[np.ndarray, np.ndarray]  # (token ids, probabilities) of one support


@dataclass(frozen=True)
class SamplerSpec:
    temperature: float
    top_k: int      # > 0: the support size
    top_p: float    # 0.0: off


def sampler_spec(sp) -> SamplerSpec | None:
    """The request's sampler in the form the sparse path needs, or None when
    the request is greedy or its sampler leaves no bounded support."""
    if getattr(sp, "is_greedy", False) or float(sp.temperature) <= 0.0:
        return None
    top_k = int(sp.top_k) if sp.top_k and sp.top_k > 0 else 0
    if top_k <= 0 or top_k > MAX_SPARSE_TOP_K:
        return None
    top_p = float(sp.top_p) if 0.0 < float(sp.top_p) < 1.0 else 0.0
    return SamplerSpec(max(float(sp.temperature), 1e-6), top_k, top_p)


# -- device side ---------------------------------------------------------------


def support(rows, spec: SamplerSpec):
    """``(ids, probs)`` of the ``top_k`` candidates of every row of ``rows``
    (``(R, V)`` logits), lazily. The probabilities are the candidates' shares
    of the FULL vocabulary's softmax — the numbers the dense path masks and
    then feeds to its top-p cutoff — so the host side can apply that cutoff
    without ever seeing the other columns."""
    import mlx.core as mx

    scaled = rows.astype(mx.float32) * (1.0 / spec.temperature)
    k = min(spec.top_k, int(scaled.shape[-1]))
    ids = mx.argpartition(-scaled, kth=k - 1, axis=-1)[..., :k]
    vals = mx.take_along_axis(scaled, ids, axis=-1)
    probs = mx.exp(vals - mx.logsumexp(scaled, axis=-1, keepdims=True))
    return ids, probs


def sample_row(row, spec: SamplerSpec):
    """Draw one token from the shaped distribution of a single logits row,
    entirely on the device: ``(token, ids, probs)`` with the support sorted by
    probability, top-p applied and renormalized — the proposal density the
    acceptance test needs, in the same form ``shape_rows`` produces."""
    import mlx.core as mx

    ids, probs = support(row[None], spec)
    ids, probs = ids[0], probs[0]
    if spec.top_p:
        order = mx.argsort(-probs)
        probs, ids = probs[order], ids[order]
        keep = (mx.cumsum(probs) - probs) < spec.top_p
        probs = mx.where(keep, probs, 0.0)
    probs = probs / mx.maximum(probs.sum(), 1e-30)
    choice = mx.random.categorical(mx.log(probs + 1e-30))
    return ids[choice].astype(mx.int32), ids, probs


# -- host side -----------------------------------------------------------------


def shape_rows(ids, probs, spec: SamplerSpec) -> List[Row]:
    """Finish what ``support`` started, per row, on host copies: the top-p
    cutoff (keep a candidate while the mass sorted before it is below top_p —
    the dense path's rule) and the renormalization over what survived."""
    ids = np.asarray(ids)
    probs = np.asarray(probs, dtype=np.float64)
    out: List[Row] = []
    for t, p in zip(ids, probs):
        if spec.top_p:
            order = np.argsort(-p, kind="stable")
            t, p = t[order], p[order]
            p = np.where((np.cumsum(p) - p) < spec.top_p, p, 0.0)
        total = float(p.sum())
        if total > 0.0:
            p = p / total
        else:  # a row of -inf logits; nothing meaningful to prefer
            p = np.full(p.shape, 1.0 / len(p))
        out.append((t.astype(np.int64), p))
    return out


def rows_of(ids, probs) -> List[Row]:
    """Rows the device already finished (``sample_row``'s ids/probs, stacked)."""
    return [
        (np.asarray(t, dtype=np.int64), np.asarray(p, dtype=np.float64))
        for t, p in zip(np.asarray(ids), np.asarray(probs))
    ]


def _prob(row: Row, token: int) -> float:
    ids, probs = row
    hit = np.nonzero(ids == token)[0]
    return float(probs[hit[0]]) if hit.size else 0.0


def _draw(row: Row, rng: np.random.Generator) -> int:
    ids, probs = row
    return int(ids[rng.choice(len(ids), p=probs / probs.sum())])


def accept(
    drafts: Sequence[int], P: Sequence[Row], Q: Sequence[Row], rng: np.random.Generator
) -> Tuple[int, int]:
    """Rejection sampling with residual correction over sparse rows.

    Draft ``d`` at position i survives with probability ``min(1, p(d)/q(d))``;
    the first rejection draws from the normalized residual ``(p - q)+`` and
    stops; when every draft survives the bonus token comes from ``P[len]``.
    Returns ``(accepted, token after the accepted prefix)``. Same rule as the
    dense ``MlxScheduler._accept_speculative`` — the committed tokens are
    distributed exactly as ``p``."""
    for i, d in enumerate(drafts):
        p_d, q_d = _prob(P[i], int(d)), _prob(Q[i], int(d))
        ratio = 0.0 if q_d <= 0.0 else min(1.0, p_d / q_d)
        if rng.random() < ratio:
            continue
        p_ids, p_probs = P[i]
        residual = dict(zip(p_ids.tolist(), p_probs.tolist()))
        for t, q in zip(*Q[i]):
            if t in residual:
                residual[t] = max(residual[t] - float(q), 0.0)
        total = sum(residual.values())
        if total <= 0.0:
            return i, _draw(P[i], rng)
        ids = np.fromiter(residual.keys(), dtype=np.int64, count=len(residual))
        probs = np.fromiter(residual.values(), dtype=np.float64, count=len(residual))
        return i, _draw((ids, probs), rng)
    return len(drafts), _draw(P[len(drafts)], rng)


def host_rng(mx_module: Any) -> np.random.Generator:
    """A host generator seeded from the device stream, so ``mx.random.seed``
    keeps controlling the whole round. One scalar read per request."""
    seed = int(mx_module.random.randint(0, 2**31 - 1).item())
    return np.random.default_rng(seed)
