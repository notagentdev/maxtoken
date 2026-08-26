"""Commit part of a verify window without re-running the model.

A hybrid checkpoint's recurrent layers are the reason speculative decoding is
awkward on this architecture. An attention cache can be rewound by trimming
rows, so a rejected draft costs nothing; a gated-delta layer keeps a single
state tensor that has already absorbed the whole window by the time the
acceptance rule runs, and there is no arithmetic that takes it back.

The way out is to keep the state the recurrence passes through anyway. Run the
window one position at a time and hold each intermediate state; once the
acceptance rule has decided that ``keep`` of the window's positions survive,
the committed state is simply the one recorded at ``keep - 1``. Attention
layers trim as usual. Nothing is recomputed and nothing is carried into the
next round.

Without this a rejected round has to roll every cache back to where the window
started and re-feed the accepted tokens in the next window, where they cost
rows a second time and crowd out the drafts that would have earned them back.

The approach is MTPLX's (Apache-2.0, https://github.com/youssofal/MTPLX): its
``gdn_capture`` records per-position conv and recurrent state during the verify
forward and commits the accepted prefix from it. This is an independent
implementation of that idea against mlx-lm's own operators, narrowed to the
layout our checkpoints use.

The captured state is not small — one position of one layer is
``num_v_heads * head_v_dim * head_k_dim`` float32, 3 MiB on Qwen3.8-27B, so a
four-row window across 48 recurrent layers holds about 0.56 GiB until the round
commits. That is the price of not re-running the trunk.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from freetoken.utils import init_logger

logger = init_logger(__name__)

# The projections a capturable gated-delta layer must expose. Anything else
# (a fused qkvz layout, a future rename) declines capture and leaves the
# caller on its snapshot-and-rollback path.
_REQUIRED = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "conv1d", "norm",
             "out_proj", "A_log", "dt_bias")


def _layers(model):
    text = getattr(model, "language_model", model)
    inner = getattr(text, "model", None)
    return list(getattr(inner, "layers", ()) or ())


def _gdn_modules(model) -> list:
    return [
        layer.linear_attn
        for layer in _layers(model)
        if getattr(layer, "is_linear", False) and hasattr(layer, "linear_attn")
    ]


def supported(model) -> bool:
    """Whether every recurrent layer of this model can be captured."""
    modules = _gdn_modules(model)
    if not modules:
        return False
    return all(all(hasattr(m, name) for name in _REQUIRED) for m in modules)


def _make_capture_call(original, captures: dict):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.gated_delta import gated_delta_update

    def capture_call(self, inputs, mask=None, cache=None):
        if cache is None:
            return original(self, inputs, mask, cache)
        B, S, _ = inputs.shape

        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        keep = self.conv_kernel_size - 1
        if cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros((B, keep, self.conv_dim), dtype=inputs.dtype)
        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        # The convolution's state after position i is the window of `keep`
        # inputs ending at i — a view per position, no copy.
        conv_states = [conv_input[:, i + 1 : i + 1 + keep, :] for i in range(S)]
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        # One position at a time: the whole point is the intermediate states,
        # and mlx-lm's kernel returns only the last one.
        state = cache[1]
        outs, states = [], []
        step_mask = None
        for i in range(S):
            if mask is not None and not isinstance(mask, str):
                step_mask = mask[:, i : i + 1]
            out_i, state = gated_delta_update(
                q[:, i : i + 1],
                k[:, i : i + 1],
                v[:, i : i + 1],
                a[:, i : i + 1],
                b[:, i : i + 1],
                self.A_log,
                self.dt_bias,
                state,
                step_mask,
                use_kernel=not self.training,
            )
            outs.append(out_i)
            states.append(state)

        captures[id(self)] = (conv_states, states)
        cache[0] = mx.contiguous(conv_states[-1])
        cache[1] = states[-1]
        cache.advance(S)

        out = self.norm(mx.concatenate(outs, axis=1), z)
        return self.out_proj(out.reshape(B, S, -1))

    return capture_call


@contextmanager
def capture(model):
    """Run a forward with every recurrent layer recording its per-position state.

    Yields the capture dict, keyed by the id of the gated-delta module, or an
    empty dict when the model's layout is not one this can record — in which
    case the caller keeps whatever fallback it had.
    """
    modules = _gdn_modules(model)
    if not modules or not supported(model):
        yield {}
        return
    cls = type(modules[0])
    if any(type(m) is not cls for m in modules):
        yield {}
        return
    captures: dict[int, Any] = {}
    original = cls.__call__
    cls.__call__ = _make_capture_call(original, captures)
    try:
        yield captures
    finally:
        cls.__call__ = original


def commit_prefix(model, cache, captures: dict, keep: int, verified: int) -> bool:
    """Bind the caches to the first ``keep`` positions of a ``verified``-row window.

    Returns False without touching anything if the captures do not cover every
    recurrent layer, so a caller can still fall back.
    """
    import mlx.core as mx

    if not captures or keep <= 0 or keep > verified:
        return False
    layers = _layers(model)
    if len(layers) != len(cache):
        return False

    # Validate every layer before mutating any of them: a half-committed set of
    # caches has no way back.
    plan = []
    for layer, entry in zip(layers, cache):
        if getattr(layer, "is_linear", False):
            recorded = captures.get(id(getattr(layer, "linear_attn", None)))
            if recorded is None or len(recorded[1]) < keep:
                return False
            plan.append((entry, recorded))
        else:
            if not hasattr(entry, "trim"):
                return False
            plan.append((entry, None))

    trim = verified - keep
    for entry, recorded in plan:
        if recorded is None:
            if trim:
                entry.trim(trim)
        else:
            conv_states, states = recorded
            entry[0] = mx.contiguous(conv_states[keep - 1])
            entry[1] = states[keep - 1]
    return True
