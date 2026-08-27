"""Commit part of a verify window without re-running the model.

A hybrid checkpoint's recurrent layers are the reason speculative decoding is
awkward on this architecture. An attention cache can be rewound by trimming
rows, so a rejected draft costs nothing; a gated-delta layer keeps a single
state tensor that has already absorbed the whole window by the time the
acceptance rule runs, and there is no arithmetic that takes it back.

Without a way back, a rejected round has to roll every cache to where the
window started and re-feed the accepted tokens in the next window, where they
cost rows a second time and crowd out the drafts that would have earned them
back.

The way out is that the recurrence is cheap and everything in front of it is
not. The window's forward runs exactly as it always did — one call into
mlx-lm's kernel, which carries the state through all T positions in registers
and returns the last one. What the capture keeps is only the recurrence's
INPUTS: the post-convolution q/k/v, the gates, and the state the window
started from. If the acceptance rule then keeps a shorter prefix, replaying
that prefix is one more pass over the recurrence alone — no projections, no
convolution, no MLP, no attention.

So an all-accept round pays nothing at all for the ability to commit, and a
rejected one pays only for the part that is small. Holding the inputs costs
about 180 KiB per layer for a four-row window; holding a state per position
would have cost 3 MiB per layer per position.

The idea of committing a captured prefix is MTPLX's (Apache-2.0,
https://github.com/youssofal/MTPLX). Their kernel records per-position state
during the forward and replays from a tape of deltas; this reaches the same
place with mlx-lm's own operators and no custom Metal.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from freetoken.utils import init_logger

logger = init_logger(__name__)

# The projections a capturable gated-delta layer must expose. Anything else
# (a fused qkvz layout, a future rename) declines capture and leaves the
# caller on its snapshot-and-rollback path.
_REQUIRED = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "conv1d",
             "norm", "out_proj", "A_log", "dt_bias")


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

        state_in = cache[1]
        out, state = gated_delta_update(
            q, k, v, a, b, self.A_log, self.dt_bias, state_in, mask,
            use_kernel=not self.training,
        )

        # References only. Nothing here is computed for the capture's sake; a
        # round that accepts everything never looks at it again.
        captures[id(self)] = {
            "gdn": self,
            "conv_input": conv_input,
            "conv_keep": keep,
            "q": q, "k": k, "v": v, "a": a, "b": b,
            "state_in": state_in,
            "mask": mask,
        }

        cache[0] = mx.contiguous(conv_input[:, S:, :])
        cache[1] = state
        cache.advance(S)

        return self.out_proj(self.norm(out, z).reshape(B, S, -1))

    return capture_call


@contextmanager
def capture(model):
    """Run a forward with every recurrent layer keeping its recurrence inputs.

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
    from mlx_lm.models.gated_delta import gated_delta_update

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
            if recorded is None:
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
            continue
        if not trim:
            continue  # the window was kept whole; the caches already hold it
        gdn = recorded["gdn"]
        conv_keep = recorded["conv_keep"]
        # The convolution's state after position keep-1 is the window of
        # `conv_keep` inputs ending there — a slice, not a computation.
        entry[0] = mx.contiguous(
            recorded["conv_input"][:, keep : keep + conv_keep, :]
        )
        # The recurrence alone, replayed over the accepted prefix. One kernel
        # call per layer, and only on a round that rejected something.
        mask = recorded["mask"]
        _, state = gated_delta_update(
            recorded["q"][:, :keep],
            recorded["k"][:, :keep],
            recorded["v"][:, :keep],
            recorded["a"][:, :keep],
            recorded["b"][:, :keep],
            gdn.A_log,
            gdn.dt_bias,
            recorded["state_in"],
            None if mask is None or isinstance(mask, str) else mask[:, :keep],
            use_kernel=not gdn.training,
        )
        entry[1] = state
    return True
