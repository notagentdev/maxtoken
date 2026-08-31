"""Bound what a prefill keeps in flight, so decode may run wide command buffers.

MLX commits a Metal command buffer every few ops or few tens of MB of fresh
allocations by default, and the GPU idles at every boundary; on a deep hybrid
MoE model that is a fifth of the decode step (metal_env.py). Widening the
buffers is the fix for decode, whose temporaries are small. A prefill chunk
is the opposite: two thousand tokens through forty layers allocate gigabytes
of activations, the encoder runs far ahead of the GPU, and everything an
uncompleted buffer references stays wired. Measured on Ornith-1.5-35B-A3B
(8 192-token prompt, 2 048-token chunks, wired memory at the peak):

    command-buffer limits      pacing        peak wired   decode
    MLX defaults               none          21.8 GB      14.5 ms
    400 ops / 1024 MB          none          > 26 GB      12.2 ms   <- the kernel panic regime
    400 ops / 1024 MB          every 4       23.0 GB      12.3 ms
    400 ops / 1024 MB          every 8       25.1 GB      12.3 ms
    400 ops /  512 MB          every 4       23.3 GB      12.5 ms

So a prefill -- any forward over more than one position -- evaluates its
hidden state every few layers. The eval is synchronous: the encoder waits for
the GPU, and at most that many layers' temporaries exist at once, whatever
the buffer limits allow. Decode (one position) never enters this path. The
cost is a few dozen host round trips per chunk, about 12% of prefill
throughput on the 27B-class models measured, against a step that no longer
takes the machine down.

Installed by patching the text model's ``__call__`` for the families whose
layer loop is known (the Qwen 3.5 / 3.8 hybrids); a model this cannot pace is
left alone and reported.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from maxtoken.utils import init_logger

logger = init_logger(__name__)

_PATCHED: Dict[str, Any] = {}

# Forwards narrower than this run unpaced. Pacing bounds the gigabytes a
# 256-2048-token chunk keeps in flight; a speculative verify window (a few
# rows) allocates megabytes, and pacing IT inserted ten synchronous evals
# into every verify round — measured 67 ms of a 53 ms round on
# Ornith-1.5-35B+MTP, the whole gap between 44 tok/s and the model's pace.
MIN_PACED_TOKENS = 32


def pace_from_env() -> int:
    """Layers between evaluations; 0 disables pacing."""
    raw = os.environ.get("MAXTOKEN_MLX_PREFILL_PACE", "4").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 4


def _text_model(model):
    return getattr(model, "language_model", model)


def _paceable(inner) -> bool:
    return all(hasattr(inner, name) for name in ("embed_tokens", "layers", "norm", "fa_idx", "ssm_idx"))


def _make_paced_call(original, pace: int):
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    def paced_call(self, inputs, cache=None, input_embeddings=None):
        if (
            inputs.ndim < 2
            or int(inputs.shape[1]) < MIN_PACED_TOKENS
            or cache is None
            or input_embeddings is not None
        ):
            return original(self, inputs, cache=cache, input_embeddings=input_embeddings)
        h = self.embed_tokens(inputs)
        fa_mask = create_attention_mask(h, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[self.ssm_idx])
        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            h = layer(h, mask=ssm_mask if layer.is_linear else fa_mask, cache=c)
            if (i + 1) % pace == 0:
                mx.eval(h)
        return self.norm(h)

    return paced_call


def install(model, pace: int | None = None) -> int:
    """Pace prefills on ``model``'s text tower. Returns the pace in force (0 if
    the model's family is unknown or pacing is disabled)."""
    pace = pace_from_env() if pace is None else int(pace)
    if pace <= 0:
        return 0
    inner = getattr(_text_model(model), "model", None)
    if inner is None or not _paceable(inner):
        logger.info("prefill pacing: model family unknown, prefill left unpaced")
        return 0
    cls = type(inner)
    if "original" not in _PATCHED:
        _PATCHED["original"] = cls.__call__
        _PATCHED["class"] = cls
        cls.__call__ = _make_paced_call(cls.__call__, pace)
    _PATCHED["pace"] = pace
    logger.info("prefill pacing: hidden state evaluated every %d layers during prefill", pace)
    return pace


def uninstall() -> None:
    cls = _PATCHED.pop("class", None)
    original = _PATCHED.pop("original", None)
    if cls is not None and original is not None:
        cls.__call__ = original
    _PATCHED.clear()
