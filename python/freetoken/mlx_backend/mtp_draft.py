"""Speculative drafting from a model's own MTP (multi-token prediction) head.

Several recent checkpoints ship a NextN predictor next to the trunk — a single
decoder block that, given the trunk's last hidden state and the token just
committed, predicts the token after it. Compared with a second model it is the
right shape for the job by construction:

* it is ONE layer (228 MB on Qwen3.8-27B, ~10x smaller than the smallest
  compatible standalone drafter on that family),
* it shares the trunk's tokenizer and ``lm_head``, so no vocabulary can drift
  and no ids can fall out of range,
* it was trained on this exact model, so acceptance is far above what a foreign
  model guesses.

Measuring two-model speculation on this hardware produced only losses (see
docs/mlx.md); this is the variant the technique was designed around.

Layout (the ``mtp.*`` namespace, typically a separate ``mtp.safetensors``)::

    mtp.pre_fc_norm_embedding   RMSNorm over the next token's embedding
    mtp.pre_fc_norm_hidden      RMSNorm over the trunk's hidden state
    mtp.fc                      Linear concat[e, h] (2H -> H)
    mtp.layers.0                one full-attention decoder block
    mtp.norm                    RMSNorm before the shared lm_head

The block is exactly the layer class the trunk's own module builds for a
full-attention position, so it is reused rather than reimplemented.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, List

from freetoken.utils import init_logger

logger = init_logger(__name__)


def find_mtp_weights(model_dir: str) -> str | None:
    """Path of the checkpoint's MTP head, or None if it ships without one."""
    for name in ("mtp.safetensors", "model-mtp-head.safetensors"):
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            return path
    hits = sorted(glob.glob(os.path.join(model_dir, "*mtp*.safetensors")))
    return hits[0] if hits else None


class MtpDrafter:
    """Drafts with the trunk's own MTP head. Interface-compatible with
    ``DraftModel`` so the scheduler's verify loop is unchanged, with one
    addition: ``draft`` needs the trunk's last hidden state."""

    def __init__(self, head, trunk, k: int):
        import mlx.core as mx

        self._mx = mx
        self.head = head
        self.trunk = trunk  # for embed_tokens + lm_head (shared with the trunk)
        self.k = max(1, int(k))
        self.cache: List[Any] | None = None
        self._pending: List[int] = []
        self._hidden = None

    # -- construction ---------------------------------------------------------

    @classmethod
    def load(cls, model, model_dir: str, k: int) -> "MtpDrafter":
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_unflatten

        weights_path = find_mtp_weights(model_dir)
        if weights_path is None:
            raise ValueError(
                f"{model_dir} ships no MTP head (looked for mtp.safetensors / "
                "model-mtp-head.safetensors); use --draft-model <path> for "
                "two-model speculation instead"
            )
        text = _text_model(model)
        args = getattr(text, "args", None) or getattr(model, "args", None)
        layer_cls, text_args = _decoder_layer_class(model)

        head = _MtpHead(layer_cls, text_args)
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        quant = cfg.get("quantization")
        if quant:
            # The head ships quantized in the trunk's own scheme; build the
            # quantized modules BEFORE loading so the shapes line up.
            nn.quantize(
                head,
                group_size=int(quant.get("group_size", 64)),
                bits=int(quant.get("bits", 4)),
                mode=str(quant.get("mode", "affine")),
                # Only the projections carry weights to quantize; the norms
                # (and any name that merely CONTAINS "norm") must be skipped by
                # type, not by name — pre_fc_norm_embedding is an RMSNorm.
                class_predicate=lambda _path, m: isinstance(m, nn.Linear),
            )
        raw = mx.load(weights_path)
        renamed = {
            k2[len("mtp.") :] if k2.startswith("mtp.") else k2: v
            for k2, v in raw.items()
        }
        head.update(tree_unflatten(list(renamed.items())))
        mx.eval(head.parameters())
        logger.info(
            f"MTP drafter: {len(renamed)} tensors from {os.path.basename(weights_path)}, "
            f"k={k} (shares the trunk's lm_head and tokenizer)"
        )
        _ = args
        return cls(head, text, k)

    # -- drafting -------------------------------------------------------------

    def make_cache(self):
        from mlx_lm.models.cache import KVCache

        return [KVCache()]

    def start(self, input_ids: List[int]) -> None:
        """New request. The head only ever sees single tokens plus the trunk's
        hidden state, so there is nothing to prefill — just reset its cache."""
        self.cache = self.make_cache()
        self._pending = list(input_ids[-1:])
        self._hidden = None

    def set_hidden(self, hidden) -> None:
        """The trunk's hidden state at the last committed position."""
        self._hidden = hidden

    def draft(self) -> List[int]:
        mx = self._mx
        if self._hidden is None:
            return []
        drafts: List[int] = []
        h = self._hidden
        tok = mx.array([self._pending[-1:]])
        for _ in range(self.k):
            logits, h = self._step(h, tok)
            nxt = int(mx.argmax(logits[0, -1]))
            drafts.append(nxt)
            tok = mx.array([[nxt]])
        return drafts

    def _step(self, hidden, token):
        """One MTP position: fuse (embedding of the next token, trunk hidden),
        run the block, and score with the trunk's own head."""
        mx = self._mx
        head = self.head
        emb = head.pre_fc_norm_embedding(self.trunk.model.embed_tokens(token))
        hid = head.pre_fc_norm_hidden(hidden)
        x = head.fc(mx.concatenate([emb, hid], axis=-1))
        mask = _causal_mask(x, self.cache)
        x = head.layers[0](x, mask=mask, cache=self.cache[0])
        out = head.norm(x)
        # The chained hidden is the NORMED block output, not the raw one: from
        # the second draft position on, the head consumes its own state instead
        # of the trunk's, and it was trained against the normed convention.
        # Feeding the pre-norm activation drifts after position 1 (measured:
        # acceptance 1/3 instead of 3/3 on a structured prompt).
        return _lm_head(self.trunk, out), out

    def commit(self, accepted: int, tail: List[int]) -> None:
        """The head's cache advanced over drafts that may not have survived.
        Rebuilding it costs one block forward per committed token — cheap for a
        single layer — so it is simply reset and re-primed from the trunk's next
        hidden state."""
        self.cache = self.make_cache()
        self._pending.extend(tail)
        self._hidden = None


class _MtpHead:
    """Module tree matching the ``mtp.*`` namespace."""

    def __new__(cls, layer_cls, args):
        import mlx.nn as nn

        head = nn.Module()
        head.pre_fc_norm_embedding = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        head.pre_fc_norm_hidden = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        head.fc = nn.Linear(2 * args.hidden_size, args.hidden_size, bias=False)
        head.layers = [_full_attention_layer(layer_cls, args)]
        head.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        return head


def _full_attention_layer(layer_cls, args):
    """Build the decoder block at a FULL-attention position: the MTP head is
    that layer, never the trunk's linear/GDN variant."""
    interval = getattr(args, "full_attention_interval", 0) or 1
    return layer_cls(args, interval - 1)


def _text_model(model):
    """The text tower: multimodal wrappers nest it under language_model."""
    return getattr(model, "language_model", model)


def _decoder_layer_class(model):
    """(DecoderLayer class, text args) for the loaded model's family."""
    import importlib

    text = _text_model(model)
    module = importlib.import_module(type(text).__module__)
    for name in ("DecoderLayer", "TransformerBlock", "Qwen3NextDecoderLayer"):
        cls = getattr(module, name, None)
        if cls is not None:
            return cls, text.model.layers[0].args if hasattr(
                text.model.layers[0], "args"
            ) else _args_of(text)
    raise ValueError(f"no decoder layer class found in {module.__name__}")


def _args_of(text):
    args = getattr(text, "args", None)
    if args is not None:
        return args
    raise ValueError("model exposes no args to build the MTP block from")


def _causal_mask(x, cache):
    from mlx_lm.models.base import create_attention_mask

    return create_attention_mask(x, cache)


def _lm_head(text, hidden):
    head = getattr(text, "lm_head", None)
    if head is not None:
        return head(hidden)
    return text.model.embed_tokens.as_linear(hidden)
