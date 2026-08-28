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
from dataclasses import dataclass
from typing import Any, List

from maxtoken.utils import init_logger

logger = init_logger(__name__)


def find_mtp_weights(model_dir: str) -> str | None:
    """Path of the checkpoint's MTP head, or None if it ships without one.

    Publishers disagree about where the head lives. Qwen3.8-27B-MTPLX puts it
    beside the shards as ``mtp.safetensors``; Ornith-1.5-35B-MTPLX puts it in
    its own directory as ``mtp/weights.safetensors``. Both are checked by name
    before falling back to a glob, and the glob searches one level down as well
    — a head in a subdirectory is otherwise invisible and the drafter refuses a
    checkpoint that plainly ships one.
    """
    for name in (
        "mtp.safetensors",
        "model-mtp-head.safetensors",
        os.path.join("mtp", "weights.safetensors"),
    ):
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            return path
    hits = sorted(
        glob.glob(os.path.join(model_dir, "*mtp*.safetensors"))
        + glob.glob(os.path.join(model_dir, "*mtp*", "*.safetensors"))
    )
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
        # Entries in the head's KV cache that belong to COMMITTED tokens. Draft
        # steps append past this mark and are trimmed back to it every round.
        self._hist_len = 0
        self.q: List[Any] = []  # proposal densities, when drafting by sampling

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
        """New request: a fresh, empty history for the head."""
        self.cache = self.make_cache()
        self._pending = list(input_ids[-1:])
        self._hidden = None
        self._hist_len = 0

    def set_hidden(self, hidden) -> None:
        """The trunk's hidden state at the last committed position."""
        self._hidden = hidden

    def draft(self, shape=None) -> List[int]:
        """Propose k tokens. With ``shape`` (a logits -> probability-vector
        function) the head SAMPLES from its own distribution and the proposal
        density q is recorded in ``self.q``; without it, greedy.

        Sampling matters for the acceptance rule. A greedy proposal is a point
        mass, so ``min(1, p/q)`` collapses to ``p(d)`` — the worst case, and
        exactly what plain sample-and-match already achieves. A proposal drawn
        from the head's own distribution is accepted with probability
        ``min(1, p(d)/q(d))``, which is 1 wherever the head is less confident
        than the target."""
        mx = self._mx
        if self._hidden is None:
            self.q = []
            return []
        # Last round's draft chain may still sit past the committed mark if the
        # scheduler never absorbed it; the head must never draft on top of
        # tokens that were rejected.
        self._trim_to(self._hist_len)
        drafts: List[int] = []
        self.q = []
        h = self._hidden
        tok = mx.array([self._pending[-1:]])
        for _ in range(self.k):
            logits, h = self._step(h, tok)
            if shape is None:
                nxt = int(mx.argmax(logits[0, -1]))
            else:
                probs = shape(logits[0, -1])
                nxt = int(mx.random.categorical(mx.log(probs + 1e-30)).item())
                self.q.append(probs)
            drafts.append(nxt)
            tok = mx.array([[nxt]])
        return drafts

    def draft_lazy(self, room: int, spec) -> "LazyDraft":
        """Propose up to ``min(k, room)`` tokens WITHOUT leaving the device.

        The chain is built as one lazy graph: each step's token feeds the next
        step's embedding as an array, so no draft ever waits for a host
        round trip. With ``spec`` (a ``spec_sample.SamplerSpec``) every step
        samples from its shaped distribution on the device and the proposal
        support is returned with the tokens; without it the chain is greedy.
        The caller decides when the graph runs — typically right away, with
        ``mx.async_eval``, while it builds the verify window on the CPU.
        """
        mx = self._mx
        from . import spec_sample

        n = min(self.k, int(room))
        if self._hidden is None or n <= 0:
            return LazyDraft(mx.zeros((0,), dtype=mx.int32), (), 0)
        self._trim_to(self._hist_len)
        h = self._hidden
        tok = mx.array([self._pending[-1:]], dtype=mx.int32)
        tokens, q_ids, q_probs = [], [], []
        for _ in range(n):
            logits, h = self._step(h, tok)
            row = logits[0, -1]
            if spec is None:
                nxt = mx.argmax(row).astype(mx.int32)
            else:
                nxt, ids, probs = spec_sample.sample_row(row, spec)
                q_ids.append(ids)
                q_probs.append(probs)
            tokens.append(nxt)
            tok = nxt.reshape(1, 1)
        q = (mx.stack(q_ids), mx.stack(q_probs)) if spec is not None else ()
        return LazyDraft(mx.stack(tokens), q, n)

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

    def _trim_to(self, offset: int) -> None:
        entry = self.cache[0]
        current = int(getattr(entry, "offset", 0))
        if current > offset:
            entry.trim(current - offset)

    def _append(self, hidden, token_ids: List[int]) -> None:
        """Run T history positions through the head in ONE block forward."""
        mx = self._mx
        head = self.head
        tokens = mx.array([[int(t) for t in token_ids]])
        emb = head.pre_fc_norm_embedding(self.trunk.model.embed_tokens(tokens))
        hid = head.pre_fc_norm_hidden(hidden)
        x = head.fc(mx.concatenate([emb, hid], axis=-1))
        head.layers[0](x, mask=_causal_mask(x, self.cache), cache=self.cache[0])

    def extend_history(self, hidden, token_ids: List[int]) -> None:
        """Prompt-side history: pair the trunk's hidden at position p with the
        token at p+1, which is the same pairing the decode rounds use.

        Without this the head enters every request knowing nothing, and has to
        earn its context back one committed token at a time — measured on this
        checkpoint, acceptance climbs from 2.10 tokens per round at 60 committed
        tokens to 2.43 at 480.
        """
        if not token_ids:
            return
        self._trim_to(self._hist_len)
        self._append(hidden, token_ids)
        self._hist_len += len(token_ids)

    def absorb(self, committed: List[int], hidden_rows) -> None:
        """Write this round's committed tokens into the head's own KV history.

        Without this the head starts every round from an empty cache and can
        only condition on the two or three tokens of the draft chain it is
        currently building — the reference engine keeps the full committed
        history for exactly this reason, and reports acceptance collapsing when
        that history is shortened.

        Draft entries are discarded first: they were built from the head's own
        chained state, while the committed ones are re-derived from the TARGET's
        hidden states, which is the conditioning the head was trained on. The
        LAST committed token is deliberately left out — the next ``draft`` call
        creates its entry as its first step, which is where its trunk hidden
        arrives.
        """
        self._trim_to(self._hist_len)
        n = len(committed)
        if n == 0 or hidden_rows is None or self._hidden is None:
            return
        mx = self._mx
        tokens = [int(self._pending[-1])] + [int(t) for t in committed[: n - 1]]
        hidden = mx.concatenate(
            [self._hidden, hidden_rows[:, : n - 1, :]], axis=1
        )
        self._append(hidden, tokens)
        self._hist_len += len(tokens)

    def commit(self, accepted: int, tail: List[int]) -> None:
        """Advance the committed stream. The head's cache is not rebuilt here —
        ``absorb`` already moved this round's committed tokens into it."""
        self._pending.extend(tail)
        self._hidden = None


@dataclass(frozen=True)
class LazyDraft:
    """A draft chain still on the device: ``tokens`` (``(n,)`` int32), the
    proposal supports ``q = (ids, probs)`` (each ``(n, top_k)``; empty for a
    greedy chain), and ``n``, known without evaluating anything."""

    tokens: Any
    q: tuple
    n: int


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
