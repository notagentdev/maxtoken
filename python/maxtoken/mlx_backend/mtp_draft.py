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


def sidecar_path(ref: str) -> str | None:
    """Resolve ``--draft-model <path>`` to an MTP sidecar, or None.

    Some checkpoints ship WITHOUT their MTP head while a sibling artifact of
    the same base model carries it (ornith-ai's 4-bit MLX conversion has no
    ``mtp.*`` tensors; the MTPLX artifact extracts them from the shared base
    unchanged). Pointing ``--draft-model`` at that file or directory drafts
    with the native head instead of a second model. A file qualifies by its
    tensor names — every name in the ``mtp.`` namespace — so an arbitrary
    two-model draft checkpoint never takes this path by accident.
    """
    import json
    import struct

    path = os.path.expanduser(str(ref))
    if os.path.isdir(path):
        return find_mtp_weights(path)
    if not (os.path.isfile(path) and path.endswith(".safetensors")):
        return None
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(min(n, 1 << 24)))
    except (OSError, ValueError):
        return None
    names = [k for k in header if k != "__metadata__"]
    if names and all(k.startswith("mtp.") for k in names):
        return path
    return None


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
    def load(
        cls, model, model_dir: str, k: int, weights_path: str | None = None
    ) -> "MtpDrafter":
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_unflatten

        # model_dir stays the TRUNK's directory either way — the head's
        # quantization scheme is read from the trunk's config; only the
        # weights may come from a sibling artifact (see sidecar_path).
        weights_path = weights_path or find_mtp_weights(model_dir)
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
        raw = mx.load(weights_path)
        renamed = {
            k2[len("mtp.") :] if k2.startswith("mtp.") else k2: v
            for k2, v in raw.items()
        }
        # A MoE trunk's head is a MoE block; its sidecar may ship the experts
        # one by one (transformers-4 layout) or as fused stacks (transformers-5
        # `experts.gate_up_proj` / `experts.down_proj`, the layout replacement
        # heads such as shisa-ai's are exported in) while mlx-lm's SwitchGLU
        # owns one stacked array per projection.
        renamed = _split_fused_experts(_stack_numbered_experts(renamed))
        norm_spec = os.environ.get("MAXTOKEN_MLX_MTP_NORM_OFFSET", "")
        renamed = _offset_norms(renamed, norm_spec)
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        quant = cfg.get("quantization")
        prequantized = any(k.endswith(".scales") for k in renamed)
        if quant and prequantized:
            # The head ships quantized — but NOT necessarily in the trunk's
            # scheme: MTPLX packs every head at INT4/g64 and records that in
            # `mtplx_mtp_quantization`, while the trunk may be g32 with 8-bit
            # attention (Qwen3.8-27B "Optimized-Speed"). Build the quantized
            # modules BEFORE loading so the shapes line up. A bf16 sidecar
            # beside a quantized trunk (Ornith-1.5-35B) skips this and runs
            # in bf16 — the precision its acceptance was validated at.
            scheme = head_quant_scheme(cfg, renamed)
            nn.quantize(
                head,
                group_size=scheme["group_size"],
                bits=scheme["bits"],
                mode=scheme["mode"],
                # Only the projections carry weights to quantize; the norms
                # (and any name that merely CONTAINS "norm") must be skipped by
                # type, not by name — pre_fc_norm_embedding is an RMSNorm.
                class_predicate=lambda _path, m: isinstance(m, nn.Linear),
            )
        head.update(tree_unflatten(list(renamed.items())))
        if quant and not prequantized and os.environ.get(
            "MAXTOKEN_MLX_MTP_HEAD_QUANT", ""
        ) == "1":
            # Opt-in only: quantizing the bf16 head into the trunk's scheme
            # saves ~1.2 GB but MEASURED SLOWER end to end on Ornith-1.5-35B
            # (acceptance 2.37 -> 2.27 tok/verify at k=2, 82 -> 76 tok/s;
            # the head's own GPU cost was already negligible). The drafter's
            # quality is worth more than its bytes. Routers keep 8 bits like
            # the trunk's quant_predicate.
            def _predicate(path, m):
                if not hasattr(m, "to_quantized"):
                    return False
                if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                    return {"group_size": 64, "bits": 8}
                return True

            nn.quantize(
                head,
                group_size=int(quant.get("group_size", 64)),
                bits=int(quant.get("bits", 4)),
                mode=str(quant.get("mode", "affine")),
                class_predicate=_predicate,
            )
        mx.eval(head.parameters())
        logger.info(
            f"MTP drafter: {len(renamed)} tensors from {os.path.basename(weights_path)}, "
            f"k={k} (shares the trunk's lm_head and tokenizer)"
        )
        _ = args
        drafter = cls(head, text, k)
        if not norm_spec and os.environ.get("MAXTOKEN_MLX_MTP_NORM_CALIBRATE", "1") != "0":
            drafter.calibrate_norms(renamed)
        return drafter

    # -- norm convention ------------------------------------------------------

    def calibrate_norms(self, weights: dict, probe_len: int = 64) -> frozenset:
        """Pick the RMSNorm convention the sidecar was written in, by measurement.

        Qwen3.5 / Qwen3-Next store every MTP RMSNorm weight zero-centred (the
        module applies ``1 + w``); mlx-lm's trunk sanitize restores ``+1`` but
        the head loads separately. Exports differ in which tensors were
        restored: vLLM/transformers-5 heads (shisa-ai) ship all seven raw,
        MTPLX artifacts restore q/k/norm always and the other four only when
        their raw mean is below 0.5 — which left Ornith-1.5's
        post_attention_layernorm (raw mean 0.87) un-restored in the shipped
        sidecar. Weight statistics cannot tell a restored 0.87 from a raw
        0.87, so the loader asks the model instead: the trunk greedily writes
        a short sequence, and every candidate convention is scored by how
        well the head predicts the trunk's own next tokens from the trunk's
        hidden states. Measured on Ornith-1.5-35B (k=2, checkpoint sampler):
        as shipped 2.35 tok/verify on AIME, with post_attention_layernorm
        restored 2.66 (91-96 -> 109-118 tok/s); the wrong convention scores
        near zero agreement, so the pick is unambiguous.

        Returns the set of norm leaves that were shifted by +1 (applied)."""
        mx = self._mx
        from mlx.utils import tree_unflatten

        norms = {k: v for k, v in weights.items() if _norm_leaf(k) is not None}
        if not norms:
            return frozenset()
        try:
            tokens, hiddens = self._probe_sequence(probe_len)
        except Exception as e:  # pragma: no cover - never block serving on the probe
            logger.warning(f"MTP head norm calibration skipped: {e!r}")
            return frozenset()
        results = []
        for shift in _norm_candidates():
            self.head.update(tree_unflatten([
                (k, (v + 1) if _norm_leaf(k) in shift else v) for k, v in norms.items()
            ]))
            results.append((self._probe_score(tokens, hiddens), shift))
        results.sort(key=lambda r: r[0][0], reverse=True)
        (best_lp, best_hit), best = results[0]
        shipped = next(r[0] for r in results if not r[1])
        self.head.update(tree_unflatten([
            (k, (v + 1) if _norm_leaf(k) in best else v) for k, v in norms.items()
        ]))
        mx.eval(self.head.parameters())
        self.start([])
        (run_lp, run_hit), runner = results[1]
        logger.info(
            "MTP head norm convention: +1 on %s (probe over %d tokens: mean logp %.2f, "
            "argmax %.0f%%; as shipped %.2f / %.0f%%; runner-up +1 on %s %.2f / %.0f%%)",
            sorted(best) or "nothing", len(tokens) - 2, best_lp, 100 * best_hit,
            shipped[0], 100 * shipped[1], sorted(runner) or "nothing", run_lp, 100 * run_hit,
        )
        return best

    def _probe_sequence(self, n: int):
        """Let the trunk greedily write ``n+2`` tokens from a fixed seed and keep
        the hidden state that produced each one — tokenizer-free, deterministic."""
        mx = self._mx
        from mlx_lm.models.cache import make_prompt_cache

        text = self.trunk
        cache = make_prompt_cache(text)
        tok = 0
        tokens, hiddens = [tok], []
        for _ in range(n + 1):
            hidden = text.model(mx.array([[tok]]), cache=cache)
            logits = _lm_head(text, hidden)
            tok = int(mx.argmax(logits[0, -1]).item())
            hiddens.append(hidden)
            tokens.append(tok)
        return tokens, hiddens

    def _probe_score(self, tokens, hiddens):
        """(mean log-prob, argmax agreement) of the head on the trunk's tokens:
        hidden_i with token_{i+1} must predict token_{i+2}, the chain the
        decode rounds use."""
        mx = self._mx
        self.start(tokens[:1])
        lp, hits, n = 0.0, 0, len(tokens) - 2
        for i in range(n):
            logits, _ = self._step(hiddens[i], mx.array([[tokens[i + 1]]]))
            row = logits[0, -1].astype(mx.float32)
            target = tokens[i + 2]
            lp += float((row[target] - mx.logsumexp(row)).item())
            hits += int(mx.argmax(row).item() == target)
        return lp / max(n, 1), hits / max(n, 1)

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
        # Evaluate the head's cache NOW, chunk by chunk. _append only builds
        # the graph; left lazy, every prefill chunk's head forward — and the
        # trunk hidden states it references — accumulates unevaluated until
        # the first draft materializes them all at once. On long prompts that
        # in-flight pile wedged the machine (free memory 0, GPU event never
        # signalling). One eval per chunk bounds it to a chunk's worth.
        self._mx.eval(*self.cache[0].state)

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


def head_quant_scheme(cfg: dict, weights: dict) -> dict:
    """The (bits, group_size, mode) a prequantized MTP head was packed with.

    Precedence: the head's own record (``mtplx_mtp_quantization``, written by
    MTPLX for every packed head), then the contract's group/mode with the
    trunk's bits, then the trunk's ``quantization`` block. Whatever the
    metadata says is checked against the packed shapes: a 4-bit weight of
    ``(out, in/8)`` uint32 next to ``(out, in/group)`` scales pins the group
    size exactly, and the tensors win over the config when they disagree —
    the config is what a converter *meant*, the shapes are what it did."""
    trunk = cfg.get("quantization") or {}
    own = cfg.get("mtplx_mtp_quantization") or {}
    contract = cfg.get("mtplx_mtp_contract") or {}
    scheme = {
        "bits": int(own.get("bits", trunk.get("bits", 4))),
        "group_size": int(
            own.get("group_size", contract.get("mtp_quant_group_size", trunk.get("group_size", 64)))
        ),
        "mode": str(own.get("mode", contract.get("mtp_quant_mode", trunk.get("mode", "affine")))),
    }
    for name, w in weights.items():
        if not name.endswith(".weight") or name[: -len(".weight")] + ".scales" not in weights:
            continue
        scales = weights[name[: -len(".weight")] + ".scales"]
        if getattr(w, "ndim", 0) != 2 or getattr(scales, "ndim", 0) != 2:
            continue
        in_features = w.shape[1] * 32 // scheme["bits"]
        if in_features % scales.shape[1]:
            continue
        group = in_features // scales.shape[1]
        if group != scheme["group_size"]:
            logger.warning(
                f"MTP head: config says group_size={scheme['group_size']} but "
                f"{name} is packed at {group} (weight {tuple(w.shape)}, scales "
                f"{tuple(scales.shape)}, {scheme['bits']}-bit); using {group}"
            )
            scheme["group_size"] = group
        break
    return scheme


def _stack_numbered_experts(weights: dict) -> dict:
    """Fold per-expert tensors into mlx-lm's stacked switch_mlp layout.

    A MoE MTP sidecar stores its block the way the source checkpoint does —
    ``layers.0.mlp.experts.<i>.gate_proj.weight`` — while mlx-lm's SwitchGLU
    owns ONE stacked array per projection,
    ``layers.0.mlp.switch_mlp.gate_proj.weight``. Ordinary keys pass through
    untouched, and a group only folds when every expert index is present;
    an incomplete group keeps its original names so ``update`` fails loudly
    on the mismatch instead of silently serving a truncated expert table.
    """
    import re

    import mlx.core as mx

    pattern = re.compile(r"^(.*)\.experts\.(\d+)\.(\w+)\.(weight|scales|biases)$")
    grouped: dict = {}
    out: dict = {}
    for key, value in weights.items():
        m = pattern.match(key)
        if m is None:
            out[key] = value
            continue
        prefix, idx, leaf, kind = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        grouped.setdefault((prefix, leaf, kind), {})[idx] = value
    for (prefix, leaf, kind), experts in sorted(grouped.items()):
        n = max(experts) + 1
        if set(experts) != set(range(n)):
            for i, v in experts.items():
                out[f"{prefix}.experts.{i}.{leaf}.{kind}"] = v
            continue
        out[f"{prefix}.switch_mlp.{leaf}.{kind}"] = mx.stack(
            [experts[i] for i in range(n)]
        )
    return out


# The seven RMSNorm weights of a Qwen3.5-family MTP block. MTPLX restores the
# first group unconditionally and the second only when the raw mean is < 0.5.
_NORM_ALWAYS = ("q_norm", "k_norm", "norm")
_NORM_LOW_SET = ("input_layernorm", "post_attention_layernorm",
                 "pre_fc_norm_embedding", "pre_fc_norm_hidden")


def _norm_leaf(key: str) -> str | None:
    """The RMSNorm module name a 1-D ``*.weight`` key belongs to, or None."""
    if not key.endswith(".weight"):
        return None
    leaf = key[: -len(".weight")].rsplit(".", 1)[-1]
    return leaf if leaf in _NORM_ALWAYS or leaf in _NORM_LOW_SET else None


def _norm_candidates():
    """Conventions worth scoring: as shipped, every subset of the four
    conditionally-restored norms (an MTPLX artifact may have missed any of
    them), and the fully raw export (all seven)."""
    from itertools import combinations

    yield frozenset()
    for r in range(1, len(_NORM_LOW_SET) + 1):
        for combo in combinations(_NORM_LOW_SET, r):
            yield frozenset(combo)
    yield frozenset(_NORM_LOW_SET + _NORM_ALWAYS)


def _offset_norms(weights: dict, spec: str) -> dict:
    """Explicit override of the calibration (MAXTOKEN_MLX_MTP_NORM_OFFSET):
    "" = calibrate, "all", "except:a,b", "only:a,b" — names are the RMSNorm
    module leaves, e.g. post_attention_layernorm. Adds +1 to the named ones."""
    if not spec:
        return weights
    mode, _, names = spec.partition(":")
    names = {n for n in names.split(",") if n}
    out = dict(weights)
    for k, v in weights.items():
        leaf = _norm_leaf(k)
        if leaf is None or getattr(v, "ndim", 0) != 1:
            continue
        if mode == "all" or (mode == "except" and leaf not in names) or (mode == "only" and leaf in names):
            out[k] = v + 1
            logger.info(f"MTP head norm offset +1: {k}")
    return out


def _split_fused_experts(weights: dict) -> dict:
    """Unfuse transformers-5 stacked experts into mlx-lm's switch_mlp layout.

    ``<prefix>.experts.gate_up_proj`` is ``[E, 2*inter, hidden]`` with the gate
    rows first (mlx-lm's qwen3_5_moe sanitize splits it the same way) and
    ``<prefix>.experts.down_proj`` is ``[E, hidden, inter]``. A gate_up stack
    without its down stack is left untouched so ``update`` reports the
    missing tensor instead of a half-mapped block.
    """
    out = dict(weights)
    for key in list(out):
        if not key.endswith(".experts.gate_up_proj"):
            continue
        prefix = key[: -len(".experts.gate_up_proj")]
        down_key = f"{prefix}.experts.down_proj"
        if down_key not in out:
            continue
        gate_up = out.pop(key)
        mid = gate_up.shape[-2] // 2
        out[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_up[..., :mid, :]
        out[f"{prefix}.switch_mlp.up_proj.weight"] = gate_up[..., mid:, :]
        out[f"{prefix}.switch_mlp.down_proj.weight"] = out.pop(down_key)
    return out


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
