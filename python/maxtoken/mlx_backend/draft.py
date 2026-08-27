"""Draft model for speculative decoding on the MLX offload path.

A small same-vocabulary model proposes ``k`` greedy continuations; the target
verifies all of them in ONE batched forward through the expert slot cache and
commits the matched prefix plus one target token (correction or bonus). The
target's output distribution is exactly preserved: every committed token is the
target's own sample for its position — drafts only decide how many positions one
expensive forward may advance.

The economics on the offload path are better than classic speculative decoding:
the verify window's routed-expert union grows sublinearly in window size, so one
expert-load round trip serves several tokens.
"""

from __future__ import annotations

from typing import Any, List

from maxtoken.utils import init_logger

logger = init_logger(__name__)


def _vocab_size(model) -> int | None:
    """Vocabulary of a loaded mlx-lm model, from its args or its embedding."""
    args = getattr(model, "args", None)
    for obj in (args, getattr(args, "text_config", None)):
        v = getattr(obj, "vocab_size", None)
        if v:
            return int(v)
    try:
        return int(model.model.embed_tokens.weight.shape[0])
    except Exception:  # noqa: BLE001 -- unknown layout: skip the check
        return None


class DraftModel:
    """Greedy drafter with its own KV cache, kept in lockstep with the target.

    Position bookkeeping: ``_pending`` holds committed-path tokens the draft
    cache has not consumed yet. ``draft()`` feeds them plus k-1 of its own
    drafts; ``commit()`` trims speculated-but-rejected tokens away again. The
    cache must be trimmable (any dense-attention model is), which ``load``
    asserts once.
    """

    def __init__(self, model: Any, k: int):
        import mlx.core as mx

        self._mx = mx
        self.model = model
        self.k = max(1, int(k))
        self.cache: List[Any] | None = None
        self._pending: List[int] = []
        self._drafted: int = 0  # draft tokens currently in the cache
        self._drafted_tokens: List[int] = []
        self._snaps: List[Any] = []
        self._fed: List[int] = []

    @classmethod
    def load(cls, model_path: str, k: int, target_vocab: int | None = None) -> "DraftModel":
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache

        model, _tokenizer = load(model_path)
        vocab = _vocab_size(model)
        if target_vocab and vocab and vocab != target_vocab:
            # Token ids are only comparable within one vocabulary. A mismatched
            # drafter proposes ids that mean something else to the target (so
            # nothing is ever accepted), and committed target ids fed back into
            # its cache can index past its embedding table entirely.
            raise ValueError(
                f"draft model {model_path} has vocab {vocab}, target has "
                f"{target_vocab}: speculative decoding needs the same tokenizer"
            )
        return cls(model, k)

    def start(self, input_ids: List[int]) -> None:
        """New request: prefill the draft cache on everything but the last token
        (which stays pending, mirroring the target's decode entry point)."""
        from mlx_lm.models.cache import make_prompt_cache

        mx = self._mx
        self.cache = make_prompt_cache(self.model)
        self._drafted = 0
        head, tail = input_ids[:-1], input_ids[-1:]
        pos = 0
        while pos < len(head):
            chunk = head[pos : pos + 2048]
            mx.eval(self.model(mx.array(chunk)[None], cache=self.cache))
            pos += len(chunk)
        self._pending = list(tail)

    @staticmethod
    def _snapshot(c):
        """Rewind point for one draft cache. Plain KV caches rewind by trimming
        (no arrays held, so the engine keeps updating in place); recurrent and
        window caches need their arrays AND their position (offset/index lives
        in meta_state) — the same rule the target's caches follow."""
        if c.is_trimmable() and not hasattr(c, "max_size"):
            return None
        return (list(c.state), c.meta_state)

    @staticmethod
    def _restore(c, snap, n: int) -> None:
        if snap is None:
            c.trim(n)
        else:
            state, meta = snap
            c.state = state
            c.meta_state = meta

    def draft(self) -> List[int]:
        """Propose k greedy tokens continuing the committed path."""
        mx = self._mx
        drafts: List[int] = []
        feed = self._pending
        # A hybrid drafter (GDN/SSM state, sliding windows) cannot rewind by
        # trimming, so snapshot before this round consumes anything and rewind
        # to here in commit(); the accepted path is then re-fed. Costs one extra
        # pass over a handful of tokens on the SMALL model — the drafter is
        # cheap by construction, and it lets any architecture draft.
        self._snaps = [self._snapshot(c) for c in self.cache]
        self._fed = list(feed)
        self._pending = []
        for _ in range(self.k):
            logits = self.model(mx.array(feed)[None], cache=self.cache)
            # Host sync per draft: keeps each lazy graph tiny. The batched
            # alternative (defer all argmaxes) measures slower (Vates A/B).
            tok = int(mx.argmax(logits[0, -1]))
            drafts.append(tok)
            feed = [tok]
        # The last draft was never fed; the cache holds pending + drafts[:-1].
        self._drafted = self.k - 1
        self._drafted_tokens = drafts
        return drafts

    def commit(self, accepted: int, tail: List[int]) -> None:
        """Reconcile after verification.

        ``accepted``: how many drafts the target confirmed (0..k).
        ``tail``: committed-path tokens the draft cache has not seen — the
        correction token on partial accept, or [d_k, bonus] on full accept.
        """
        if any(snap is not None for snap in self._snaps):
            # Non-trimmable (hybrid/window) caches: rewind to before this round
            # and re-pend everything the cache therefore no longer holds — the
            # tokens it was fed plus the accepted drafts plus the tail.
            for c, snap in zip(self.cache, self._snaps, strict=True):
                self._restore(c, snap, self._drafted)
            self._pending = self._fed + self._drafted_tokens[:accepted] + list(tail)
        else:
            overshoot = self._drafted - accepted
            if overshoot > 0:
                for c in self.cache:
                    c.trim(overshoot)
            self._pending.extend(tail)
        self._drafted = 0
