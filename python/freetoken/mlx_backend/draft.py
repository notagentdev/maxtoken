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

from freetoken.utils import init_logger

logger = init_logger(__name__)


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

    @classmethod
    def load(cls, model_path: str, k: int) -> "DraftModel":
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache

        model, _tokenizer = load(model_path)
        probe = make_prompt_cache(model)
        if not all(c.is_trimmable() for c in probe):
            raise ValueError(
                f"draft model {model_path} has non-trimmable caches; speculative "
                "bookkeeping needs trim() (use a plain-attention draft model)"
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

    def draft(self) -> List[int]:
        """Propose k greedy tokens continuing the committed path."""
        mx = self._mx
        drafts: List[int] = []
        feed = self._pending
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
        return drafts

    def commit(self, accepted: int, tail: List[int]) -> None:
        """Reconcile after verification.

        ``accepted``: how many drafts the target confirmed (0..k).
        ``tail``: committed-path tokens the draft cache has not seen — the
        correction token on partial accept, or [d_k, bonus] on full accept.
        """
        overshoot = self._drafted - accepted
        if overshoot > 0:
            for c in self.cache:
                c.trim(overshoot)
        self._drafted = 0
        self._pending.extend(tail)
