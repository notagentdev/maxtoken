"""Chunked prefill must cover the prompt exactly once, store or no store.

The chunk boundary exists so hybrid models get restore points for the prefix
cache. Its advance was tied to the store actually being there, so with the
store off (``--cache-type naive``) the boundary stopped moving after the first
one, the next chunk clamped to zero tokens, and the model was handed an empty
array — a crash on every prompt longer than BOUNDARY_TOKENS.
"""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core", reason="prefill builds mx arrays")

from maxtoken.mlx_backend.prefix_cache import BOUNDARY_TOKENS  # noqa: E402
from maxtoken.mlx_backend.worker import MlxScheduler  # noqa: E402


class _RecordingModel:
    """Stands in for the trunk: records the width of every chunk it is fed."""

    def __init__(self):
        self.widths = []

    def __call__(self, tokens, cache=None):
        width = int(tokens.shape[1])
        assert width > 0, "prefill fed the model an empty chunk"
        self.widths.append(width)
        return mx.zeros((1, width, 8))


def _sched(model):
    s = MlxScheduler.__new__(MlxScheduler)
    s._mx = mx
    s.model = model
    s.prefix_store = None
    return s


@pytest.mark.parametrize(
    "prompt_len",
    [
        BOUNDARY_TOKENS // 2,
        BOUNDARY_TOKENS + 1,
        BOUNDARY_TOKENS * 3 + 7,
    ],
)
def test_prefill_covers_the_prompt_without_a_prefix_store(prompt_len):
    model = _RecordingModel()
    ids = list(range(prompt_len))
    _sched(model)._prefill_into(None, ids, 0)
    # Everything but the last token, which the first decode step consumes.
    assert sum(model.widths) == prompt_len - 1


def test_hidden_spans_tile_the_prompt():
    """The MTP head's history is built from these callbacks, so the spans have
    to tile the prompt with no gap and no overlap — a gap would pair a hidden
    state with the wrong token for the rest of the request."""
    text = SimpleNamespace(
        model=lambda tokens, cache=None: mx.zeros((1, int(tokens.shape[1]), 8))
    )
    s = _sched(_RecordingModel())
    s.model = SimpleNamespace(language_model=text)
    ids = list(range(BOUNDARY_TOKENS * 2 + 5))

    spans = []
    s._prefill_into(None, ids, 0, on_hidden=lambda h, a, b: spans.append((a, b)))

    assert spans[0][0] == 0
    assert spans[-1][1] == len(ids) - 1
    for (_, previous_end), (start, _) in zip(spans, spans[1:]):
        assert start == previous_end
