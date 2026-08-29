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


def test_a_lone_prefill_uses_wide_chunks_that_end_on_boundaries():
    """Alone, the prefill takes 2048-token chunks (7% faster on the 27B, and
    the fp16 GEMM amortizes its dequantized matrices); with another request
    active it narrows to 512 so the other one gets its turns. Either way a
    chunk that is not the last ends on a BOUNDARY_TOKENS multiple -- the
    prefix store's restore points."""
    model = _RecordingModel()
    s = _sched(model)
    s.active = {}
    ids = list(range(5000))
    s._prefill_into(None, ids, 0)
    assert model.widths == [2048, 2048, 903]
    ends = [sum(model.widths[: i + 1]) for i in range(len(model.widths) - 1)]
    assert all(e % BOUNDARY_TOKENS == 0 for e in ends)

    busy = _RecordingModel()
    s = _sched(busy)
    s.active = {1: object(), 2: object()}
    s._prefill_into(None, ids, 0)
    assert busy.widths[0] == 512 and sum(busy.widths) == 4999
    ends = [sum(busy.widths[: i + 1]) for i in range(len(busy.widths) - 1)]
    assert all(e % BOUNDARY_TOKENS == 0 for e in ends)

    # A prefix hit that starts off the grid reaches the next boundary first.
    off = _RecordingModel()
    s = _sched(off)
    s.active = {}
    s._prefill_into(None, ids, 300)
    assert (300 + off.widths[0]) % BOUNDARY_TOKENS == 0
    assert 300 + sum(off.widths) == 4999
