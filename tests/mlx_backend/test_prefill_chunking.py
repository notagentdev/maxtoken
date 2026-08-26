"""Chunked prefill must cover the prompt exactly once, store or no store.

The chunk boundary exists so hybrid models get restore points for the prefix
cache. Its advance was tied to the store actually being there, so with the
store off (``--cache-type naive``) the boundary stopped moving after the first
one, the next chunk clamped to zero tokens, and the model was handed an empty
array — a crash on every prompt longer than BOUNDARY_TOKENS.
"""

import pytest

mx = pytest.importorskip("mlx.core", reason="prefill builds mx arrays")

from freetoken.mlx_backend.prefix_cache import BOUNDARY_TOKENS  # noqa: E402
from freetoken.mlx_backend.worker import MlxScheduler  # noqa: E402


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
