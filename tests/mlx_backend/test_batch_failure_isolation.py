"""A failed batched forward must cost the requests, never the server.

The round-robin decode path has isolated per-request failures since it was
written. The continuous-batching path had no such guard: one prompt large enough
to run Metal out of memory raised out of the step, the worker exited, and the
supervisor then stopped the whole API server — so a single oversized request
took the engine down and it stayed down.

That is reachable from any prompt on a checkpoint whose weights approach the
GPU's working set, which is why this is about every model rather than one.
"""

from types import SimpleNamespace
from typing import List

import numpy as np
import pytest

from maxtoken.core import SamplingParams
from maxtoken.message import DetokenizeMsg, ErrorReplyMsg, UserMsg
from maxtoken.mlx_backend.worker import MlxScheduler


class _ExplodingBatcher:
    """Stands in for mlx-lm's BatchGenerator when the GPU says no."""

    def __init__(self, exc: BaseException):
        self.exc = exc
        self.calls = 0

    def next(self):
        self.calls += 1
        raise self.exc

    def insert_segments(self, *a, **kw):
        return [0]


def _scheduler(batcher) -> MlxScheduler:
    s = MlxScheduler.__new__(MlxScheduler)
    s.tokenizer = SimpleNamespace(decode=lambda ids: "")
    s.eos_token_ids = frozenset({99})
    s.max_seq_len = 4096
    s.active = {}
    s.sent = []
    s._reply = lambda replies: s.sent.extend(replies)
    s.offload_state = None
    s._decode_steps = 0
    s.config = SimpleNamespace(decode_log_interval=40, max_running_req=4)
    s.prefix_store = None
    s.batch_gen = batcher
    s._batch_uid = {}
    s._our_uid = {}
    s._prefill_batch = 4
    s._prefill_step = 2048
    s._mx = SimpleNamespace(get_active_memory=lambda: 0, clear_cache=lambda: None)
    s.model = SimpleNamespace()          # _batch_admit probes it for a mapped store
    s._lookup_prefix = lambda ids: (None, 0)
    s.rebuilt = 0

    def rebuild():
        s.rebuilt += 1
        return batcher

    s._make_batch_generator = rebuild
    s._make_generator = lambda ids, sp: (iter(()), None, 0)
    return s


def _admit(s: MlxScheduler, uid: int) -> None:
    s._handle(UserMsg(uid=uid, input_ids=np.array([uid, 7, 7], dtype=np.int32),
                      sampling_params=SamplingParams(max_tokens=8)))


OOM = RuntimeError(
    "[METAL] Command buffer execution failed: Insufficient Memory "
    "(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"
)


def test_the_worker_survives_a_failed_batch():
    s = _scheduler(_ExplodingBatcher(OOM))
    _admit(s, 1)
    _admit(s, 2)

    s._step()  # must not raise

    assert s.active == {}, "the failed batch's requests are done"
    errors = [m for m in s.sent if isinstance(m, ErrorReplyMsg)]
    assert {m.uid for m in errors} == {1, 2}
    assert all("Insufficient Memory" in m.error for m in errors)


def test_the_batcher_is_rebuilt_rather_than_reused():
    """Its prompt cache is half-written after an aborted command buffer, so the
    next request would fail on state belonging to one that already died."""
    s = _scheduler(_ExplodingBatcher(OOM))
    _admit(s, 1)
    s._step()
    assert s.rebuilt == 1
    assert s._batch_uid == {} and s._our_uid == {}


def test_a_failed_shape_is_not_tried_again_at_the_same_size():
    s = _scheduler(_ExplodingBatcher(OOM))
    _admit(s, 1)
    s._step()
    assert (s._prefill_batch, s._prefill_step) == (2, 1024)
    _admit(s, 2)
    s._step()
    assert (s._prefill_batch, s._prefill_step) == (1, 512)


def test_the_reduction_stops_at_one_prompt_of_256_tokens():
    s = _scheduler(_ExplodingBatcher(OOM))
    for uid in range(1, 8):
        _admit(s, uid)
        s._step()
    assert (s._prefill_batch, s._prefill_step) == (1, 256)


def test_a_worker_with_no_way_back_still_fails_loudly():
    """If the batcher cannot be rebuilt there is nothing left to serve with, and
    a silent no-op would busy-loop on every future request."""
    s = _scheduler(_ExplodingBatcher(OOM))

    def refuse():
        raise RuntimeError("no device")

    s._make_batch_generator = refuse
    _admit(s, 1)
    with pytest.raises(RuntimeError, match="no device"):
        s._step()


def test_healthy_batches_are_untouched():
    class _Working:
        def next(self):
            return [], []

        def insert_segments(self, *a, **kw):
            return [0]

    s = _scheduler(_Working())
    _admit(s, 1)
    s._step()
    assert s.rebuilt == 0
    assert (s._prefill_batch, s._prefill_step) == (4, 2048)
    assert not [m for m in s.sent if isinstance(m, ErrorReplyMsg)]
    assert not [m for m in s.sent if isinstance(m, DetokenizeMsg)]
