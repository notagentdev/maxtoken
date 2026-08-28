"""Protocol-level tests for the MLX scheduler worker.

These run on any platform: the scheduler's message handling, termination logic and
round-robin stepping are exercised against fake generators and a fake tokenizer, so
neither mlx nor model weights are needed. Real-model execution is covered by the
smoke path in docs/mlx.md (`mt serve --model ...`).
"""

from types import SimpleNamespace
from typing import Iterator, List

import numpy as np
import pytest

from maxtoken.core import SamplingParams
from maxtoken.message import (
    AbortBackendMsg,
    BatchBackendMsg,
    CacheRebuildBackendMsg,
    CacheRebuildResultMsg,
    DetokenizeMsg,
    ErrorReplyMsg,
    ExitMsg,
    PromptAdmittedMsg,
    UserMsg,
)
from maxtoken.mlx_backend.worker import MlxScheduler, _filter_kwargs


EOS = 99


class FakeTokenizer:
    """Token id -> its decimal string; decode() joins with nothing."""

    def decode(self, ids: List[int]) -> str:
        return "".join(str(i) for i in ids)


def make_scheduler(script: dict[int, List[int]] | None = None) -> MlxScheduler:
    """A scheduler with everything __init__ would set, minus zmq and mlx.

    ``script`` maps uid -> token sequence; _make_generator serves from it so tests
    control exactly what "the model" produces.
    """
    sched = MlxScheduler.__new__(MlxScheduler)
    sched.tokenizer = FakeTokenizer()
    sched.eos_token_ids = frozenset({EOS})
    sched.max_seq_len = 64
    sched.active = {}
    sched._mx = SimpleNamespace(get_active_memory=lambda: 0)
    sched.sent = []
    sched._reply = lambda replies: sched.sent.extend(replies)
    sched.offload_state = None
    sched._decode_steps = 0
    sched.config = SimpleNamespace(decode_log_interval=40)
    sched.prefix_store = None
    sched.batch_gen = None
    sched._batch_uid = {}
    sched._our_uid = {}
    script = script or {}

    def fake_make_generator(input_ids: List[int], sp: SamplingParams):
        uid = input_ids[0]  # tests encode the uid as the first prompt token
        return iter((t, None) for t in script.get(uid, [])), None, 0

    sched._make_generator = fake_make_generator
    return sched


def user_msg(uid: int, prompt_len: int = 4, **sp) -> UserMsg:
    ids = [uid] + [7] * (prompt_len - 1)
    return UserMsg(
        uid=uid,
        input_ids=np.array(ids, dtype=np.int32),
        sampling_params=SamplingParams(**sp),
    )


def sent_for(sched: MlxScheduler, uid: int) -> List[DetokenizeMsg]:
    return [m for m in sched.sent if isinstance(m, DetokenizeMsg) and m.uid == uid]


def test_admission_reply_carries_prompt_tokens():
    sched = make_scheduler({1: [5, EOS]})
    replies, do_exit = sched._handle(user_msg(1, prompt_len=6, max_tokens=8))
    assert not do_exit
    assert isinstance(replies[0], PromptAdmittedMsg)
    assert replies[0].prompt_tokens == 6
    assert 1 in sched.active


def test_eos_finishes_with_stop():
    sched = make_scheduler({1: [5, 6, EOS, 7]})
    sched._handle(user_msg(1, max_tokens=10))
    while sched.active:
        sched._step()
    msgs = sent_for(sched, 1)
    assert [m.next_token for m in msgs] == [5, 6, EOS]
    assert [m.finished for m in msgs] == [False, False, True]
    assert msgs[-1].finish_reason == "stop"
    assert msgs[-1].matched_stop is None


def test_max_tokens_finishes_with_length():
    sched = make_scheduler({1: [5, 5, 5, 5, 5]})
    sched._handle(user_msg(1, max_tokens=3))
    while sched.active:
        sched._step()
    msgs = sent_for(sched, 1)
    assert len(msgs) == 3
    assert msgs[-1].finished and msgs[-1].finish_reason == "length"


def test_stop_string_sets_matched_stop():
    sched = make_scheduler({1: [1, 2, 3, 4]})
    sched._handle(user_msg(1, max_tokens=10, stop_strs=["23"]))
    while sched.active:
        sched._step()
    msgs = sent_for(sched, 1)
    assert msgs[-1].finished
    assert msgs[-1].finish_reason == "stop"
    assert msgs[-1].matched_stop == "23"
    assert msgs[-1].stop_strs == ["23"]


def test_ignore_eos_runs_to_length():
    sched = make_scheduler({1: [EOS] * 5})
    sched._handle(user_msg(1, max_tokens=4, ignore_eos=True))
    while sched.active:
        sched._step()
    msgs = sent_for(sched, 1)
    assert len(msgs) == 4
    assert msgs[-1].finish_reason == "length"


def test_round_robin_interleaves_requests():
    sched = make_scheduler({1: [11, EOS], 2: [22, 22, EOS]})
    sched._handle(user_msg(1, max_tokens=10))
    sched._handle(user_msg(2, max_tokens=10))
    sched._step()  # one token EACH — neither request waits for the other to finish
    assert [m.uid for m in sched.sent if isinstance(m, DetokenizeMsg)] == [1, 2]
    while sched.active:
        sched._step()
    assert sent_for(sched, 1)[-1].finished and sent_for(sched, 2)[-1].finished


def test_abort_drops_request_silently():
    sched = make_scheduler({1: [5, 5, 5]})
    sched._handle(user_msg(1, max_tokens=10))
    replies, do_exit = sched._handle(AbortBackendMsg(uid=1))
    assert replies == [] and not do_exit
    assert sched.active == {}


def test_prompt_over_context_is_a_terminal_error():
    sched = make_scheduler()
    replies, _ = sched._handle(user_msg(1, prompt_len=64, max_tokens=4))
    assert isinstance(replies[0], ErrorReplyMsg)
    assert replies[0].code == "context_length_exceeded"
    assert 1 not in sched.active


def test_cache_rebuild_is_rejected_not_ignored():
    sched = make_scheduler()
    replies, _ = sched._handle(CacheRebuildBackendMsg(request_id="r1", num_pages=8))
    assert isinstance(replies[0], CacheRebuildResultMsg)
    assert replies[0].status == "failed"
    assert replies[0].request_id == "r1"


def test_rebuild_max_seq_len_applies_without_offload():
    """The context slider works on resident models too: KV is per-request, so
    the ceiling is a scheduler variable, not an expert-cache property."""
    sched = make_scheduler()
    replies, _ = sched._handle(
        CacheRebuildBackendMsg(request_id="r2", max_seq_len=4096)
    )
    assert replies[0].status == "ok"
    assert replies[0].max_seq_len == 4096
    assert sched.max_seq_len == 4096
    # admission enforces the new ceiling immediately
    err, _ = sched._handle(user_msg(1, prompt_len=5000, max_tokens=4))
    assert isinstance(err[0], ErrorReplyMsg)


def test_rebuild_max_seq_len_floors_at_1024():
    sched = make_scheduler()
    replies, _ = sched._handle(
        CacheRebuildBackendMsg(request_id="r3", max_seq_len=8)
    )
    assert replies[0].status == "ok"
    assert sched.max_seq_len == 1024


def test_rebuild_busy_while_serving():
    sched = make_scheduler(script={1: [5, EOS]})
    sched._handle(user_msg(1))
    replies, _ = sched._handle(
        CacheRebuildBackendMsg(request_id="r4", max_seq_len=2048)
    )
    assert replies[0].status == "busy"
    assert sched.max_seq_len == 64  # unchanged


def test_exit_msg_inside_batch_requests_exit():
    sched = make_scheduler({1: [5, EOS]})
    batch = BatchBackendMsg(data=[user_msg(1, max_tokens=2), ExitMsg()])
    replies, do_exit = sched._handle(batch)
    assert do_exit
    assert any(isinstance(r, PromptAdmittedMsg) for r in replies)


def test_generator_error_isolates_the_request():
    sched = make_scheduler({1: [5], 2: [8, EOS]})

    def boom():
        raise RuntimeError("metal says no")
        yield  # pragma: no cover

    sched._handle(user_msg(1, max_tokens=4))
    sched._handle(user_msg(2, max_tokens=4))
    sched.active[1].generator = boom()
    while sched.active:
        sched._step()
    errs = [m for m in sched.sent if isinstance(m, ErrorReplyMsg)]
    assert len(errs) == 1 and errs[0].uid == 1
    assert sent_for(sched, 2)[-1].finished  # request 2 unaffected


def test_filter_kwargs_drops_unknown_only():
    def fn(a, b=1):
        return a, b

    assert _filter_kwargs(fn, {"a": 1, "b": 2, "c": 3}) == {"a": 1, "b": 2}

    def var_kw(**kwargs):
        return kwargs

    assert _filter_kwargs(var_kw, {"x": 1}) == {"x": 1}


def test_the_cuda_only_flags_are_gone():
    """There is one execution backend now, so its flags are not options to get
    wrong. A flag that parses but reaches nothing is worse than no flag."""
    from maxtoken.server.args import parse_args

    for flag, value in (
        ("--backend", "cuda"),
        ("--tp-size", "2"),
        ("--cuda-graph-max-bs", "8"),
        ("--attention-backend", "flashinfer"),
        ("--nvfp4-backend", "marlin"),
        ("--disable-pynccl", None),
    ):
        argv = ["--model-path", "/nonexistent", flag]
        if value is not None:
            argv.append(value)
        with pytest.raises(SystemExit):
            parse_args(argv)


# ------------------------------------------------- cache snapshot / rollback

class _FakeWindowCache:
    """A window cache in the shape mlx-lm's RotatingKVCache has: the position
    lives in meta_state, and trimmability depends on it."""

    max_size = 4

    def __init__(self):
        self.keys = [0]
        self.offset = 0
        self._idx = 0

    def is_trimmable(self):
        return self.offset < self.max_size

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        self._idx -= n
        return n

    @property
    def state(self):
        return [self.keys]

    @state.setter
    def state(self, v):
        self.keys = v[0]

    @property
    def meta_state(self):
        return (str(self.offset), str(self._idx))

    @meta_state.setter
    def meta_state(self, v):
        self.offset, self._idx = int(v[0]), int(v[1])


class _FakePlainCache:
    """A plain KV cache: always trimmable, rewinds by offset."""

    def __init__(self):
        self.offset = 0

    def is_trimmable(self):
        return True

    def trim(self, n):
        self.offset -= n
        return n

    @property
    def state(self):
        return []

    @property
    def meta_state(self):
        return ()


def test_plain_kv_cache_rolls_back_by_trimming():
    """No snapshot for a plain KV cache: holding its arrays would force a
    copy-on-write of the whole cache on every decode step."""
    c = _FakePlainCache()
    snap = MlxScheduler._cache_snapshot(c)
    assert snap is None
    c.offset = 5
    MlxScheduler._cache_rollback(c, snap, 3)
    assert c.offset == 2


def test_window_cache_rollback_restores_its_position():
    """Regression: restoring only the arrays left the offset past the buffer,
    and the next update computed a negative size (DeepSeek-V4's sliding-window
    attention died with '[full] Negative dimensions not allowed')."""
    c = _FakeWindowCache()
    c.keys, c.offset, c._idx = [1, 2], 2, 2
    snap = MlxScheduler._cache_snapshot(c)
    assert snap is not None  # window caches never take the trim path

    # a step advances past the window
    c.keys, c.offset, c._idx = [1, 2, 3, 4, 5], 5, 1
    MlxScheduler._cache_rollback(c, snap, 3)

    assert c.keys == [1, 2]
    assert (c.offset, c._idx) == (2, 2)


def test_a_prefilling_request_yields_the_step_to_the_others():
    """A generator that is still processing its prompt yields None per chunk;
    the step sends nothing for it and moves on, so a request that arrived
    behind a long prompt decodes while that prompt is still being prefilled."""
    sched = make_scheduler({2: [22, 22, EOS]})
    # Request 1: three prefill chunks, then tokens.
    long_gen = iter([None, None, None, (11, None), (EOS, None)])
    sched._make_generator = lambda ids, sp, _orig=sched._make_generator: (
        (long_gen, None, 0) if ids[0] == 1 else _orig(ids, sp)
    )
    sched._handle(user_msg(1, max_tokens=10))
    sched._handle(user_msg(2, max_tokens=10))
    sched._step()  # request 1 chunk 1; request 2 gets a token (first step: nobody known prefilling yet)
    assert [m.uid for m in sched.sent if isinstance(m, DetokenizeMsg)] == [2]
    assert 1 in sched.active and sched.active[1].prefilling
    sched._step()  # chunk 2; request 2 now gets several rounds per step and finishes
    assert [m.uid for m in sched.sent if isinstance(m, DetokenizeMsg)] == [2, 2, 2]
    assert sent_for(sched, 2)[-1].finished
    while sched.active:
        sched._step()
    assert [m.next_token for m in sent_for(sched, 1)] == [11, EOS]


def test_decoding_requests_get_several_rounds_per_prefill_chunk():
    """One chunk of prefill is seconds, one decode round a tenth of that: while
    request 1 is on its prompt, request 2 is not held to one token per chunk."""
    from maxtoken.mlx_backend.worker import DECODE_ROUNDS_WHILE_PREFILLING

    sched = make_scheduler({2: [22] * 40})
    long_gen = iter([None] * 5 + [(11, None), (EOS, None)])
    sched._make_generator = lambda ids, sp, _orig=sched._make_generator: (
        (long_gen, None, 0) if ids[0] == 1 else _orig(ids, sp)
    )
    sched._handle(user_msg(1, max_tokens=10))
    sched._handle(user_msg(2, max_tokens=40))
    sched._step()  # request 1 reports itself prefilling; request 2: one round
    sched._step()  # request 2 now gets DECODE_ROUNDS_WHILE_PREFILLING rounds
    assert len(sent_for(sched, 2)) == 1 + DECODE_ROUNDS_WHILE_PREFILLING
    # Once nobody is prefilling, back to one round per step (fair between decoders).
    while sched.active.get(1) is not None and sched.active[1].prefilling:
        sched._step()
    before = len(sent_for(sched, 2))
    sched._step()
    assert len(sent_for(sched, 2)) - before <= 1 + DECODE_ROUNDS_WHILE_PREFILLING
