"""Reasoning budget: stop a request that thinks itself out of its output budget.

A weak model can loop inside <think> until max_tokens is gone and return empty
content (observed on a 27B research MoE). --max-reasoning-tokens and Anthropic's
thinking.budget_tokens cap the tokens spent inside the reasoning block.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from freetoken.core import SamplingParams
from freetoken.message import UserReply
from freetoken.server.generation import (
    ContentDelta,
    GenDone,
    GenSpec,
    ReasoningDelta,
    generate_events,
    generate_full,
)


class _State:
    """Feeds one token per ack, like the scheduler does."""

    def __init__(self, chunks: list[str], max_reasoning_tokens=None):
        self.config = SimpleNamespace(
            model_path="/models/unit-model",
            served_model_name="unit-model",
            tool_call_parser="llama3",
            reasoning_parser="qwen3",
            max_reasoning_tokens=max_reasoning_tokens,
        )
        self.chunks = chunks
        self.sent = None

    def new_user(self) -> int:
        return 42

    async def send_one(self, msg):
        self.sent = msg

    async def wait_for_ack(self, uid: int):
        for i, text in enumerate(self.chunks):
            yield UserReply(
                uid=42,
                incremental_output=text,
                finished=(i == len(self.chunks) - 1),
                prompt_tokens_delta=3 if i == 0 else 0,
                completion_tokens_delta=1,
                cached_tokens=0,
                finish_reason="stop" if i == len(self.chunks) - 1 else None,
            )


def _spec(max_reasoning_tokens=None) -> GenSpec:
    return GenSpec(
        messages=[{"role": "user", "content": "hi"}],
        sampling_params=SamplingParams(max_tokens=64),
        max_reasoning_tokens=max_reasoning_tokens,
    )


def _run(agen):
    async def _collect():
        return [ev async for ev in agen]

    return asyncio.run(_collect())


# The qwen3 parser starts inside reasoning (the template opens <think>), so the
# leading chunks are thinking and everything after </think> is the answer.
LOOPING = ["think "] * 10 + ["</think>", "answer"]


def test_budget_stops_a_looping_thinker():
    state = _State(LOOPING, max_reasoning_tokens=4)
    events = _run(generate_events(42, _spec(), state))

    reasoning = "".join(e.text for e in events if isinstance(e, ReasoningDelta))
    content = "".join(e.text for e in events if isinstance(e, ContentDelta))
    done = [e for e in events if isinstance(e, GenDone)][-1]

    assert reasoning.count("think") == 4  # stopped at the budget, not at token 10
    assert content == ""  # cut before the answer: a truncation
    assert done.finish_reason == "length"


def test_no_budget_lets_it_finish():
    state = _State(LOOPING)
    events = _run(generate_events(42, _spec(), state))

    content = "".join(e.text for e in events if isinstance(e, ContentDelta))
    done = [e for e in events if isinstance(e, GenDone)][-1]
    assert content == "answer"
    assert done.finish_reason == "stop"


def test_budget_ignores_answer_tokens():
    """Only tokens inside the thinking block count — a long answer is not
    truncated by a reasoning budget."""
    state = _State(["short ", "</think>"] + ["word "] * 20, max_reasoning_tokens=4)
    events = _run(generate_events(42, _spec(), state))

    content = "".join(e.text for e in events if isinstance(e, ContentDelta))
    done = [e for e in events if isinstance(e, GenDone)][-1]
    assert content.count("word") == 20
    assert done.finish_reason == "stop"


def test_request_budget_and_server_cap_take_the_lower():
    # server 8, request 3 -> 3 wins
    state = _State(LOOPING, max_reasoning_tokens=8)
    events = _run(generate_events(42, _spec(max_reasoning_tokens=3), state))
    reasoning = "".join(e.text for e in events if isinstance(e, ReasoningDelta))
    assert reasoning.count("think") == 3

    # server 2, request 9 -> 2 wins
    state = _State(LOOPING, max_reasoning_tokens=2)
    events = _run(generate_events(42, _spec(max_reasoning_tokens=9), state))
    reasoning = "".join(e.text for e in events if isinstance(e, ReasoningDelta))
    assert reasoning.count("think") == 2


def test_budget_applies_to_the_non_streaming_path():
    state = _State(LOOPING, max_reasoning_tokens=4)
    result = asyncio.run(generate_full(42, _spec(), state))
    assert result.reasoning.count("think") == 4
    assert result.content == ""
    assert result.finish_reason == "length"
