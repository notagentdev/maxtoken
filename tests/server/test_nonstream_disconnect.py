"""A non-stream request whose client goes away must not keep the engine busy.

Before this, only the SSE paths watched the connection; a client-side timeout on
`stream: false` left the model generating to max_tokens (32,791 tokens over 76
minutes on Qwen3.8-27B after a reviewer app's 120 s timeout, 2026-09-12)."""
import asyncio

import pytest

from maxtoken.server.generation import ClientGone, until_disconnect


class _Request:
    def __init__(self, gone_after: int):
        self.polls = 0
        self.gone_after = gone_after

    async def is_disconnected(self):
        self.polls += 1
        return self.polls >= self.gone_after


class _State:
    def __init__(self):
        self.aborted = []

    async def abort_user(self, uid):
        self.aborted.append(uid)


async def _never():
    await asyncio.sleep(3600)


async def _quick():
    await asyncio.sleep(0.01)
    return "done"


def test_a_finished_generation_returns_its_result_and_never_aborts():
    state = _State()
    request = _Request(gone_after=10_000)
    result = asyncio.run(until_disconnect(_quick(), request, state, uid=7, poll_s=0.01))
    assert result == "done"
    assert state.aborted == []


def test_a_dropped_client_aborts_the_engine_and_stops_waiting():
    state = _State()
    request = _Request(gone_after=3)

    async def run():
        with pytest.raises(ClientGone) as info:
            await until_disconnect(_never(), request, state, uid=7, poll_s=0.01)
        assert info.value.code == "client_disconnected"
        # the generation task itself is gone, not left running in the loop
        await asyncio.sleep(0.02)
        assert all(t.done() for t in asyncio.all_tasks() if t is not asyncio.current_task())

    asyncio.run(run())
    assert state.aborted == [7]


def test_without_a_request_object_it_is_a_plain_await():
    assert asyncio.run(until_disconnect(_quick(), None, _State(), uid=1)) == "done"


def test_the_error_reads_as_a_generation_error_for_the_adapters():
    from maxtoken.server.generation import GenerationError

    assert issubclass(ClientGone, GenerationError)
