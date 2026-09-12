"""POST /admin/reload relaunches the engine workers with the boot configuration.

The route must close admission first, hand every in-flight request a terminal error reply,
tear the old workers down, launch new ones through the same callback the boot used and
supervise them under a NEW generation -- so the old supervisor's death report is ignored
instead of latching the server "failed". Fake workers and a fake launcher stand in for the
engine; the supervisor and its drain are the real ones.
"""

from __future__ import annotations

import asyncio
import queue
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

from maxtoken.server.supervisor import BackendHandle, LoadProgress, run_backend_supervisor


class _FakeProc:
    def __init__(self, name):
        self.name = name
        self.alive = True
        self.terminated = False

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated = True
        self.alive = False

    def join(self, timeout=None):
        return None

    def kill(self):
        self.alive = False


def _state(**over):
    s = SimpleNamespace(
        maintenance_state="serving",
        fatal_error=None,
        ready_at=1.0,
        context_length_override=4096,
        reasoning_budget_override=None,
        last_rebuild={"status": "ok"},
        load_progress=LoadProgress(),
        ack_map={},
        event_map={},
        backend_processes=[],
        backend_generation=0,
        start_backend=None,
        supervise=None,
        config=SimpleNamespace(served_model_name="m", max_seq_len=8192),
    )
    for k, v in over.items():
        setattr(s, k, v)
    return s


def _wire(state):
    """A launcher that returns an already-ready fake handle (no acks expected) and a
    supervisor starter mirroring run_api_server's, generation checks included."""
    launched = []
    failures = []

    def start_backend():
        handle = BackendHandle(
            processes=[_FakeProc("scheduler"), _FakeProc("detok")],
            ack_queue=queue.Queue(),
            expected_acks=0,
        )
        launched.append(handle)
        return handle

    def supervise(handle, generation):
        def on_ready():
            if state.backend_generation == generation and state.maintenance_state == "loading":
                state.maintenance_state = "serving"
                state.ready_at = time.monotonic()

        def on_failure(message):
            if state.backend_generation != generation:
                return
            failures.append(message)
            state.fatal_error = message
            state.maintenance_state = "failed"

        import threading

        threading.Thread(
            target=run_backend_supervisor,
            args=(handle, state.load_progress, on_ready),
            kwargs={"on_failure": on_failure, "poll": 0.01},
            daemon=True,
        ).start()

    state.start_backend = start_backend
    state.supervise = supervise
    return launched, failures


def _run_reload(state):
    from maxtoken.server.api_server import reload_backend

    return asyncio.run(reload_backend(state))


def _wait_serving(state, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if state.maintenance_state == "serving":
            return True
        time.sleep(0.01)
    return False


def test_reload_relaunches_under_a_new_generation_and_reopens_the_gate():
    state = _state()
    launched, failures = _wire(state)
    old = [_FakeProc("old-scheduler"), _FakeProc("old-detok")]
    state.backend_processes = old
    # An old-generation supervisor is watching the old workers, as at boot.
    state.supervise(BackendHandle(processes=old, ack_queue=queue.Queue(), expected_acks=0), 0)
    assert _wait_serving(state)

    body, status = _run_reload(state)
    assert status == 200 and body["status"] == "ok" and body["generation"] == 1
    assert all(p.terminated for p in old), "old workers must be torn down"
    assert len(launched) == 1 and state.backend_processes == launched[0].processes
    assert state.backend_generation == 1
    assert state.context_length_override is None and state.last_rebuild is None
    assert _wait_serving(state), "the new generation's ready ack must reopen the gate"
    time.sleep(0.1)  # give the old supervisor time to notice its workers died
    assert failures == [], "the superseded supervisor must not latch the server failed"
    assert state.fatal_error is None


def test_in_flight_requests_get_a_terminal_error_reply():
    from maxtoken.server.stats import StatsTracker

    state = _state(stats=StatsTracker())
    _wire(state)
    state.stats.on_new_user(7)
    event = asyncio.Event()

    async def run():
        state.ack_map[7] = []
        state.event_map[7] = event
        return await __import__("maxtoken.server.api_server", fromlist=["reload_backend"]).reload_backend(state)

    body, status = asyncio.run(run())
    assert status == 200
    reply = state.ack_map[7][-1]
    assert reply.finished and reply.error == "model reloading" and reply.uid == 7
    assert event.is_set()
    assert state.stats.active == 0, "the aborted request must not stay counted as active"
    assert state.stats.completed == 0, "nor be counted as completed"


def test_reload_is_refused_while_loading_or_rebuilding_and_without_a_launcher():
    for busy in ("loading", "rebuilding", "stopping"):
        state = _state(maintenance_state=busy)
        _wire(state)
        body, status = _run_reload(state)
        assert status == 409 and body["status"] == "busy"
    state = _state()  # no start_backend wired: a server that never booted an engine
    body, status = _run_reload(state)
    assert status == 503 and body["status"] == "unsupported"


def test_route_answers_through_the_app():
    import maxtoken.server.api_server as api

    state = _state(maintenance_state="rebuilding")
    prev = api._GLOBAL_STATE
    api._GLOBAL_STATE = state
    try:
        r = TestClient(api.app).post("/admin/reload")
        assert r.status_code == 409
        assert r.json()["status"] == "busy"
    finally:
        api._GLOBAL_STATE = prev


def test_reload_starts_the_launch_counters_over():
    """The console's "This launch" / "Tokens processed" cards count from the
    relaunch, like the uptime — before this they carried the old workers'
    totals across the reload."""
    from maxtoken.server.stats import StatsTracker

    state = _state()
    _wire(state)
    state.supervise(BackendHandle(processes=[], ack_queue=queue.Queue(), expected_acks=0), 0)
    assert _wait_serving(state)
    state.stats = StatsTracker()
    state.stats.completed = 7
    state.stats.prompt_tokens_total = 700
    state.stats.completion_tokens_total = 2100
    state.stats._decode.append((0.0, 5))

    body, status = _run_reload(state)
    assert status == 200 and body["status"] == "ok"
    tr = state.stats
    assert (tr.completed, tr.prompt_tokens_total, tr.completion_tokens_total) == (0, 0, 0)
    assert tr.decode_tps() == 0.0
