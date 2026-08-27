"""The terminal status bar has to show that the engine is working, not just that
it is busy.

While a prompt is being processed nothing else on the line moves: no tokens have
arrived, so the counters and the rate all read zero. A clock is the only thing
that distinguishes a long prefill from a hung server, which is the same reason
the web console grew a spinner.
"""

import pytest

from maxtoken.shell.tui import ShellStats


def _stats() -> ShellStats:
    s = ShellStats()
    s.model_label = "model"
    return s


def test_nothing_is_shown_between_turns():
    s = _stats()
    assert s.wait_seconds(now=100.0) is None
    assert "prompt" not in s.format(now=100.0)
    assert "ttft" not in s.format(now=100.0)


def test_the_clock_runs_while_the_prompt_is_processed():
    s = _stats()
    s.mark_started(now=100.0)
    assert s.wait_seconds(now=102.4) == pytest.approx(2.4)
    line = s.format(now=102.4)
    assert "[prefill]" in line and "prompt 2.4s" in line
    # and it keeps moving with the clock, which is the whole point
    assert "prompt 5.0s" in s.format(now=105.0)


def test_it_freezes_at_the_first_token_and_becomes_the_ttft():
    s = _stats()
    s.mark_started(now=100.0)
    s.add_completion_tokens(1, now=103.1)
    for later in (104.0, 130.0):
        line = s.format(now=later)
        assert "ttft 3.1s" in line, "the wait is over; the number must stop"
        assert "prompt" not in line


def test_a_finished_turn_drops_it_again():
    s = _stats()
    s.mark_started(now=100.0)
    s.add_completion_tokens(1, now=101.0)
    s.mark_finished(now=105.0)
    assert s.wait_seconds(now=105.0) is None
    assert "ttft" not in s.format(now=105.0)


def test_a_turn_that_never_produced_a_token_still_reads_as_waiting():
    """An aborted or failed turn must not report a TTFT it never reached."""
    s = _stats()
    s.mark_started(now=100.0)
    assert "prompt" in s.format(now=101.0)
    s.mark_finished(now=101.0)
    assert s.wait_seconds(now=101.0) is None
