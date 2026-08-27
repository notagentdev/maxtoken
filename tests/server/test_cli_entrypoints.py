"""Both spellings of the entry point must reach the same commands.

``python -m maxtoken`` was once the server itself, so it took serve flags and
ignored a subcommand: ``python -m maxtoken serve --model ...`` failed with
"unrecognized arguments: serve" while every document showed ``mt serve``. Those
two spellings have to agree, and the old flags-first form has to keep working —
it is in shells and scripts that predate the dispatcher.
"""

import subprocess
import sys

import pytest


def _run(args):
    return subprocess.run(
        [sys.executable, "-m", "maxtoken", *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize("command", ["serve", "shell", "ctl", "daemon", "launch"])
def test_every_command_is_reachable_as_a_module(command):
    result = _run([command, "--help"])
    assert result.returncode == 0, result.stderr
    assert f"mt {command}" in result.stdout


def test_bare_flags_still_mean_serve():
    """The pre-dispatcher spelling: flags where a command was expected."""
    result = _run(["--model-path"])  # missing its value, so serve's parser complains
    assert "--model-path" in result.stderr
    assert "unknown mt command" not in result.stderr


def test_no_arguments_prints_the_command_list():
    result = _run([])
    assert result.returncode == 2
    for command in ("serve", "shell", "ctl", "daemon", "launch"):
        assert command in result.stderr


def test_an_unknown_word_is_reported_as_such():
    result = _run(["bench"])
    assert result.returncode == 2
    assert "unknown mt command: bench" in result.stderr
