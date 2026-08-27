"""Reasoning-parser auto-inference: the qwen3 family needs the chat template
consulted — Coder/Instruct-2507 checkpoints never think, and attaching the
parser routes their entire completion into reasoning_content."""

import json

import pytest

from maxtoken.server.args import parse_args


def _parse(model_path: str):
    out = parse_args(["--model", str(model_path)])
    return out[0] if isinstance(out, tuple) else out


def _write_model(tmp_path, template: str):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_next",
                "architectures": ["Qwen3NextForCausalLM"],
                "max_position_embeddings": 4096,
            }
        )
    )
    (tmp_path / "chat_template.jinja").write_text(template)


# The distinguishing feature is the add_generation_prompt block: thinking
# templates open an implicit <think> there, never-thinking ones end at the
# bare assistant header (even when they mention <think> while parsing prior
# turns, as Instruct-2507 does).
_THINKING_TAIL = (
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}{{- '<think>\\n' }}"
    "{%- endif %}"
)
_PLAIN_TAIL = (
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)
_HISTORY_SPLIT = (
    "{%- set r = content.split('</think>')[0].split('<think>')[-1] %}"
)


def test_thinking_template_keeps_parser(tmp_path):
    _write_model(tmp_path, _HISTORY_SPLIT + _THINKING_TAIL)
    assert _parse(tmp_path).reasoning_parser == "qwen3"


def test_never_thinking_template_drops_parser(tmp_path):
    _write_model(tmp_path, _PLAIN_TAIL)
    assert _parse(tmp_path).reasoning_parser is None


def test_history_think_mentions_do_not_count(tmp_path):
    # Instruct-2507 shape: <think> appears in history parsing only.
    _write_model(tmp_path, _HISTORY_SPLIT + _PLAIN_TAIL)
    assert _parse(tmp_path).reasoning_parser is None


def test_no_template_keeps_parser(tmp_path):
    # Incomplete snapshot (weights + config only): stay on the hybrid default.
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "max_position_embeddings": 4096})
    )
    assert _parse(tmp_path).reasoning_parser == "qwen3"


def test_explicit_flag_wins(tmp_path):
    _write_model(tmp_path, _PLAIN_TAIL)
    out = parse_args(["--model", str(tmp_path), "--reasoning-parser", "qwen3"])
    cfg = out[0] if isinstance(out, tuple) else out
    assert cfg.reasoning_parser == "qwen3"
