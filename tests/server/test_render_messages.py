"""render_messages normalizes what clients send into what chat templates accept."""

from maxtoken.server.generation import render_messages


def test_developer_role_is_rendered_as_system():
    """OpenAI's newer SDKs send the system prompt as role "developer"; the
    templates only know "system", and a Qwen template raised "Unexpected
    message role" -- which an agent showed as a bare 400."""
    out = render_messages(
        [
            {"role": "developer", "content": "You are terse."},
            {"role": "user", "content": "wer bist du?"},
        ]
    )
    assert [m["role"] for m in out] == ["system", "user"]
    assert out[0]["content"] == "You are terse."


def test_other_roles_pass_through_untouched():
    out = render_messages(
        [
            {"role": "system", "content": "s"},
            {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
            {"role": "assistant", "content": "x"},
            {"role": "tool", "content": "r", "tool_call_id": "call_1"},
        ]
    )
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool"]
    assert out[1]["content"] == "ab"
