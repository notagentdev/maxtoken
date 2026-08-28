"""LM Studio's model listing: what a client learns about the model from /api/v0.

The number that matters is the context length. An agent that sizes its
conversation from this listing must see the model's real window here, and the
window currently enforced once a runtime override narrowed it -- otherwise it
falls back to a guess (one did: 32k for a 262k model).
"""

from types import SimpleNamespace

from fastapi.testclient import TestClient

import maxtoken.server.api_server as api


def _state(**over):
    s = SimpleNamespace(
        maintenance_state="serving",
        config=SimpleNamespace(
            model_path="ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit",
            served_model_name="Ornith-1.5-35B-A3B-MLX-4bit",
            max_seq_len=262144,
            model_config=SimpleNamespace(
                model_type="qwen3_5_moe", quantization={"bits": 4, "group_size": 64}
            ),
        ),
    )
    for k, v in over.items():
        setattr(s, k, v)
    return s


def _with(state):
    prev = api._GLOBAL_STATE
    api._GLOBAL_STATE = state
    return prev


def test_listing_reports_kind_state_and_the_real_window():
    prev = _with(_state())
    try:
        body = TestClient(api.app).get("/api/v0/models").json()
        assert body["object"] == "list" and len(body["data"]) == 1
        card = body["data"][0]
        assert card["id"] == "Ornith-1.5-35B-A3B-MLX-4bit"
        assert card["type"] == "llm" and card["state"] == "loaded"
        assert card["max_context_length"] == 262144
        assert card["loaded_context_length"] == 262144
        assert card["arch"] == "qwen3_5_moe" and card["quantization"] == "4bit"
        assert card["publisher"] == "ornith-ai" and card["compatibility_type"] == "mlx"
    finally:
        _with(prev)


def test_a_runtime_context_override_is_the_loaded_window():
    prev = _with(_state(context_length_override=16384))
    try:
        card = TestClient(api.app).get("/api/v0/models").json()["data"][0]
        assert card["max_context_length"] == 262144, "the trained ceiling stays"
        assert card["loaded_context_length"] == 16384, "what admission enforces now"
    finally:
        _with(prev)


def test_retrieve_returns_the_card_or_404():
    prev = _with(_state())
    try:
        client = TestClient(api.app)
        assert client.get("/api/v0/models/Ornith-1.5-35B-A3B-MLX-4bit").json()["type"] == "llm"
        r = client.get("/api/v0/models/nope")
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "model_not_found"
    finally:
        _with(prev)


def test_a_loading_engine_is_not_reported_as_loaded():
    prev = _with(_state(maintenance_state="loading"))
    try:
        card = TestClient(api.app).get("/api/v0/models").json()["data"][0]
        assert card["state"] == "loading"
        assert "loaded_context_length" not in card
    finally:
        _with(prev)


def test_the_rest_chat_route_is_gated_like_the_openai_one():
    prev = _with(_state(maintenance_state="loading"))
    try:
        r = TestClient(api.app).post(
            "/api/v0/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 503
    finally:
        _with(prev)
