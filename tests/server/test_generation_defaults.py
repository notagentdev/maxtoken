"""Server-wide generation defaults: temperature and the output cap.

They fill whatever a request leaves unset -- the console's chat, an agent, an
SDK -- and a request's own values still win. Set from the Console tab through
/admin/cache/rebuild (a frontend knob, no engine work), read back from
/admin/cache/status, and consumed by every protocol through effective_sampling().
"""

from types import SimpleNamespace

from fastapi.testclient import TestClient

import maxtoken.server.api_server as api
from maxtoken.server.generation import DEFAULT_MAX_OUTPUT_TOKENS, resolve_sampling


def _state(**over):
    from maxtoken.server.stats import StatsTracker

    s = SimpleNamespace(
        maintenance_state="serving",
        rebuild_futures={},
        last_rebuild=None,
        stats=StatsTracker(),
        config=SimpleNamespace(max_seq_len=8192, max_output_tokens=None, max_reasoning_tokens=None),
    )
    for k, v in over.items():
        setattr(s, k, v)
    return s


def _with_state(state):
    prev_state, prev_sampling = api._GLOBAL_STATE, api._MODEL_SAMPLING
    api._GLOBAL_STATE = state
    api._MODEL_SAMPLING = {"temperature": 1.0, "top_k": 20, "top_p": 0.95}
    return prev_state, prev_sampling


def _restore(prev):
    api._GLOBAL_STATE, api._MODEL_SAMPLING = prev


def test_defaults_start_from_the_checkpoint_and_the_32k_cap():
    prev = _with_state(_state())
    try:
        gen = TestClient(api.app).get("/admin/cache/status").json()["generation"]
        assert gen == {"temperature": 1.0, "top_k": 20, "top_p": 0.95,
                       "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS, "thinking": "model"}
    finally:
        _restore(prev)


def test_console_sets_temperature_and_cap_without_engine_work():
    state = _state()
    prev = _with_state(state)
    try:
        client = TestClient(api.app)
        r = client.post("/admin/cache/rebuild", json={"temperature": 0.3, "max_output_tokens": 8192})
        assert r.status_code == 200, r.text
        assert r.json() == {"status": "ok", "temperature": 0.3, "max_output_tokens": 8192}
        assert state.maintenance_state == "serving", "a frontend knob never touches the gate"
        assert api.effective_sampling()["temperature"] == 0.3
        assert api.effective_sampling()["max_tokens"] == 8192
        gen = client.get("/admin/cache/status").json()["generation"]
        assert gen["temperature"] == 0.3 and gen["max_output_tokens"] == 8192
        # 0 = back to the server default; the temperature stays where it was set
        r = client.post("/admin/cache/rebuild", json={"max_output_tokens": 0})
        assert r.status_code == 200
        assert api.effective_sampling().get("max_tokens") is None
        assert api.effective_sampling()["temperature"] == 0.3
    finally:
        _restore(prev)


def test_invalid_values_are_422():
    prev = _with_state(_state())
    try:
        client = TestClient(api.app)
        assert client.post("/admin/cache/rebuild", json={"temperature": -0.1}).status_code == 422
        assert client.post("/admin/cache/rebuild", json={"max_output_tokens": -5}).status_code == 422
    finally:
        _restore(prev)


def test_max_output_tokens_flag_feeds_the_defaults():
    prev = _with_state(_state(config=SimpleNamespace(max_seq_len=8192, max_output_tokens=4096,
                                                     max_reasoning_tokens=None)))
    try:
        assert api.effective_sampling()["max_tokens"] == 4096
    finally:
        _restore(prev)


def test_resolve_sampling_takes_the_cap_from_the_defaults_but_the_request_wins():
    base = dict(temperature=None, top_k=None, top_p=None, ignore_eos=False)
    assert resolve_sampling(max_tokens=None, model_sampling={"max_tokens": 4096}, **base).max_tokens == 4096
    assert resolve_sampling(max_tokens=None, model_sampling={}, **base).max_tokens == DEFAULT_MAX_OUTPUT_TOKENS
    assert resolve_sampling(max_tokens=77, model_sampling={"max_tokens": 4096}, **base).max_tokens == 77
    sp = resolve_sampling(max_tokens=None, model_sampling={"temperature": 0.3, "max_tokens": 10}, **base)
    assert sp.temperature == 0.3 and sp.max_tokens == 10


# --- thinking default -------------------------------------------------------
# "Reasoning budget: no cap" is a cap, not a switch: Qwen3.5 / Ornith templates
# think unless a request says enable_thinking=false. The server-wide switch
# (--thinking, or the console's Generation row) folds that flag into every
# request that carries none of its own.

def _spec(**ctk):
    from maxtoken.server.generation import GenSpec, SamplingParams

    return GenSpec(messages=[{"role": "user", "content": "hi"}],
                   sampling_params=SamplingParams(), chat_template_kwargs=dict(ctk))


def test_thinking_default_is_the_checkpoints_until_the_console_says_otherwise():
    from maxtoken.server.generation import apply_thinking_default

    state = _state()
    prev = _with_state(state)
    try:
        client = TestClient(api.app)
        assert client.get("/admin/cache/status").json()["generation"]["thinking"] == "model"
        assert apply_thinking_default(_spec(), state).chat_template_kwargs == {}

        r = client.post("/admin/cache/rebuild", json={"thinking": "off"})
        assert r.status_code == 200 and r.json() == {"status": "ok", "thinking": "off"}
        assert state.maintenance_state == "serving", "a frontend knob never touches the gate"
        assert client.get("/admin/cache/status").json()["generation"]["thinking"] == "off"
        folded = apply_thinking_default(_spec(), state).chat_template_kwargs
        assert folded["enable_thinking"] is False and folded["thinking_mode"] == "disabled"
        # a request's own flag wins, whatever the server default says
        assert apply_thinking_default(_spec(enable_thinking=True), state).chat_template_kwargs == {
            "enable_thinking": True}
        assert apply_thinking_default(_spec(reasoning_effort="high"), state).chat_template_kwargs == {
            "reasoning_effort": "high"}
        # idempotent: the second application changes nothing
        again = apply_thinking_default(apply_thinking_default(_spec(), state), state)
        assert again.chat_template_kwargs == folded

        r = client.post("/admin/cache/rebuild", json={"thinking": "on"})
        assert r.status_code == 200
        assert apply_thinking_default(_spec(), state).chat_template_kwargs["enable_thinking"] is True

        r = client.post("/admin/cache/rebuild", json={"thinking": "model"})
        assert r.status_code == 200
        assert apply_thinking_default(_spec(), state).chat_template_kwargs == {}
        assert client.post("/admin/cache/rebuild", json={"thinking": "maybe"}).status_code == 422
    finally:
        _restore(prev)


def test_thinking_flag_seeds_the_default_and_the_console_overrides_it():
    from maxtoken.server.args import ServerArgs
    from maxtoken.server.generation import apply_thinking_default, thinking_default

    assert ServerArgs.thinking is None
    state = _state(config=SimpleNamespace(max_seq_len=8192, max_output_tokens=None,
                                          max_reasoning_tokens=None, thinking="off"))
    assert thinking_default(state) is False
    assert apply_thinking_default(_spec(), state).chat_template_kwargs["enable_thinking"] is False
    state.thinking_override = True
    assert thinking_default(state) is True


def test_model_card_reads_config_json_for_mlx_lm_only_models(tmp_path):
    import json

    from maxtoken.server.stats import derive_model_card

    (tmp_path / "config.json").write_text(json.dumps({"text_config": {
        "num_experts": 256, "layer_types": ["linear_attention"] * 3 + ["full_attention"]}}))

    class _Cfg:
        served_model_name = "ornith"
        max_seq_len = 4096
        model_path = str(tmp_path)

        @property
        def model_config(self):
            raise ValueError("not in the registry")

    assert derive_model_card(_Cfg()) == {"id": "ornith", "ctx": 4096, "attn": "hybrid_linear", "moe": True}


def test_thinking_cli_flag_parses():
    from unittest.mock import patch

    from maxtoken.server.args import parse_args

    class _Config:
        def to_dict(self):
            return {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}

    with patch("maxtoken.utils.cached_load_hf_config", lambda _p: _Config()):
        assert parse_args(["--model", "/models/anon", "--thinking", "off"])[0].thinking == "off"
        assert parse_args(["--model", "/models/anon"])[0].thinking is None
