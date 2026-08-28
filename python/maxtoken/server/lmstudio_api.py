"""LM Studio's own model listing, for clients that discover models the LM Studio way.

An OpenAI ``/v1/models`` entry is a name and nothing else, so a client that
needs to know what a model can hold asks LM Studio's REST listing instead:
``GET /api/v0/models`` reports every model's kind, load state and context
length (``max_context_length``, and ``loaded_context_length`` once loaded).
Coding agents size their conversations from that number. One that could not
get it here fell back to a conservative 32k window for a model serving 262k
-- a third of the context it was told the model had, exactly the output cap,
which looked like a bug on our side and was a listing we did not serve.

The listing is served beside ``/v1``, not under it: LM Studio puts it at
``{host}/api/v0/models``, and clients strip the ``/v1`` from their base URL
to find it. The chat and completion routes of that REST API are aliases of
the OpenAI handlers so a client that speaks only ``/api/v0`` still gets
answers; LM Studio's ``stats`` block is not reproduced.
"""

from __future__ import annotations

from typing import Any, Callable

from fastapi import FastAPI, Request

from .api_models import ChatCompletionRequest, CompletionRequest
from .model_meta import model_config_or_none
from .openai_api import (
    _maintenance_gate,
    _model_context_length,
    _served_model_name,
    create_error_response,
    handle_chat_completion,
    handle_completion,
)
from .request_logger import log_request


def _context_ceiling(state: Any) -> int | None:
    """The model's trained ceiling, unaffected by a runtime override."""
    ceiling = getattr(state, "context_length_ceiling", None)
    if ceiling:
        return int(ceiling)
    try:
        value = int(state.config.max_seq_len)
    except Exception:  # noqa: BLE001 -- metadata route: unknown, never 500
        return None
    return value if value > 0 else None


def _quantization_label(model_config: Any) -> str | None:
    quant = getattr(model_config, "quantization", None)
    if isinstance(quant, dict):
        bits = quant.get("bits")
        return f"{bits}bit" if bits else None
    return str(quant) if quant else None


def model_card(state: Any) -> dict[str, Any]:
    served = _served_model_name(state)
    root = str(getattr(state.config, "model_path", "") or "")
    publisher = root.split("/")[0] if "/" in root and not root.startswith(("/", ".")) else None
    model_config = model_config_or_none(state.config)
    maintenance = getattr(state, "maintenance_state", "serving")
    state_label = "loaded" if maintenance in ("serving", "rebuilding") else (
        "loading" if maintenance == "loading" else "not-loaded"
    )
    ceiling = _context_ceiling(state)
    loaded = _model_context_length(state) or ceiling
    card: dict[str, Any] = {
        "id": served,
        "object": "model",
        "type": "llm",
        "publisher": publisher,
        "arch": getattr(model_config, "model_type", None),
        "compatibility_type": "mlx",
        "quantization": _quantization_label(model_config),
        "state": state_label,
        "max_context_length": ceiling,
    }
    if state_label == "loaded" and loaded:
        card["loaded_context_length"] = int(loaded)
    return card


def register_lmstudio_routes(
    app: FastAPI,
    get_state: Callable[[], Any],
    get_model_sampling: Callable[[], dict[str, Any]],
) -> None:
    @app.get("/api/v0/models")
    async def lmstudio_models():
        return {"object": "list", "data": [model_card(get_state())]}

    @app.get("/api/v0/models/{model_id:path}")
    async def lmstudio_model(model_id: str):
        state = get_state()
        if model_id != _served_model_name(state):
            return create_error_response(
                f"The model '{model_id}' does not exist",
                status_code=404,
                err_type="invalid_request_error",
                param="model",
                code="model_not_found",
            )
        return model_card(state)

    @app.post("/api/v0/chat/completions")
    async def lmstudio_chat(req: ChatCompletionRequest, request: Request):
        log_request("/api/v0/chat/completions", req, request)
        state = get_state()
        if (gate := _maintenance_gate(state)) is not None:
            return gate
        return await handle_chat_completion(req, request, state, get_model_sampling())

    @app.post("/api/v0/completions")
    async def lmstudio_completions(req: CompletionRequest, request: Request):
        log_request("/api/v0/completions", req, request)
        state = get_state()
        if (gate := _maintenance_gate(state)) is not None:
            return gate
        return await handle_completion(req, request, state, get_model_sampling())
