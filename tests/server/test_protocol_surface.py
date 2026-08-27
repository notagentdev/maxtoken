"""The OpenAI and Anthropic surfaces are a contract with other people's clients.

Nothing in this repo gets to change them by accident. A client that speaks
either protocol reaches this server without knowing anything about it, so a
route that quietly disappears or moves is a broken integration somewhere else,
found by somebody else, later.

The second half of the file guards the other direction: the ``/v1`` prefix
belongs to those protocols, and our own endpoints have no business appearing
there. The ones that already do are listed by name, so the list can shrink but
never silently grow.
"""

import pytest

from maxtoken.server.api_server import app

OPENAI = {
    "/v1",
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/models",
    "/v1/models/{model_id:path}",
    "/v1/responses",
    "/v1/responses/{response_id}",
    "/v1/responses/{response_id}/cancel",
}

ANTHROPIC = {
    "/v1/messages",
    "/v1/messages/count_tokens",
}

# Ours. They answer under /admin now; these spellings remain so existing
# clients, dashboards and scripts keep working, and they are kept out of the
# published schema so a protocol client never discovers them. Shrinking this
# set is welcome — growing it is what the test forbids.
LEGACY_ALIASES = {
    "/v1/stats",
    "/v1/requests",
    "/v1/cache/status",
    "/v1/cache/rebuild",
    "/v1/admin/prepare-stop",
}


def _paths() -> set[str]:
    return {r.path for r in app.routes if getattr(r, "path", "").startswith("/v1")}


@pytest.mark.parametrize("route", sorted(OPENAI))
def test_openai_route_is_served(route):
    assert route in _paths()


@pytest.mark.parametrize("route", sorted(ANTHROPIC))
def test_anthropic_route_is_served(route):
    assert route in _paths()


def test_nothing_new_squats_in_the_protocol_namespace():
    unexpected = _paths() - OPENAI - ANTHROPIC - LEGACY_ALIASES
    assert not unexpected, (
        "these are ours, not protocol, and /v1 is the protocol's namespace — "
        f"serve them under /admin instead: {sorted(unexpected)}"
    )


def test_the_published_schema_under_v1_is_protocol_only():
    """What a client discovers, as opposed to what happens to answer.

    The legacy aliases still respond, but an OpenAI client reading /openapi.json
    must see the protocol and nothing else — otherwise a strict gateway or a
    code generator picks up endpoints that are none of its business.
    """
    published = {p for p in app.openapi()["paths"] if p.startswith("/v1")}
    # FastAPI renders path params without their converter.
    expected = {p.replace("{model_id:path}", "{model_id}") for p in OPENAI | ANTHROPIC}
    assert published == expected


@pytest.mark.parametrize("legacy", sorted(LEGACY_ALIASES))
def test_legacy_alias_still_answers_and_shares_its_handler(legacy):
    canonical = {
        "/v1/stats": "/admin/stats",
        "/v1/requests": "/admin/requests",
        "/v1/cache/status": "/admin/cache/status",
        "/v1/cache/rebuild": "/admin/cache/rebuild",
        "/v1/admin/prepare-stop": "/admin/prepare-stop",
    }[legacy]
    handlers = {}
    for route in app.routes:
        path = getattr(route, "path", None)
        if path in (legacy, canonical):
            handlers.setdefault(path, getattr(route, "endpoint", None))
    assert handlers.get(legacy) is not None, f"{legacy} stopped answering"
    assert handlers[legacy] is handlers[canonical], (
        f"{legacy} and {canonical} drifted apart"
    )


def test_the_protocol_routes_accept_their_methods():
    """A route that exists but rejects the protocol's verb is still broken."""
    methods = {
        r.path: set(getattr(r, "methods", set()) or set())
        for r in app.routes
        if getattr(r, "path", "").startswith("/v1")
    }
    for route in ("/v1/chat/completions", "/v1/completions", "/v1/messages",
                  "/v1/messages/count_tokens", "/v1/responses"):
        assert "POST" in methods[route], route
    for route in ("/v1/models", "/v1/models/{model_id:path}"):
        assert "GET" in methods[route], route
