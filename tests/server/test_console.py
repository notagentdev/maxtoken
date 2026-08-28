"""The built-in web console: served from the package, no backend state needed."""

from fastapi.testclient import TestClient

from maxtoken.server.api_server import app


def test_console_served():
    client = TestClient(app)
    r = client.get("/console")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "MaxToken Console" in r.text
    # the page drives exactly these same-origin endpoints, and reaches our own
    # ones under /admin rather than in the protocols' /v1 namespace
    for path in ("/health", "/admin/stats", "/admin/cache/status", "/admin/requests"):
        assert path in r.text


def test_a_waiting_turn_shows_a_spinner_and_a_clock():
    """An empty assistant bubble reads as a hung server.

    A long prompt or a reasoning model can take seconds before its first token,
    so the bubble carries a spinner and a running clock until then, and the
    clock has to be stopped on every exit — including an abort or an HTTP error,
    or it ticks forever in a bubble nobody is filling.
    """
    page = TestClient(app).get("/console").text
    assert ".spinner" in page and "@keyframes spin" in page
    assert "Processing prompt" in page
    assert "prefers-reduced-motion" in page, "an infinite spinner needs the opt-out"
    # started once, cleared on the first content token and again in `finally`
    assert page.count("endStatus()") >= 2
    assert "clearInterval(ticker)" in page


def test_root_redirects_to_console():
    client = TestClient(app)
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/console"


def test_generation_defaults_live_in_the_console_tab_not_the_chat():
    """Temperature and the output cap are server-wide defaults, edited from the
    Console tab and applied through /admin/cache/rebuild; the chat sends
    neither, so it gets exactly what an agent or SDK gets."""
    page = TestClient(app).get("/console").text
    console, chat = page.split('id="chatView"', 1)
    assert 'id="genTemp"' in console and 'id="genMax"' in console and 'id="genApply"' in console
    assert 'id="maxTok"' not in page and 'id="temp"' not in page
    assert "max_output_tokens" in page
    # the chat request carries neither field any more
    assert 'temperature: parseFloat($("temp")' not in page
    assert 'max_tokens: parseInt($("maxTok")' not in page
