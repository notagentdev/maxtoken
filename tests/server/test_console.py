"""The built-in web console: served from the package, no backend state needed."""

from fastapi.testclient import TestClient

from freetoken.server.api_server import app


def test_console_served():
    client = TestClient(app)
    r = client.get("/console")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "MaxToken Console" in r.text
    # the page drives exactly these same-origin endpoints
    for path in ("/health", "/v1/stats", "/v1/cache/status", "/v1/requests"):
        assert path in r.text


def test_root_redirects_to_console():
    client = TestClient(app)
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/console"
