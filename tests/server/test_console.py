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


def test_root_redirects_to_console():
    client = TestClient(app)
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/console"
