"""The browser-tab icon."""

from fastapi.testclient import TestClient


def test_favicon_ico_is_served_at_the_root(client: TestClient) -> None:
    resp = client.get("/favicon.ico")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/x-icon"
    assert resp.content[:4] == b"\x00\x00\x01\x00"  # ICO header


def test_pages_link_the_icons_and_the_files_exist(client: TestClient) -> None:
    html = client.get("/").text
    assert 'rel="icon"' in html and 'rel="apple-touch-icon"' in html
    for path in ("/static/favicon.svg", "/static/apple-touch-icon.png"):
        assert path in html
        assert client.get(path).status_code == 200
