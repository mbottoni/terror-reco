"""The stylesheet is a cached static file, and fonts are served locally."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.static_assets as assets
from app.static_assets import static_url

IMMUTABLE = "public, max-age=31536000, immutable"


def _stylesheet_href(client: TestClient, path: str = "/") -> str:
    match = re.search(r'<link rel="stylesheet" href="([^"]+)">', client.get(path).text)
    assert match, "page must link the stylesheet"
    return match.group(1)


class TestPages:
    @pytest.mark.parametrize("path", ["/", "/auth/login", "/no-such-page"])
    def test_link_a_fingerprinted_stylesheet_and_inline_none(
        self, client: TestClient, path: str
    ) -> None:
        page = client.get(path).text
        assert re.search(r'href="/static/styles\.css\?v=[0-9a-f]{12}"', page)
        assert "<style>" not in page

    def test_make_no_third_party_font_request(self, client: TestClient) -> None:
        page = client.get("/").text
        assert "fonts.googleapis.com" not in page and "fonts.gstatic.com" not in page
        assert 'rel="preload" href="/static/fonts/inter-latin-v20.woff2"' in page


class TestStylesheet:
    def test_is_served_with_a_long_immutable_cache(self, client: TestClient) -> None:
        resp = client.get(_stylesheet_href(client))
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/css")
        assert resp.headers["cache-control"] == IMMUTABLE
        assert ".card{" in resp.text and "@font-face" in resp.text

    def test_without_the_fingerprint_it_is_not_immutable(self, client: TestClient) -> None:
        resp = client.get("/static/styles.css")
        assert resp.status_code == 200
        assert resp.headers.get("cache-control") != IMMUTABLE

    def test_every_font_it_references_exists(self, client: TestClient) -> None:
        css = client.get("/static/styles.css").text
        fonts = re.findall(r'url\("(fonts/[^"]+)"\)', css)
        assert len(fonts) >= 2
        for font in fonts:
            resp = client.get(f"/static/{font}")
            assert resp.status_code == 200
            assert resp.content[:4] == b"wOF2"
            assert resp.headers["cache-control"] == IMMUTABLE


class TestStaticUrl:
    def test_changes_when_the_file_changes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import os

        monkeypatch.setattr(assets, "STATIC_DIR", tmp_path)
        monkeypatch.setattr(assets, "_HASH_CACHE", {})
        file = tmp_path / "a.css"
        file.write_text("body{color:red}")
        first = static_url("a.css")
        assert static_url("a.css") == first
        file.write_text("body{color:blue}")
        os.utime(file, (file.stat().st_atime, file.stat().st_mtime + 5))
        assert static_url("a.css") != first

    def test_missing_file_gets_a_plain_url(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(assets, "STATIC_DIR", tmp_path)
        assert static_url("nope.css") == "/static/nope.css"

    def test_missing_static_file_is_404_not_cached(self, client: TestClient) -> None:
        resp = client.get("/static/nope.css?v=abc")
        assert resp.status_code == 404
        assert resp.headers.get("cache-control") != IMMUTABLE
