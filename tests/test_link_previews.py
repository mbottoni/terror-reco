"""Pasted links unfurl with a title, a description and an image."""

from __future__ import annotations

import re
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.main as main_mod
from app.main import looks_like_bot
from app.models import Impression

BROWSER = {"User-Agent": "Mozilla/5.0 (Macintosh) AppleWebKit/537.36 Chrome/120 Safari/537.36"}
FILMS = [
    {"imdb_id": "tt0000001", "title": "The Wicker Man", "overview": "p", "keywords": "k"},
    {"imdb_id": "tt0000002", "title": 'Mid"sommar <b>', "overview": "p", "keywords": "k"},
]


def _meta(page: str, key: str) -> str:
    match = re.search(rf'<meta (?:property|name)="{re.escape(key)}" content="([^"]*)">', page)
    assert match, f"no {key} meta tag"
    return match.group(1)


@pytest.fixture()
def ranker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        main_mod, "recommend_movies_advanced", lambda **kw: [dict(f) for f in FILMS]
    )


class TestEveryPage:
    @pytest.mark.parametrize("path", ["/", "/auth/login", "/no-such-page"])
    def test_has_the_preview_tags_with_absolute_urls(self, client: TestClient, path: str) -> None:
        page = client.get(path).text
        assert _meta(page, "og:title")
        assert _meta(page, "og:description") == _meta(page, "description")
        assert _meta(page, "twitter:card") == "summary_large_image"
        for key in ("og:image", "twitter:image", "og:url"):
            assert _meta(page, key).startswith("http")
        assert _meta(page, "og:image").endswith("/static/og.png")

    def test_preview_image_is_a_real_png_of_the_declared_size(self, client: TestClient) -> None:
        resp = client.get("/static/og.png")
        assert resp.status_code == 200 and resp.content[:8] == b"\x89PNG\r\n\x1a\n"
        width = int.from_bytes(resp.content[16:20], "big")
        height = int.from_bytes(resp.content[20:24], "big")
        assert (width, height) == (1200, 630)

    def test_urls_use_the_configured_origin_not_the_host_header(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(main_mod.settings, "PUBLIC_BASE_URL", "https://real.example")
        page = client.get("/", headers={"Host": "evil.example"}).text
        assert _meta(page, "og:image") == "https://real.example/static/og.png"
        assert _meta(page, "og:url") == "https://real.example/"
        assert "evil.example" not in page


class TestResultsPage:
    def test_names_the_mood_and_lists_the_films(self, client: TestClient, ranker: None) -> None:
        page = client.get("/recommend", params={"mood": "folk horror"}, headers=BROWSER).text
        assert "folk horror" in _meta(page, "og:title")
        assert _meta(page, "og:description").startswith("The Wicker Man, Mid")
        assert _meta(page, "twitter:title") == _meta(page, "og:title")

    def test_og_url_keeps_the_query_so_a_shared_set_stays_shared(
        self, client: TestClient, ranker: None
    ) -> None:
        page = client.get("/recommend", params={"mood": "folk horror", "ids": "tt0000001"}).text
        assert "ids=tt0000001" in _meta(page, "og:url")

    def test_hostile_text_cannot_break_out_of_the_attribute(
        self, client: TestClient, ranker: None
    ) -> None:
        mood = '"><script>alert(1)</script>'
        page = client.get("/recommend", params={"mood": mood}, headers=BROWSER).text
        head = page[: page.index("</head>")]
        assert "<script>alert(1)</script>" not in head
        assert '<b>"' not in head and "<b>" not in _meta(page, "og:description")

    def test_long_moods_are_truncated(self, client: TestClient, ranker: None) -> None:
        page = client.get("/recommend", params={"mood": "x" * 400}, headers=BROWSER).text
        assert len(_meta(page, "og:title")) < 140


class TestCrawlersAreNotAudience:
    @pytest.mark.parametrize(
        "agent",
        [
            "Slackbot-LinkExpanding 1.0 (+https://api.slack.com/robots)",
            "facebookexternalhit/1.1",
            "Twitterbot/1.0",
            "Mozilla/5.0 (compatible; Discordbot/2.0; +https://discordapp.com)",
            "WhatsApp/2.23",
            "curl/8.4.0",
            "",
        ],
    )
    def test_preview_fetchers_do_not_record_an_impression(
        self, client: TestClient, ranker: None, db: Any, agent: str
    ) -> None:
        page = client.get("/recommend", params={"mood": "x"}, headers={"User-Agent": agent}).text
        assert "var RESULT_KEY = null" in page
        assert db.query(Impression).count() == 0

    def test_a_browser_does(self, client: TestClient, ranker: None, db: Any) -> None:
        client.get("/recommend", params={"mood": "x"}, headers=BROWSER)
        assert db.query(Impression).count() == 1

    def test_classifier(self) -> None:
        class _Req:
            def __init__(self, agent: str) -> None:
                self.headers = {"user-agent": agent}

        assert looks_like_bot(_Req("Googlebot/2.1"))  # type: ignore[arg-type]
        assert not looks_like_bot(_Req(BROWSER["User-Agent"]))  # type: ignore[arg-type]
