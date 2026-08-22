"""Guards on how film text reaches the page.

Film data comes from TMDB/OMDb. Every fragment the detail modal builds used to
be raw string concatenation into ``innerHTML``, which renders wrong for any
title containing ``&`` or a quote and is an injection path as soon as that text
becomes user-influenced.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.main as main_mod
from app.security import safe_url

RESULTS_TEMPLATE = Path(main_mod.TEMPLATES_DIR) / "results.html"

HOSTILE_TITLE = "<script>alert('xss')</script> & \"quotes\""


class TestNoRawInterpolation:
    def test_film_fields_are_never_concatenated_into_markup_unescaped(self) -> None:
        """Only ``imdb_id`` may be spliced in raw, and only because it is validated.

        ``isImdbId()`` rejects anything that is not ``tt<digits>``, which is a
        stronger guarantee than escaping. Every other field must go through
        ``esc()``, ``safeUrl()`` or the DOM API.
        """
        source = RESULTS_TEMPLATE.read_text()
        raw = re.findall(r"\+\s*(?:m|f)\.(\w+)\s*\+", source)
        offenders = sorted({field for field in raw if field != "imdb_id"})
        assert not offenders, f"film fields concatenated into markup unescaped: {offenders}"

    def test_the_escaping_helpers_exist(self) -> None:
        source = RESULTS_TEMPLATE.read_text()
        for helper in ("function esc(", "function safeUrl(", "function isImdbId("):
            assert helper in source


class TestResultsPageRendering:
    @pytest.fixture()
    def _hostile_film(self, monkeypatch: pytest.MonkeyPatch) -> None:
        film: dict[str, Any] = {
            "imdb_id": "tt0000001",
            "title": HOSTILE_TITLE,
            "overview": "A plot with <b>markup</b> in it.",
            "poster_url": "javascript:alert(1)",
            "director": "Ridley & Tony",
            "keywords": "isolation",
            "year": "1979",
        }

        def _fake(**kwargs: Any) -> list[dict[str, Any]]:
            return [film]

        monkeypatch.setattr(main_mod, "recommend_movies_advanced", _fake)

    def test_hostile_film_text_is_escaped_server_side(
        self, client: TestClient, _hostile_film: Any
    ) -> None:
        resp = client.get("/recommend", params={"mood": "anything", "strategy": "semantic"})
        assert resp.status_code == 200
        # Jinja escapes it in the card markup and JSON-escapes it in the MOVIES
        # array; either way the raw tag must not survive into the document.
        assert "<script>alert" not in resp.text

    def test_a_hostile_poster_url_never_reaches_an_attribute(
        self, client: TestClient, _hostile_film: Any
    ) -> None:
        """The URL may appear as JS *data* -- it must not appear as a *URL*.

        ``safeUrl()`` gates it client-side before any assignment to ``.src``,
        and the ``safe_url`` Jinja filter gates the server-rendered card.
        """
        resp = client.get("/recommend", params={"mood": "anything", "strategy": "semantic"})
        assert 'src="javascript:' not in resp.text
        assert 'href="javascript:' not in resp.text
        # With no usable poster the card must fall back to the placeholder.
        assert "card-no-poster" in resp.text


class TestSafeUrl:
    """Autoescaping protects text, not URL schemes -- this covers the gap."""

    @pytest.mark.parametrize(
        "url",
        [
            "https://image.tmdb.org/t/p/w500/abc.jpg",
            "http://example.com/poster.jpg",
            "/static/assets/spooky.gif",
        ],
    )
    def test_allows_http_and_same_origin(self, url: str) -> None:
        assert safe_url(url) == url

    @pytest.mark.parametrize(
        "url",
        [
            "javascript:alert(1)",
            "JavaScript:alert(1)",
            "  javascript:alert(1)  ",
            "data:text/html;base64,PHNjcmlwdD4=",
            "vbscript:msgbox(1)",
            "//evil.example.com/poster.jpg",  # protocol-relative: another origin
            "",
            None,
        ],
    )
    def test_rejects_everything_else(self, url: str | None) -> None:
        assert safe_url(url) == ""
