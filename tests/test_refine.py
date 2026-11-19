"""Adjusting a search from the results page, without starting over."""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient

import app.main as main_mod
from app.main import refine_context

FILM: dict[str, Any] = {
    "imdb_id": "tt0000001",
    "title": "Some Film",
    "overview": "A plot.",
    "poster_url": "https://example.com/p.jpg",
    "keywords": "isolation",
    "year": "1979",
}


def _ctx(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "mood": "folk horror",
        "strategy": "semantic",
        "min_year": None,
        "max_year": None,
        "min_rating": None,
        "limit": 6,
        "english": False,
    }
    return refine_context(**{**base, **overrides})


def _query(url: str) -> dict[str, list[str]]:
    return parse_qs(urlsplit(url).query)


class TestRefineContext:
    def test_no_filters_means_no_chips(self) -> None:
        ctx = _ctx()
        assert ctx["chips"] == []
        assert ctx["params"] == {"strategy": "semantic"}

    def test_each_active_filter_gets_a_chip(self) -> None:
        ctx = _ctx(min_year=1980, max_year=1999, min_rating=6.5, english=True, limit=12)
        labels = [chip["label"] for chip in ctx["chips"]]
        assert labels == ["From 1980", "Up to 1999", "IMDb 6.5+", "English only", "12 results"]

    def test_removing_a_chip_drops_only_that_filter(self) -> None:
        ctx = _ctx(min_year=1980, min_rating=7.0, english=True)
        by_label = {chip["label"]: _query(chip["remove_url"]) for chip in ctx["chips"]}
        without_year = by_label["From 1980"]
        assert "min_year" not in without_year
        assert without_year["min_rating"] == ["7"]
        assert without_year["english"] == ["1"]
        assert without_year["mood"] == ["folk horror"]
        assert without_year["strategy"] == ["semantic"]

    def test_shuffle_repeats_the_search_without_a_seed(self) -> None:
        query = _query(_ctx(min_year=1980)["shuffle_url"])
        assert query["mood"] == ["folk horror"] and query["min_year"] == ["1980"]
        assert "seed" not in query

    def test_mood_is_url_encoded(self) -> None:
        ctx = _ctx(mood='cabin & "woods" ?x=1', min_year=1980)
        assert _query(ctx["shuffle_url"])["mood"] == ['cabin & "woods" ?x=1']
        assert _query(ctx["chips"][0]["remove_url"])["mood"] == ['cabin & "woods" ?x=1']


class TestResultsPage:
    @pytest.fixture()
    def calls(self, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
        seen: list[dict[str, Any]] = []

        def _fake(**kwargs: Any) -> list[dict[str, Any]]:
            seen.append(kwargs)
            return [dict(FILM)]

        monkeypatch.setattr(main_mod, "recommend_movies_advanced", _fake)
        return seen

    def test_search_bar_is_prefilled_and_keeps_the_filters(
        self, client: TestClient, calls: list[dict[str, Any]]
    ) -> None:
        params = {"mood": "folk horror", "strategy": "semantic", "min_year": 1980, "english": 1}
        page = client.get("/recommend", params=params).text
        assert 'id="refine-mood" name="mood" type="text" value="folk horror"' in page
        assert '<input type="hidden" name="min_year" value="1980">' in page
        assert '<input type="hidden" name="english" value="1">' in page
        assert '<option value="semantic" selected>' in page

    def test_chips_and_shuffle_are_rendered(
        self, client: TestClient, calls: list[dict[str, Any]]
    ) -> None:
        params = {"mood": "folk horror", "min_rating": 6.5, "seed": 42}
        page = client.get("/recommend", params=params).text
        assert "IMDb 6.5+" in page
        assert "Shuffle" in page
        shuffle = page[page.index('class="share-btn refine-shuffle"') :]
        assert "seed" not in shuffle[: shuffle.index(">")]

    def test_hostile_mood_is_escaped_in_the_search_bar(
        self, client: TestClient, calls: list[dict[str, Any]]
    ) -> None:
        page = client.get("/recommend", params={"mood": '"><script>alert(1)</script>'}).text
        assert "<script>alert(1)</script>" not in page

    def test_following_a_remove_link_runs_the_search_without_that_filter(
        self, client: TestClient, calls: list[dict[str, Any]]
    ) -> None:
        ctx = _ctx(min_year=1980, min_rating=7.0)
        remove_year = next(c["remove_url"] for c in ctx["chips"] if c["label"] == "From 1980")
        assert client.get(remove_year).status_code == 200
        assert calls[-1]["min_year"] is None
        assert calls[-1]["min_rating"] == 7.0

    def test_empty_result_with_filters_points_at_the_filters(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(main_mod, "recommend_movies_advanced", lambda **kw: [])
        with_filter = client.get("/recommend", params={"mood": "x", "min_rating": 9.5}).text
        assert "with these filters" in with_filter and "IMDb 9.5+" in with_filter
        without = client.get("/recommend", params={"mood": "x"}).text
        assert "Try a different vibe" in without
