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
        assert "No films pass these filters" in with_filter and "IMDb 9.5+" in with_filter
        without = client.get("/recommend", params={"mood": "x"}).text
        assert "Try a different vibe" in without


class TestWhichFilterEmptiedTheResults:
    CORPUS = [
        {
            "imdb_id": "tt1",
            "title": "Old Good",
            "year": "1975",
            "vote_average": 8.6,
            "language": "English",
        },
        {
            "imdb_id": "tt2",
            "title": "Old Okay",
            "year": "1978",
            "vote_average": 6.0,
            "language": "English",
        },
        {
            "imdb_id": "tt3",
            "title": "New Good",
            "year": "2015",
            "vote_average": 8.8,
            "language": "Korean",
        },
    ]

    @pytest.fixture(autouse=True)
    def _corpus(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import app.services.corpus as corpus_mod

        monkeypatch.setattr(corpus_mod, "get_corpus_and_embeddings", lambda: (self.CORPUS, None))
        monkeypatch.setattr(main_mod, "recommend_movies_advanced", lambda **kw: [])

    def test_counts_films_that_would_pass_without_each_filter(self) -> None:
        from app.services.recommender import films_without_each_filter

        counts = films_without_each_filter(
            min_year=1970, max_year=1980, min_rating=9.0, english_only=False
        )
        # Nothing is rated 9+. Dropping the rating leaves the two 1970s films;
        # dropping either year bound still leaves nothing.
        assert counts == {"min_year": 0, "max_year": 0, "min_rating": 2, "all": 3}

    def test_page_names_the_filter_and_what_removing_it_gains(self, client: TestClient) -> None:
        params = {"mood": "x", "min_year": 1970, "max_year": 1980, "min_rating": 9.0}
        page = client.get("/recommend", params=params).text
        assert "No films pass these filters" in page
        assert "Remove &ldquo;IMDb 9+&rdquo;" in page
        assert "2 films would qualify" in page
        # The year filters are not the culprit, so they are not offered.
        assert "Remove &ldquo;From 1970&rdquo;" not in page

    def test_singular_count_reads_correctly(self, client: TestClient) -> None:
        page = client.get("/recommend", params={"mood": "x", "min_year": 2010, "english": 1}).text
        assert "Remove &ldquo;English only&rdquo;" in page
        assert "1 film would qualify" in page

    def test_when_no_single_removal_helps_it_offers_to_clear_everything(
        self, client: TestClient
    ) -> None:
        params = {"mood": "x", "min_year": 1990, "max_year": 2000, "min_rating": 9.5}
        page = client.get("/recommend", params=params).text
        assert "Removing any single filter is not enough" in page
        assert "Clear all filters" in page and "all 3 films" in page

    def test_explaining_never_breaks_the_page(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(**kwargs: Any) -> dict[str, int]:
            raise RuntimeError("no corpus")

        monkeypatch.setattr(main_mod, "films_without_each_filter", _boom)
        resp = client.get("/recommend", params={"mood": "x", "min_rating": 9.9})
        assert resp.status_code == 200 and "No films pass these filters" in resp.text


class TestSuggestions:
    """One-click nudges: older, newer, more mainstream, more obscure."""

    def _by_label(self, ctx: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
        return {s["label"]: _query(s["url"]) for s in ctx["suggestions"]}

    def test_era_nudges_are_relative_to_the_films_on_screen(self) -> None:
        nudges = self._by_label(_ctx(years=[1973, 1978, 1981, 1984, 1992, 2019]))
        # Median 1982: older means before it, newer means after it.
        assert nudges["Older"]["max_year"] == ["1981"]
        assert nudges["Newer"]["min_year"] == ["1983"]
        assert nudges["Older"]["mood"] == ["folk horror"]

    def test_no_era_nudges_without_years(self) -> None:
        labels = [s["label"] for s in _ctx(years=[])["suggestions"]]
        assert labels == ["More mainstream", "More obscure"]

    def test_an_era_nudge_drops_a_filter_it_would_contradict(self) -> None:
        nudges = self._by_label(_ctx(years=[2015, 2018, 2020], min_year=2010, max_year=2022))
        assert nudges["Older"]["max_year"] == ["2017"] and nudges["Older"]["min_year"] == ["2010"]
        nudges = self._by_label(_ctx(years=[2015, 2018, 2020], min_year=2018))
        assert nudges["Older"]["max_year"] == ["2017"] and "min_year" not in nudges["Older"]
        nudges = self._by_label(_ctx(years=[1975, 1978, 1980], max_year=1978))
        assert nudges["Newer"]["min_year"] == ["1979"] and "max_year" not in nudges["Newer"]

    def test_other_filters_survive_a_nudge(self) -> None:
        nudges = self._by_label(_ctx(years=[1980], min_rating=7.0, english=True))
        assert nudges["Older"]["min_rating"] == ["7"] and nudges["Older"]["english"] == ["1"]

    def test_tilt_nudges_switch_to_the_unified_ranking(self) -> None:
        nudges = self._by_label(_ctx(strategy="semantic", years=[1980]))
        assert nudges["More obscure"]["tilt"] == ["obscure"]
        assert nudges["More obscure"]["strategy"] == ["unified"]

    def test_active_tilt_becomes_a_removable_chip_and_is_not_offered_again(self) -> None:
        ctx = _ctx(strategy="unified", tilt="obscure", years=[1980])
        assert [c["label"] for c in ctx["chips"]] == ["More obscure"]
        assert "tilt" not in _query(ctx["chips"][0]["remove_url"])
        labels = [s["label"] for s in ctx["suggestions"]]
        assert "More mainstream" in labels and "More obscure" not in labels
        assert ctx["params"]["tilt"] == "obscure"

    def test_unknown_tilt_is_ignored(self) -> None:
        ctx = _ctx(tilt="chaotic")
        assert ctx["chips"] == [] and "tilt" not in ctx["params"]


class TestTiltReachesTheRanker:
    @pytest.fixture()
    def weights_seen(self, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
        seen: list[Any] = []
        film = {**FILM, "_embedding_row": 0}
        monkeypatch.setattr(main_mod, "recommend_movies_advanced", lambda **kw: [dict(film)])

        def _unified(**kwargs: Any) -> list[dict[str, Any]]:
            seen.append(kwargs.get("weights"))
            return [dict(FILM)]

        monkeypatch.setattr(main_mod, "recommend_unified_semantic", _unified)
        return seen

    @pytest.mark.parametrize(("tilt", "popularity"), [("obscure", 0.0), ("mainstream", 0.45)])
    def test_tilt_sets_the_popularity_weight_and_keeps_the_others(
        self, client: TestClient, weights_seen: list[Any], tilt: str, popularity: float
    ) -> None:
        from app.services.unified_recommender import DEFAULT_WEIGHTS

        page = client.get("/recommend", params={"mood": "x", "tilt": tilt}).text
        weights = weights_seen[-1]
        assert weights["popularity"] == popularity
        assert weights["semantic"] == DEFAULT_WEIGHTS["semantic"]
        assert "Unified" in page  # a tilt implies the unified ranking

    def test_no_tilt_leaves_the_default_weights(
        self, client: TestClient, weights_seen: list[Any]
    ) -> None:
        client.get("/recommend", params={"mood": "x", "strategy": "unified"})
        assert weights_seen[-1] is None

    def test_bogus_tilt_is_a_normal_search(
        self, client: TestClient, weights_seen: list[Any]
    ) -> None:
        resp = client.get("/recommend", params={"mood": "x", "tilt": "<script>"})
        # The default strategy with its default weights: no tilt was applied.
        assert resp.status_code == 200 and weights_seen == [None]

    def test_page_shows_the_nudges_and_keeps_the_tilt_when_refining(
        self, client: TestClient, weights_seen: list[Any]
    ) -> None:
        page = client.get("/recommend", params={"mood": "x", "tilt": "obscure"}).text
        assert 'class="suggest-chip"' in page and ">Older</a>" in page
        assert '<input type="hidden" name="tilt" value="obscure">' in page
