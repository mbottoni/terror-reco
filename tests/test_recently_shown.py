"""Repeating a search brings different films, without ever emptying the page."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as main_mod
import app.services.corpus as corpus_mod
import app.services.unified_recommender as unified
from app.main import RECENT_FILMS_PER_MOOD, RECENT_MOODS, app
from app.services.recommender import recommend_movies_advanced

BROWSER = {"User-Agent": "Mozilla/5.0 Chrome/120"}


def _corpus(n: int) -> list[dict[str, Any]]:
    return [
        {"imdb_id": f"tt{i:07d}", "title": f"Film {i:02d}", "overview": "p", "keywords": "k"}
        for i in range(n)
    ]


@pytest.fixture()
def corpus30(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Thirty films whose similarity to any query falls with their index."""
    corpus = _corpus(30)
    vecs = np.zeros((30, 4), dtype=np.float32)
    vecs[:, 0] = np.linspace(1.0, 0.1, 30)
    monkeypatch.setattr(corpus_mod, "get_corpus_and_embeddings", lambda: (corpus, vecs))
    monkeypatch.setattr(
        unified, "_embed_sbert", lambda texts: np.tile([1.0, 0, 0, 0], (len(texts), 1)).astype("f")
    )
    monkeypatch.setattr(unified, "_QUERY_CACHE", unified.OrderedDict())
    return corpus


@pytest.fixture()
def demotions(monkeypatch: pytest.MonkeyPatch) -> list[set[str]]:
    """Record the films each search is told to demote.

    The ranker is replaced with one that returns six new films per call, so
    these tests are exact: they check what the route remembers and passes on,
    without depending on a random draw.
    """
    seen: list[set[str]] = []
    counter = iter(range(10_000))

    def _fake(**kwargs: Any) -> list[dict[str, Any]]:
        seen.append(set(kwargs.get("demote_ids") or set()))
        return [
            {"imdb_id": f"tt{next(counter):07d}", "title": "F", "overview": "p", "keywords": "k"}
            for _ in range(6)
        ]

    monkeypatch.setattr(main_mod, "recommend_movies_advanced", _fake)
    return seen


def _titles(page: str) -> list[str]:
    return re.findall(r'<h2 class="card-title">([^<]+)</h2>', page)


def _search(client: Any, mood: str = "folk horror", **params: Any) -> list[str]:
    resp = client.get("/recommend", params={"mood": mood, **params}, headers=BROWSER)
    assert resp.status_code == 200
    return _titles(resp.text)


class TestDemotionInTheSemanticStrategy:
    def test_demoted_films_move_behind_fresh_ones(self, corpus30: Any) -> None:
        top = {f"tt{i:07d}" for i in range(6)}
        picked = recommend_movies_advanced(mood="x", limit=6, temperature=0, demote_ids=top)
        assert [m["title"] for m in picked] == [f"Film {i:02d}" for i in range(6, 12)]

    def test_page_is_still_full_when_everything_was_seen(self, corpus30: Any) -> None:
        everything = {m["imdb_id"] for m in corpus30}
        picked = recommend_movies_advanced(mood="x", limit=6, temperature=0, demote_ids=everything)
        assert [m["title"] for m in picked] == [f"Film {i:02d}" for i in range(6)]

    def test_no_demotion_means_no_change(self, corpus30: Any) -> None:
        a = recommend_movies_advanced(mood="x", limit=6, temperature=0)
        b = recommend_movies_advanced(mood="x", limit=6, temperature=0, demote_ids=set())
        assert a == b


class TestRepeatSearches:
    @pytest.mark.parametrize("strategy", ["semantic", "unified"])
    def test_consecutive_searches_for_one_mood_do_not_overlap(
        self, client: Any, corpus30: Any, strategy: str
    ) -> None:
        first = _search(client, strategy=strategy)
        second = _search(client, strategy=strategy)
        third = _search(client, strategy=strategy)
        assert len(first) == len(second) == len(third) == 6
        assert not set(first) & set(second)
        assert not (set(first) | set(second)) & set(third)

    def test_a_different_mood_is_not_affected(self, client: Any, demotions: list[set[str]]) -> None:
        _search(client, mood="folk horror")
        _search(client, mood="folk horror")
        _search(client, mood="space horror")
        _search(client, mood="  Folk Horror ")  # the same mood, typed differently
        assert demotions[0] == set()
        assert len(demotions[1]) == 6  # the first page, now sitting out
        assert demotions[2] == set()  # a new mood starts with a clean slate
        assert len(demotions[3]) == 12  # ...and the first mood's two pages are both remembered

    def test_films_come_back_after_a_few_searches(
        self, client: Any, demotions: list[set[str]]
    ) -> None:
        """Memory is three pages deep, so nothing is hidden for long."""
        for _ in range(5):
            _search(client)
        sizes = [len(d) for d in demotions]
        assert sizes == [0, 6, 12, 18, 18]  # grows to the cap, then stays there
        first_page = demotions[1]
        assert first_page <= demotions[3]  # still sitting out on the fourth search
        assert not first_page & demotions[4]  # eligible again on the fifth

    def test_seeded_requests_stay_reproducible(self, client: Any, corpus30: Any) -> None:
        _search(client)  # leaves a memory behind
        a = _search(client, seed=7)
        b = _search(client, seed=7)
        assert a == b == [f"Film {i:02d}" for i in range(6)]

    def test_shared_links_ignore_and_do_not_feed_the_memory(
        self, client: Any, corpus30: Any
    ) -> None:
        ids = "tt0000000,tt0000001"
        assert _search(client, ids=ids) == ["Film 00", "Film 01"]
        assert _search(client, ids=ids) == ["Film 00", "Film 01"]

    def test_two_visitors_do_not_share_a_memory(
        self, client: Any, demotions: list[set[str]]
    ) -> None:
        _search(client)
        _search(client)
        with TestClient(app) as other:
            other.get("/recommend", params={"mood": "folk horror"}, headers=BROWSER)
        assert len(demotions[1]) == 6
        assert demotions[2] == set()  # the second visitor starts from nothing


class TestMemoryStaysSmall:
    class _Req:
        def __init__(self) -> None:
            self.session: dict[str, Any] = {}

    def test_capped_per_mood_and_in_number_of_moods(self) -> None:
        req = self._Req()
        for page in range(10):
            films = [{"imdb_id": f"tt{page:03d}{i:04d}"} for i in range(6)]
            main_mod.remember_shown(req, "one mood", films)  # type: ignore[arg-type]
        assert len(main_mod.recently_shown(req, "one mood")) == RECENT_FILMS_PER_MOOD  # type: ignore[arg-type]
        for n in range(RECENT_MOODS + 3):
            main_mod.remember_shown(req, f"mood {n}", [{"imdb_id": "tt0000001"}])  # type: ignore[arg-type]
        assert len(req.session["seen"]) == RECENT_MOODS
        assert main_mod.recently_shown(req, "one mood") == set()  # type: ignore[arg-type]

    def test_mood_key_ignores_case_and_spacing(self) -> None:
        req = self._Req()
        main_mod.remember_shown(req, "Folk Horror ", [{"imdb_id": "tt0000001"}])  # type: ignore[arg-type]
        assert main_mod.recently_shown(req, "folk horror") == {"tt0000001"}  # type: ignore[arg-type]

    def test_corrupt_session_value_is_ignored(self) -> None:
        req = self._Req()
        req.session["seen"] = "not a dict"
        assert main_mod.recently_shown(req, "x") == set()  # type: ignore[arg-type]
        main_mod.remember_shown(req, "x", [{"imdb_id": "tt0000001"}])  # type: ignore[arg-type]
        assert main_mod.recently_shown(req, "x") == {"tt0000001"}  # type: ignore[arg-type]

    def test_session_cookie_stays_well_under_the_browser_limit(
        self, client: Any, corpus30: Any
    ) -> None:
        for mood in ("a", "b", "c", "d"):
            for _ in range(4):
                _search(client, mood=mood)
        cookie = client.cookies.get("terror_session") or ""
        assert 0 < len(cookie) < 2500
