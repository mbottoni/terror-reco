"""A shared link shows the films that were on screen, not a new draw."""

from __future__ import annotations

import re
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.main as main_mod
import app.services.corpus as corpus_mod
from app.models import SearchHistory
from app.services.recommender import MAX_SHARED_FILMS, films_by_ids

CORPUS = [
    {"imdb_id": f"tt000000{i}", "title": f"Film {i}", "overview": "p", "keywords": "k"}
    for i in range(1, 8)
]


@pytest.fixture(autouse=True)
def _corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(corpus_mod, "get_corpus_and_embeddings", lambda: (CORPUS, None))


def _card_titles(page: str) -> list[str]:
    return re.findall(r'<h2 class="card-title">([^<]+)</h2>', page)


class TestFilmsByIds:
    def test_returns_films_in_the_order_asked(self) -> None:
        films = films_by_ids(["tt0000003", "tt0000001", "tt0000002"])
        assert [f["title"] for f in films] == ["Film 3", "Film 1", "Film 2"]

    def test_skips_unknown_malformed_and_duplicate_ids(self) -> None:
        ids = ["tt0000001", "tt9999999", "nonsense", "'; drop table", "tt0000001", " tt0000002 "]
        assert [f["imdb_id"] for f in films_by_ids(ids)] == ["tt0000001", "tt0000002"]

    def test_is_capped(self) -> None:
        many = [f"tt{n:07d}" for n in range(100, 100 + MAX_SHARED_FILMS + 30)]
        assert len(films_by_ids(many + ["tt0000001"])) == 0  # the cap is hit before the real id

    def test_nothing_valid_means_nothing(self) -> None:
        assert films_by_ids([]) == [] and films_by_ids(["", "x"]) == []


class TestSharedLink:
    @pytest.fixture()
    def ranker_calls(self, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []

        def _fake(**kwargs: Any) -> list[dict[str, Any]]:
            calls.append(kwargs)
            return [dict(CORPUS[6])]

        monkeypatch.setattr(main_mod, "recommend_movies_advanced", _fake)
        return calls

    def test_shows_exactly_the_listed_films_without_ranking(
        self, client: TestClient, ranker_calls: list[dict[str, Any]]
    ) -> None:
        resp = client.get(
            "/recommend", params={"mood": "anything", "ids": "tt0000005,tt0000002,tt0000004"}
        )
        assert resp.status_code == 200
        assert _card_titles(resp.text) == ["Film 5", "Film 2", "Film 4"]
        assert ranker_calls == []
        assert "shared set" in resp.text

    def test_unresolvable_ids_fall_back_to_a_normal_search(
        self, client: TestClient, ranker_calls: list[dict[str, Any]]
    ) -> None:
        resp = client.get("/recommend", params={"mood": "anything", "ids": "tt9999999,junk"})
        assert _card_titles(resp.text) == ["Film 7"]
        assert len(ranker_calls) == 1
        assert "shared set" not in resp.text

    def test_share_button_builds_the_link_from_the_films_on_screen(
        self, client: TestClient, ranker_calls: list[dict[str, Any]]
    ) -> None:
        page = client.get("/recommend", params={"mood": "anything"}).text
        script = page[page.index("function shareResults") :]
        script = script[: script.index("var btn")]
        assert "searchParams.set('ids'" in script and "m.imdb_id" in script
        assert "seed" not in script.replace("A seed would not", "").replace("a seeded request", "")

    def test_opening_a_shared_set_is_not_saved_as_the_viewers_search(
        self, client: TestClient, ranker_calls: list[dict[str, Any]], db: Any
    ) -> None:
        page = client.get("/auth/register").text
        csrf = re.search(r'name="csrf" value="([^"]+)"', page).group(1)  # type: ignore[union-attr]
        data = {
            "email": "viewer@example.com",
            "password": "password123",
            "confirm_password": "password123",
            "csrf": csrf,
        }
        client.post("/auth/register", data=data)
        client.get("/recommend", params={"mood": "shared", "ids": "tt0000001"})
        assert db.query(SearchHistory).count() == 0
        client.get("/recommend", params={"mood": "mine"})
        assert [h.mood for h in db.query(SearchHistory).all()] == ["mine"]

    def test_overlong_ids_parameter_is_rejected(self, client: TestClient) -> None:
        resp = client.get("/recommend", params={"mood": "x", "ids": "tt0000001," * 60})
        assert resp.status_code == 422
