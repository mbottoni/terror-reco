"""What a visitor is served is the top of the ranking, not a random draw from it.

The final step used to pick ``limit`` films at random from the top 3 x limit.
Measured as served, that cost a third of the page's quality. These tests pin
the replacement: the best-ranked films, with variety left to the score noise
and to demoting what was just shown.
"""

from __future__ import annotations

from typing import Any

import pytest

import app.services.corpus as corpus_mod
from app.services.corpus import load_corpus
from app.services.recommender import recommend_movies_advanced
from app.services.unified_recommender import first_of_each_collection

pytestmark = pytest.mark.skipif(not load_corpus(), reason="corpus not built in this environment")


def _fake_search(calls: list[dict[str, Any]]) -> Any:
    """A stand-in for semantic_search that ranks the corpus in file order."""

    def search(
        query: str,
        corpus: list[dict[str, Any]],
        embeddings: Any,
        top_k: int = 60,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        calls.append({"top_k": top_k, "temperature": temperature})
        return [
            {**movie, "_semantic_score": 1.0 - i / 1000, "_embedding_row": i}
            for i, movie in enumerate(corpus[:top_k])
        ]

    return search


@pytest.fixture()
def search_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(corpus_mod, "semantic_search", _fake_search(calls))
    monkeypatch.setattr(corpus_mod, "get_corpus_and_embeddings", lambda: (load_corpus(), None))
    return calls


def _expected_top(limit: int) -> list[str]:
    ranked = load_corpus()[:60]
    return [ranked[i]["imdb_id"] for i in first_of_each_collection(ranked)][:limit]


def test_the_page_is_the_top_of_the_ranking_even_with_randomness_on(
    search_calls: list[dict[str, Any]],
) -> None:
    for seed in range(5):
        page = recommend_movies_advanced(mood="anything", limit=6, seed=seed, temperature=1.0)
        assert [m["imdb_id"] for m in page] == _expected_top(6)


def test_just_seen_films_give_way_to_the_next_best(search_calls: list[dict[str, Any]]) -> None:
    first = _expected_top(6)
    page = recommend_movies_advanced(mood="anything", limit=6, demote_ids=set(first))
    assert [m["imdb_id"] for m in page] == _expected_top(12)[6:]


def test_the_candidate_pool_does_not_grow_with_the_limit(
    search_calls: list[dict[str, Any]],
) -> None:
    """The unified strategy asks for 60; it used to be given the whole corpus."""
    pool = recommend_movies_advanced(mood="anything", limit=60, keep_internal=True)
    assert search_calls[-1]["top_k"] == 60
    assert [m["imdb_id"] for m in pool] == [m["imdb_id"] for m in load_corpus()[:60]]


def test_the_api_seed_makes_results_reproducible(client: Any) -> None:
    def titles(seed: int) -> list[str]:
        body = client.get("/api/recommendations", params={"mood": "haunted house", "seed": seed})
        return [m["title"] for m in body.json()["results"]]

    assert titles(1) == titles(1) == titles(2)


def test_without_a_seed_the_api_still_returns_a_full_page(client: Any) -> None:
    body = client.get("/api/recommendations", params={"mood": "haunted house"}).json()
    assert body["count"] == 6
