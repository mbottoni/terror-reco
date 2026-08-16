"""Tests for match explanations and item-to-item similarity."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from app.services import corpus as corpus_mod
from app.services.recommender import explain_match, similar_movies


class TestExplainMatch:
    def test_reports_only_keywords_the_query_actually_hit(self) -> None:
        movie = {"keywords": "folk horror, pagan ritual, isolation, romance"}
        assert explain_match("eerie folk horror rituals", movie) == [
            "folk horror",
            "pagan ritual",
        ]

    def test_returns_nothing_when_no_keywords_match(self) -> None:
        assert explain_match("zombie apocalypse", {"keywords": "vampire, gothic"}) == []

    def test_handles_missing_keywords_field(self) -> None:
        assert explain_match("anything", {}) == []

    def test_stopword_only_query_explains_nothing(self) -> None:
        """Otherwise every film would 'match' on 'the' and look explained."""
        assert explain_match("the and of in", {"keywords": "the thing, and then"}) == []

    def test_caps_the_number_of_terms(self) -> None:
        movie = {"keywords": "horror, horror film, horror movie, horror story, horror tale"}
        assert len(explain_match("horror", movie, max_terms=2)) == 2


@pytest.mark.asyncio
class TestSimilarMovies:
    async def test_returns_neighbours_and_excludes_the_seed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_corpus: list[dict[str, Any]] = [
            {"imdb_id": "tt1", "title": "Seed"},
            {"imdb_id": "tt2", "title": "Near"},
            {"imdb_id": "tt3", "title": "Far"},
        ]
        # tt2 points the same way as tt1; tt3 is orthogonal.
        embs = np.array([[1.0, 0.0], [0.96, 0.28], [0.0, 1.0]], dtype=np.float32)
        monkeypatch.setattr(corpus_mod, "get_corpus_and_embeddings", lambda: (fake_corpus, embs))

        result = await similar_movies(imdb_id="tt1", limit=2)
        titles = [m["title"] for m in result]
        assert "Seed" not in titles, "a film must never be similar to itself"
        assert titles[0] == "Near"

    async def test_unknown_id_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            corpus_mod,
            "get_corpus_and_embeddings",
            lambda: ([{"imdb_id": "tt1", "title": "X"}], np.array([[1.0, 0.0]], dtype=np.float32)),
        )
        assert await similar_movies(imdb_id="nope") == []

    async def test_empty_corpus_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(corpus_mod, "get_corpus_and_embeddings", lambda: ([], np.zeros((0, 0))))
        assert await similar_movies(imdb_id="tt1") == []

    async def test_internal_fields_are_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = [{"imdb_id": "tt1", "title": "A"}, {"imdb_id": "tt2", "title": "B", "_row": 1}]
        embs = np.array([[1.0, 0.0], [0.9, 0.4]], dtype=np.float32)
        monkeypatch.setattr(corpus_mod, "get_corpus_and_embeddings", lambda: (fake, embs))
        result = await similar_movies(imdb_id="tt1", limit=1)
        assert all(not k.startswith("_") for m in result for k in m)
