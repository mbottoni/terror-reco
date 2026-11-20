"""Query embeddings are computed once per distinct query."""

from __future__ import annotations

import numpy as np
import pytest

import app.services.unified_recommender as unified


@pytest.fixture()
def embedder(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def _fake(texts: list[str]) -> np.ndarray:
        calls.append(list(texts))
        return np.full((len(texts), 4), float(len(calls)), dtype=np.float32)

    monkeypatch.setattr(unified, "_embed_sbert", _fake)
    monkeypatch.setattr(unified, "_QUERY_CACHE", unified.OrderedDict())
    return calls


class TestEmbedQuery:
    def test_repeat_queries_hit_the_model_once(self, embedder: list[list[str]]) -> None:
        first = unified.embed_query("folk horror")
        second = unified.embed_query("folk horror")
        assert len(embedder) == 1
        assert first is second and first.shape == (1, 4)

    def test_key_is_the_normalised_text(self, embedder: list[list[str]]) -> None:
        unified.embed_query("  Folk Horror ")
        unified.embed_query("folk horror")
        assert embedder == [["folk horror"]]

    def test_different_queries_are_separate(self, embedder: list[list[str]]) -> None:
        a = unified.embed_query("folk horror")
        b = unified.embed_query("slasher")
        assert len(embedder) == 2 and not np.array_equal(a, b)

    def test_cached_vector_cannot_be_mutated(self, embedder: list[list[str]]) -> None:
        vec = unified.embed_query("folk horror")
        with pytest.raises(ValueError):
            vec[0, 0] = 99.0

    def test_swapping_the_embedder_invalidates_entries(
        self, embedder: list[list[str]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A vector from one embedder must never answer for another."""
        unified.embed_query("folk horror")
        monkeypatch.setattr(
            unified, "_embed_sbert", lambda texts: np.zeros((len(texts), 4), dtype=np.float32)
        )
        assert float(unified.embed_query("folk horror").sum()) == 0.0

    def test_cache_is_bounded(
        self, embedder: list[list[str]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(unified, "_QUERY_CACHE_SIZE", 3)
        for i in range(5):
            unified.embed_query(f"query {i}")
        assert list(unified._QUERY_CACHE) == ["query 2", "query 3", "query 4"]
        unified.embed_query("query 0")  # evicted, so it is computed again
        assert len(embedder) == 6


class TestCallers:
    def test_unified_request_embeds_the_mood_once(self, embedder: list[list[str]]) -> None:
        """Retrieval and scoring used to embed the same mood separately."""
        from app.services.corpus import semantic_search

        films = [
            {"imdb_id": f"tt{i}", "title": f"F{i}", "overview": "p", "keywords": "k"}
            for i in range(3)
        ]
        vecs = np.ones((3, 4), dtype=np.float32)
        pool = semantic_search("folk horror", films, vecs, top_k=3, temperature=0.0)
        unified.compute_signals("folk horror", pool)
        mood_calls = [c for c in embedder if c == ["folk horror"]]
        assert len(mood_calls) == 1
