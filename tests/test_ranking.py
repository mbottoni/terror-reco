"""Tests for the ranking signals: BM25, MMR diversity, and determinism.

These assert on *properties* rather than exact orderings -- the pipeline is
deliberately stochastic, so any test pinning a specific result would be
flaky by construction. The one exception is the determinism test, which is
precisely about making a fixed seed reproducible.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.services.corpus import embedding_text
from app.services.recommender import _passes_filters
from app.services.unified_recommender import _bm25_scores, _mmr, recommend_unified_semantic


def _movie(title: str, overview: str = "", **extra: Any) -> dict[str, Any]:
    return {
        "title": title,
        "overview": overview,
        "imdb_id": f"tt{abs(hash(title)) % 10**7}",
        **extra,
    }


class TestBM25:
    def test_stopwords_do_not_drive_the_score(self) -> None:
        """The old overlap ratio scored on 'a'/'in'/'the'; BM25 must not."""
        items = [
            _movie("Cabin", overview="a slow burn in the woods with a cabin"),
            _movie("Unrelated", overview="a of the in on at to for with and or but"),
        ]
        scores = _bm25_scores("a slow burn in the cabin", items)
        assert scores[0] > scores[1]
        assert scores[1] == 0.0

    def test_rare_terms_outweigh_common_ones(self) -> None:
        """IDF: a term in every document carries no signal."""
        items = [
            _movie("A", overview="horror lovecraftian"),
            _movie("B", overview="horror"),
            _movie("C", overview="horror"),
            _movie("D", overview="horror"),
        ]
        scores = _bm25_scores("horror lovecraftian", items)
        assert scores[0] > scores[1]

    def test_keywords_field_contributes(self) -> None:
        """Tone vocabulary lives in `keywords`, not the plot."""
        with_kw = _movie("X", overview="a man visits an island", keywords="folk horror, pagan")
        without = _movie("Y", overview="a man visits an island")
        scores = _bm25_scores("folk horror pagan ritual", [with_kw, without])
        assert scores[0] > scores[1]

    def test_empty_query_and_empty_items_are_safe(self) -> None:
        assert len(_bm25_scores("", [_movie("A")])) == 1
        assert len(_bm25_scores("anything", [])) == 0


class TestMMR:
    def test_embedding_path_prefers_dissimilar_items(self) -> None:
        items = [_movie(c) for c in "ABC"]
        # A and B identical in embedding space; C orthogonal.
        embs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        sims = np.array([1.0, 0.99, 0.5], dtype=np.float32)
        picked = [m["title"] for m in _mmr(items, sims, k=2, lambda_=0.5, embeddings=embs)]
        assert picked[0] == "A"
        # B is nearly as relevant but redundant with A, so C wins the 2nd slot.
        assert picked[1] == "C"

    def test_lambda_one_ignores_diversity(self) -> None:
        items = [_movie(c) for c in "ABC"]
        embs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        sims = np.array([1.0, 0.99, 0.5], dtype=np.float32)
        picked = [m["title"] for m in _mmr(items, sims, k=2, lambda_=1.0, embeddings=embs)]
        assert picked == ["A", "B"]

    def test_falls_back_to_lexical_without_embeddings(self) -> None:
        items = [
            _movie("A", overview="ghost haunted mansion"),
            _movie("B", overview="ghost haunted mansion"),
            _movie("C", overview="alien spaceship orbit"),
        ]
        sims = np.array([1.0, 0.99, 0.5], dtype=np.float32)
        picked = [m["title"] for m in _mmr(items, sims, k=2, lambda_=0.5)]
        assert picked == ["A", "C"]

    def test_returns_everything_when_pool_smaller_than_k(self) -> None:
        items = [_movie("A"), _movie("B")]
        sims = np.array([1.0, 0.5], dtype=np.float32)
        assert len(_mmr(items, sims, k=6, lambda_=0.7)) == 2


class TestDeterminism:
    def _items(self) -> list[dict[str, Any]]:
        return [
            _movie(f"Film {i}", overview=f"a horror story number {i}", vote_average=5 + i % 5)
            for i in range(20)
        ]

    def test_same_seed_and_zero_temperature_reproduce(self) -> None:
        items = self._items()
        a = recommend_unified_semantic(mood="scary", items=items, limit=5, seed=1, temperature=0.0)
        b = recommend_unified_semantic(mood="scary", items=items, limit=5, seed=1, temperature=0.0)
        assert [m["title"] for m in a] == [m["title"] for m in b]

    def test_zero_temperature_is_stable_across_seeds(self) -> None:
        """temperature=0 means no randomness, so the seed must not matter."""
        items = self._items()
        a = recommend_unified_semantic(mood="scary", items=items, limit=5, seed=1, temperature=0.0)
        b = recommend_unified_semantic(mood="scary", items=items, limit=5, seed=99, temperature=0.0)
        assert [m["title"] for m in a] == [m["title"] for m in b]

    def test_empty_items_returns_empty(self) -> None:
        assert recommend_unified_semantic(mood="scary", items=[], limit=5) == []


class TestPrefilter:
    def test_rating_filter(self) -> None:
        assert _passes_filters(
            _movie("A", vote_average=8.0),
            min_year=None,
            max_year=None,
            min_rating=7.5,
            english_only=False,
        )
        assert not _passes_filters(
            _movie("B", vote_average=6.0),
            min_year=None,
            max_year=None,
            min_rating=7.5,
            english_only=False,
        )

    def test_missing_values_are_excluded_not_crashed(self) -> None:
        sparse = _movie("NoData")
        for kwargs in (
            {"min_year": 1980, "max_year": None, "min_rating": None, "english_only": False},
            {"min_year": None, "max_year": None, "min_rating": 5.0, "english_only": False},
            {"min_year": None, "max_year": None, "min_rating": None, "english_only": True},
        ):
            assert not _passes_filters(sparse, **kwargs)  # type: ignore[arg-type]

    def test_year_range_and_language(self) -> None:
        movie = _movie("A", year="1985", language="English, Italian")
        base = {"min_year": 1980, "max_year": 1990, "min_rating": None, "english_only": True}
        assert _passes_filters(movie, **base)  # type: ignore[arg-type]
        assert not _passes_filters(movie, **{**base, "min_year": 1990})  # type: ignore[arg-type]


class TestEmbeddingText:
    def test_composes_keywords_and_genre_not_just_plot(self) -> None:
        """The plot alone lacks tone vocabulary; that was the retrieval gap."""
        text = embedding_text(
            {
                "title": "The Wicker Man",
                "genre": "Horror, Mystery",
                "keywords": "folk horror, pagan ritual",
                "overview": "A policeman visits an island.",
            }
        )
        for fragment in ("The Wicker Man", "Mystery", "folk horror", "policeman"):
            assert fragment in text

    def test_missing_fields_are_skipped_cleanly(self) -> None:
        text = embedding_text({"title": "Solo", "overview": "", "genre": None})
        assert text == "Solo"
