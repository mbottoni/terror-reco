"""Tests for the ranking signals: BM25, MMR diversity, and determinism.

These assert on *properties* rather than exact orderings -- the pipeline is
deliberately stochastic, so any test pinning a specific result would be
flaky by construction. The one exception is the determinism test, which is
precisely about making a fixed seed reproducible.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from app.services.corpus import embedding_text
from app.services.recommender import _passes_filters
from app.services.unified_recommender import (
    Bm25Stats,
    _bm25_scores,
    _bm25_stats,
    _mmr,
    corpus_bm25_stats,
    recommend_unified_semantic,
)


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


class TestBM25CollectionStatistics:
    """IDF must come from the corpus, not from the pool a query retrieved."""

    def test_pool_idf_collapses_for_a_term_every_candidate_shares(self) -> None:
        """The bug: in a pool retrieved *for* "vampire", every film says vampire."""
        pool = [_movie(f"V{i}", overview="vampire castle") for i in range(4)]
        pool.append(_movie("Odd", overview="vampire castle zeppelin"))
        corpus_stats = Bm25Stats(n_docs=500, avg_len=3.0, df={"vampire": 20, "zeppelin": 1})

        pool_only = _bm25_scores("vampire", pool)
        with_corpus = _bm25_scores("vampire", pool, stats=corpus_stats)
        # Rare in the corpus (20/500), universal in the pool (5/5): the pool
        # estimate is a small fraction of the true weight.
        assert with_corpus[0] > pool_only[0] * 5

    def test_score_does_not_depend_on_the_other_candidates(self) -> None:
        film = _movie("Target", overview="pagan ritual island")
        stats = Bm25Stats(n_docs=500, avg_len=3.0, df={"pagan": 5, "ritual": 12})
        pool_a = [film, _movie("A", overview="pagan pagan"), _movie("B", overview="ritual")]
        pool_b = [film, _movie("C", overview="shark"), _movie("D", overview="clown")]

        assert _bm25_scores("pagan ritual", pool_a, stats=stats)[0] == pytest.approx(
            _bm25_scores("pagan ritual", pool_b, stats=stats)[0]
        )
        # ...which is exactly what pool-level statistics get wrong.
        assert _bm25_scores("pagan ritual", pool_a)[0] != pytest.approx(
            _bm25_scores("pagan ritual", pool_b)[0]
        )

    def test_term_unknown_to_the_collection_still_scores_on_a_match(self) -> None:
        stats = Bm25Stats(n_docs=500, avg_len=2.0, df={})
        scores = _bm25_scores(
            "zeppelin", [_movie("A", overview="zeppelin"), _movie("B")], stats=stats
        )
        assert scores[0] > 0 and scores[1] == 0

    def test_stats_count_each_document_once_per_term(self) -> None:
        stats = _bm25_stats([_movie("A", overview="ghost ghost ghost"), _movie("B", overview="x")])
        assert stats.n_docs == 2
        assert stats.df["ghost"] == 1

    def test_shipped_corpus_yields_statistics(self) -> None:
        stats = corpus_bm25_stats()
        if stats is None:
            pytest.skip("corpus not built in this environment")
        assert stats.n_docs >= 100
        # "horror" is in nearly every record's genre; a tone word is not.
        assert stats.df.get("horror", 0) > stats.df.get("lovecraftian", 0)


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


class TestFranchiseDeduplication:
    """No more than one film per TMDB collection on a results page."""

    def test_keeps_the_best_ranked_entry_of_each_collection(self) -> None:
        from app.services.unified_recommender import first_of_each_collection

        items = [
            {"title": "Saw", "collection_id": 656},
            {"title": "Standalone A", "collection_id": None},
            {"title": "Saw II", "collection_id": 656},
            {"title": "Scream", "collection_id": 2602},
            {"title": "Standalone B"},
            {"title": "Scream 2", "collection_id": 2602},
            {"title": "Saw III", "collection_id": 656},
        ]
        kept = [items[i]["title"] for i in first_of_each_collection(items)]
        assert kept == ["Saw", "Standalone A", "Scream", "Standalone B"]

    def test_films_outside_any_collection_are_never_dropped(self) -> None:
        from app.services.unified_recommender import first_of_each_collection

        items = [{"title": f"F{i}", "collection_id": None} for i in range(5)]
        assert first_of_each_collection(items) == [0, 1, 2, 3, 4]
        assert first_of_each_collection([]) == []

    def test_shipped_corpus_carries_collections(self) -> None:
        from app.services.corpus import load_corpus

        corpus = load_corpus()
        if not corpus:
            pytest.skip("corpus not built in this environment")
        assert all("collection_id" in m for m in corpus), "run scripts/add_collections.py"
        in_collection = sum(1 for m in corpus if m.get("collection_id"))
        assert in_collection > len(corpus) * 0.2

    def test_unified_results_never_repeat_a_franchise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nine near-identical sequels and three other films: at most one sequel."""
        import app.services.unified_recommender as unified

        sequels = [
            _movie(f"Slash {i}", overview="masked killer stalks teens", collection_id=7)
            for i in range(9)
        ]
        others = [_movie(f"Other {i}", overview="masked killer stalks teens") for i in range(3)]
        items = sequels + others
        rng = np.random.default_rng(0)

        def _fake(texts: list[str]) -> np.ndarray:
            vecs = rng.normal(size=(len(texts), 8)).astype(np.float32)
            return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

        monkeypatch.setattr(unified, "_embed_sbert", _fake)
        monkeypatch.setattr(unified, "_QUERY_CACHE", unified.OrderedDict())
        for seed in range(5):
            picked = recommend_unified_semantic(
                mood="masked killer", items=[dict(m) for m in items], limit=4, seed=seed
            )
            from_series = [m for m in picked if m.get("collection_id") == 7]
            assert len(from_series) <= 1
            assert len(picked) == 4

    def test_semantic_results_never_repeat_a_franchise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import app.services.corpus as corpus_mod
        import app.services.unified_recommender as unified
        from app.services.recommender import recommend_movies_advanced

        corpus = [
            {"imdb_id": f"tt{i}", "title": f"Slash {i}", "overview": "p", "collection_id": 7}
            for i in range(10)
        ] + [{"imdb_id": f"tt9{i}", "title": f"Other {i}", "overview": "p"} for i in range(10)]
        vecs = np.ones((len(corpus), 4), dtype=np.float32)
        monkeypatch.setattr(corpus_mod, "get_corpus_and_embeddings", lambda: (corpus, vecs))
        monkeypatch.setattr(
            unified, "_embed_sbert", lambda texts: np.ones((len(texts), 4), dtype=np.float32)
        )
        monkeypatch.setattr(unified, "_QUERY_CACHE", unified.OrderedDict())
        for seed in range(5):
            picked = recommend_movies_advanced(mood="x", limit=6, seed=seed)
            assert sum(1 for m in picked if m.get("collection_id") == 7) <= 1
            assert len(picked) == 6

    def test_raw_pool_for_the_unified_ranker_is_not_deduplicated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import app.services.corpus as corpus_mod
        import app.services.unified_recommender as unified
        from app.services.recommender import recommend_movies_advanced

        corpus = [
            {"imdb_id": f"tt{i}", "title": f"Slash {i}", "overview": "p", "collection_id": 7}
            for i in range(10)
        ]
        vecs = np.ones((len(corpus), 4), dtype=np.float32)
        monkeypatch.setattr(corpus_mod, "get_corpus_and_embeddings", lambda: (corpus, vecs))
        monkeypatch.setattr(
            unified, "_embed_sbert", lambda texts: np.ones((len(texts), 4), dtype=np.float32)
        )
        monkeypatch.setattr(unified, "_QUERY_CACHE", unified.OrderedDict())
        pool = recommend_movies_advanced(mood="x", limit=60, keep_internal=True, temperature=0)
        assert len(pool) == 10
