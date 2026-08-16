from __future__ import annotations

from math import log
from typing import Any

import numpy as np

from .strategies.base import RecommenderStrategy
from .strategies.embedding_omdb import EmbeddingOMDbStrategy
from .strategies.keyword_omdb import KeywordOMDbStrategy


def get_strategy(name: str) -> RecommenderStrategy:
    key = (name or "").strip().lower()
    if key in ("embed", "embedding", "tfidf"):
        return EmbeddingOMDbStrategy()
    # default
    return KeywordOMDbStrategy()


async def recommend_movies(
    mood: str, limit: int = 5, strategy: str = "keyword"
) -> list[dict[str, Any]]:
    impl = get_strategy(strategy)
    return await impl.recommend(mood=mood, limit=limit)


def _score_popularity(detail: dict[str, Any]) -> float:
    rating = float(detail.get("vote_average") or 0.0)
    votes_str = (detail.get("imdbVotes") or detail.get("imdb_votes_raw") or "0").replace(",", "")
    metascore_str = detail.get("Metascore") or detail.get("metascore_raw") or "0"
    try:
        votes = int(votes_str)
    except Exception:
        votes = 0
    try:
        metascore = int(metascore_str)
    except Exception:
        metascore = 0
    return rating * (1 + log(1 + votes)) + 0.02 * metascore


def _movie_year(movie: dict[str, Any]) -> int | None:
    year_str = movie.get("year") or ""
    try:
        return int(str(year_str)[:4]) if year_str else None
    except (TypeError, ValueError):
        return None


def _passes_filters(
    movie: dict[str, Any],
    *,
    min_year: int | None,
    max_year: int | None,
    min_rating: float | None,
    english_only: bool,
) -> bool:
    year = _movie_year(movie)
    if min_year is not None and (year is None or year < min_year):
        return False
    if max_year is not None and (year is None or year > max_year):
        return False
    if min_rating is not None:
        rating = movie.get("vote_average")
        if rating is None or float(rating) < min_rating:
            return False
    if english_only and "english" not in (movie.get("language") or "").lower():
        return False
    return True


async def recommend_movies_advanced(
    *,
    mood: str,
    limit: int = 6,
    min_year: int | None = None,
    max_year: int | None = None,
    min_rating: float | None = None,
    kind: str = "movie",  # "movie" | "series" | "both"  (corpus is movies only)
    english_only: bool = False,
    seed: int | None = None,
    temperature: float = 1.0,
) -> list[dict[str, Any]]:
    """Return horror movies ranked by semantic similarity to *mood*.

    Uses a pre-built corpus of horror movies + sentence-transformer
    embeddings.  The corpus is built offline by ``scripts/build_corpus.py``;
    every request is a fast numpy dot-product with no network I/O.

    Filters are applied to the corpus *before* ranking, not to the top-K
    afterwards.  Filtering after retrieval meant a restrictive filter
    (e.g. ``min_rating=8``) could leave only a handful of survivors out of
    the top-K, so the user got weak matches that merely happened to pass.

    Passing ``seed`` with ``temperature=0`` makes the whole call
    reproducible, which offline evaluation depends on.
    """
    from .corpus import CorpusNotBuiltError, get_corpus_embeddings, load_corpus, semantic_search

    corpus = load_corpus()
    if not corpus:
        # Building inside a request used to be the behaviour here; a crawl that
        # got rate-limited mid-request is what silently froze the corpus at 21
        # films. Fail loudly instead so the problem is visible.
        raise CorpusNotBuiltError(
            "Horror corpus is empty. Build it first:  make corpus  "
            "(or: python scripts/build_corpus.py --target 500)"
        )

    embeddings = get_corpus_embeddings(corpus)

    # Prefilter, keeping corpus rows and embedding rows aligned.
    keep = [
        i
        for i, movie in enumerate(corpus)
        if _passes_filters(
            movie,
            min_year=min_year,
            max_year=max_year,
            min_rating=min_rating,
            english_only=english_only,
        )
    ]
    if not keep:
        return []
    if len(keep) < len(corpus):
        corpus = [corpus[i] for i in keep]
        embeddings = embeddings[keep]

    candidates = semantic_search(
        mood,
        corpus,
        embeddings,
        top_k=max(limit * 10, 60),
        temperature=temperature,
        seed=seed,
    )

    # Strip internal scoring fields
    filtered = [{k: v for k, v in movie.items() if not k.startswith("_")} for movie in candidates]

    # Weighted random sampling from the top pool so that the final
    # selection varies between requests while remaining relevant.
    pool_size = min(len(filtered), max(limit * 3, 18))
    pool = filtered[:pool_size]
    if len(pool) <= limit:
        return pool

    # temperature=0 means "no randomness anywhere", so skip sampling too --
    # otherwise a deterministic search would still be shuffled at this stage.
    if temperature <= 0:
        return pool[:limit]

    # Scores decay linearly from 1.0 (best) to 0.3 (end of pool)
    weights = [1.0 - 0.7 * i / (len(pool) - 1) for i in range(len(pool))]
    total = sum(weights)
    probs = [w / total for w in weights]

    rng = np.random.default_rng(seed)
    chosen_idx = rng.choice(len(pool), size=limit, replace=False, p=probs)
    chosen_idx.sort()  # preserve rough relevance order
    return [pool[int(i)] for i in chosen_idx]
