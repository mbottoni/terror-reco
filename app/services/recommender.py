from __future__ import annotations

from math import log
from typing import Any

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


async def recommend_movies_advanced(
    *,
    mood: str,
    limit: int = 6,
    min_year: int | None = None,
    max_year: int | None = None,
    min_rating: float | None = None,
    kind: str = "movie",  # "movie" | "series" | "both"  (corpus is movies only)
    english_only: bool = False,
    pages: int = 2,
) -> list[dict[str, Any]]:
    """Return horror movies ranked by semantic similarity to *mood*.

    Uses a pre-built corpus of horror movies + sentence-transformer
    embeddings.  The corpus is built once from OMDb on first call and
    cached to disk; every subsequent call is a fast numpy dot-product.

    No hardcoded mood-to-keyword mapping is involved -- matching is
    purely ML-based.
    """
    from .corpus import build_corpus, get_corpus_embeddings, load_corpus, semantic_search

    # Load or build the horror movie corpus (one-time cost)
    corpus = load_corpus()
    if not corpus:
        print("Building horror movie corpus (first run, takes a few minutes)...")
        corpus = await build_corpus(pages=pages)

    # Get pre-computed plot embeddings
    embeddings = get_corpus_embeddings(corpus)

    # Semantic search: rank entire corpus by similarity to the user text.
    # Temperature > 0 adds controlled noise so results vary per request.
    candidates = semantic_search(
        mood, corpus, embeddings, top_k=max(limit * 10, 60), temperature=1.0,
    )

    # Apply optional filters (year range, language)
    filtered: list[dict[str, Any]] = []
    for movie in candidates:
        year_str = movie.get("year") or ""
        try:
            year_int = int(str(year_str)[:4]) if year_str else None
        except Exception:
            year_int = None
        if min_year is not None and (year_int is None or year_int < min_year):
            continue
        if max_year is not None and (year_int is None or year_int > max_year):
            continue
        if min_rating is not None:
            rating_val = movie.get("vote_average")
            if rating_val is None or float(rating_val) < min_rating:
                continue
        if english_only:
            lang = (movie.get("language") or "").lower()
            if "english" not in lang:
                continue

        # Strip internal scoring field
        filtered.append({k: v for k, v in movie.items() if not k.startswith("_")})

    # Weighted random sampling from the top pool so that the final
    # selection varies between requests while remaining relevant.
    pool_size = min(len(filtered), max(limit * 3, 18))
    pool = filtered[:pool_size]
    if len(pool) <= limit:
        return pool

    # Scores decay linearly from 1.0 (best) to 0.3 (end of pool)
    weights = [1.0 - 0.7 * i / (len(pool) - 1) for i in range(len(pool))]
    total = sum(weights)
    probs = [w / total for w in weights]

    # numpy weighted choice without replacement
    import numpy as np

    rng = np.random.default_rng()
    chosen_idx = rng.choice(len(pool), size=limit, replace=False, p=probs)
    chosen_idx.sort()  # preserve rough relevance order
    return [pool[int(i)] for i in chosen_idx]
