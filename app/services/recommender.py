from __future__ import annotations

from math import log
from typing import Any

import numpy as np

from .strategies.base import RecommenderStrategy
from .strategies.keyword_omdb import KeywordOMDbStrategy


def get_strategy(name: str) -> RecommenderStrategy:
    """Live-OMDb fallback strategies.

    The TF-IDF strategy was removed: it fitted a fresh vectorizer on 30-120
    documents per request, where IDF is statistically meaningless, made live
    OMDb calls on the request path, and scored below the corpus pipeline on
    every metric.  Only the keyword strategy remains as a no-corpus fallback.
    """
    del name  # only one live strategy remains
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


def similar_movies(*, imdb_id: str, limit: int = 6) -> list[dict[str, Any]]:
    """Films most like *imdb_id*, by cosine on the cached corpus embeddings.

    Item-to-item similarity needs no query encoding at all -- the seed film's
    vector is already in the cached matrix -- so this is a single dot product
    against 500 rows.

    Synchronous on purpose: there is no I/O here, only numpy.  Callers on the
    request path must hand it to a threadpool rather than run it on the event
    loop -- see :func:`recommend_movies_advanced`.
    """
    from .corpus import get_corpus_and_embeddings

    corpus, embeddings = get_corpus_and_embeddings()
    if not corpus or not embeddings.size:
        return []

    seed_idx = next((i for i, m in enumerate(corpus) if m.get("imdb_id") == imdb_id), None)
    if seed_idx is None:
        return []

    sims = (embeddings[seed_idx : seed_idx + 1] @ embeddings.T).ravel()
    sims[seed_idx] = -np.inf  # never recommend the film itself
    top = np.argsort(-sims)[:limit]
    return [
        {k: v for k, v in corpus[int(i)].items() if not k.startswith("_")}
        for i in top
        if np.isfinite(sims[i])
    ]


def explain_match(mood: str, movie: dict[str, Any], max_terms: int = 4) -> list[str]:
    """Which of the film's keywords the query actually hit.

    The tone/subgenre vocabulary in `keywords` is what makes mood queries work
    at all, so showing the overlap explains a recommendation in the user's own
    words rather than presenting it as magic.
    """
    from .unified_recommender import _tokenize

    def fold(tokens: list[str]) -> set[str]:
        # Crude plural folding so "rituals" matches the keyword "ritual".
        # Deliberately NOT applied inside _tokenize: that feeds BM25, and
        # changing it would shift ranking and invalidate the recorded
        # evaluation numbers. This is display-only.
        return {t[:-1] if len(t) > 3 and t.endswith("s") else t for t in tokens}

    query_terms = fold(_tokenize(mood))
    if not query_terms:
        return []

    matched: list[str] = []
    for keyword in (movie.get("keywords") or "").split(","):
        keyword = keyword.strip()
        if not keyword:
            continue
        if query_terms & fold(_tokenize(keyword)):
            matched.append(keyword)
        if len(matched) >= max_terms:
            break
    return matched


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


def recommend_movies_advanced(
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
    keep_internal: bool = False,
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

    **This function is synchronous and CPU-bound**: a sentence-transformer
    forward pass on the query plus a dense matmul over the corpus.  It used to
    be declared ``async`` despite containing no ``await``, which meant it ran
    directly on the event loop and serialised every concurrent request behind
    it.  Request handlers must call it through ``run_in_threadpool``; torch and
    numpy both release the GIL, so the work really does overlap.
    """
    from .corpus import CorpusNotBuiltError, get_corpus_and_embeddings, semantic_search

    corpus, embeddings = get_corpus_and_embeddings()
    if not corpus:
        # Building inside a request used to be the behaviour here; a crawl that
        # got rate-limited mid-request is what silently froze the corpus at 21
        # films. Fail loudly instead so the problem is visible.
        raise CorpusNotBuiltError(
            "Horror corpus is empty. Build it first:  make corpus  "
            "(or: python scripts/build_corpus.py --target 500)"
        )

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
    filtered_corpus = len(keep) < len(corpus)
    if filtered_corpus:
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

    if filtered_corpus:
        # `_embedding_row` indexes the *sliced* matrix; translate it back to the
        # full corpus so downstream lookups are not silently off-by-filter.
        for movie in candidates:
            movie["_embedding_row"] = keep[movie["_embedding_row"]]

    # Strip internal scoring fields unless a caller needs them (the unified
    # re-ranker uses `_embedding_row` to avoid re-encoding cached vectors).
    filtered = (
        list(candidates)
        if keep_internal
        else [{k: v for k, v in m.items() if not k.startswith("_")} for m in candidates]
    )

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
