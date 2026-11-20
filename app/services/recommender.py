from __future__ import annotations

import re
from math import log
from typing import Any

import numpy as np

# Every strategy here searches the local corpus. The two live-OMDb strategies
# are gone: TF-IDF fitted a vectorizer on 30-120 documents per request, where
# IDF is meaningless, and the keyword one existed only as a no-corpus fallback,
# which has no trigger now that the corpus ships in the repo and the image.


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


_IMDB_ID_RE = re.compile(r"^tt\d{5,10}$")
MAX_SHARED_FILMS = 20


def films_by_ids(ids: list[str]) -> list[dict[str, Any]]:
    """Corpus films for *ids*, in the order given.

    Backs shared result links, which carry the exact films that were on
    screen. Unknown or malformed ids are skipped, duplicates collapse, and
    the list is capped, so a hand-edited URL cannot make this do much.
    """
    from .corpus import get_corpus_and_embeddings

    wanted: list[str] = []
    for raw in ids:
        candidate = raw.strip()
        if _IMDB_ID_RE.match(candidate) and candidate not in wanted:
            wanted.append(candidate)
        if len(wanted) >= MAX_SHARED_FILMS:
            break
    if not wanted:
        return []
    corpus, _ = get_corpus_and_embeddings()
    by_id = {m.get("imdb_id"): m for m in corpus}
    return [
        {k: v for k, v in by_id[i].items() if not k.startswith("_")} for i in wanted if i in by_id
    ]


def explain_match(mood: str, movie: dict[str, Any], max_terms: int = 4) -> list[str]:
    """Which of the film's tags and keywords the query actually hit.

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
    # Tags first: they are the words for what a film is like ("slow burn",
    # "folk horror"); TMDB keywords are mostly what is in it.
    terms = f"{movie.get('tags') or ''},{movie.get('keywords') or ''}"
    for keyword in dict.fromkeys(terms.split(",")):
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


def films_without_each_filter(
    *,
    min_year: int | None,
    max_year: int | None,
    min_rating: float | None,
    english_only: bool,
) -> dict[str, int]:
    """For each active filter, how many films pass once that one is dropped.

    Answers "which filter emptied my results?". Filters run before ranking,
    so a search with none of them passing returns nothing whatever the mood
    was, and the page should point at the filter, not at the mood. The
    ``"all"`` entry is the count with every filter removed.
    """
    from .corpus import get_corpus_and_embeddings

    corpus, _ = get_corpus_and_embeddings()
    active: dict[str, Any] = {
        "min_year": min_year,
        "max_year": max_year,
        "min_rating": min_rating,
        "english_only": english_only or None,
    }
    active = {name: value for name, value in active.items() if value is not None}

    def count(without: str) -> int:
        def keep(name: str) -> Any:
            return None if name == without else active.get(name)

        return sum(
            1
            for movie in corpus
            if _passes_filters(
                movie,
                min_year=keep("min_year"),
                max_year=keep("max_year"),
                min_rating=keep("min_rating"),
                english_only=bool(keep("english_only")),
            )
        )

    counts = {name: count(name) for name in active}
    counts["all"] = len(corpus)
    return counts


def recommend_movies_advanced(
    *,
    mood: str,
    limit: int = 6,
    min_year: int | None = None,
    max_year: int | None = None,
    min_rating: float | None = None,
    english_only: bool = False,
    seed: int | None = None,
    temperature: float = 1.0,
    keep_internal: bool = False,
    demote_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return horror movies ranked by semantic similarity to *mood*.

    Films in *demote_ids* are moved behind everything else before the final
    draw. They are demoted, not removed, so the page is still full when most
    of the good matches have been seen already.

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
    from .corpus import (
        CorpusNotBuiltError,
        get_corpus_and_embeddings,
        resolve_query,
        semantic_search,
    )

    corpus, embeddings = get_corpus_and_embeddings()
    if not corpus:
        # Building inside a request used to be the behaviour here; a crawl that
        # got rate-limited mid-request is what silently froze the corpus at 21
        # films. Fail loudly instead so the problem is visible.
        raise CorpusNotBuiltError(
            "Horror corpus is empty. Build it first:  make corpus  "
            "(or: python scripts/build_corpus.py --target 500)"
        )

    # Against the whole corpus, before filtering: "like Alien, after 2010"
    # still means Alien, though the filter removes it from the candidates.
    query_vector, exclude_ids = resolve_query(mood, corpus, embeddings)

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
        # 60 leaves room for the franchise rule and for demoted films. It
        # must not grow with ``limit``: the unified strategy asks for a pool of
        # 60, and ten times that made the pool the whole corpus, which widens
        # the score spread that semantic_search() scales its noise by.
        top_k=max(limit, 60),
        temperature=temperature,
        seed=seed,
        query_vector=query_vector,
        exclude_ids=exclude_ids,
    )
    # Nothing but named films passed the filters: those were masked, not removed.
    candidates = [m for m in candidates if str(m.get("imdb_id") or "") not in exclude_ids]

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

    # One film per franchise in what the user sees. Skipped when the caller
    # wants the raw pool (keep_internal): the unified re-ranker de-duplicates
    # after blending, so it can pick which entry of a series to show.
    if not keep_internal:
        from .unified_recommender import first_of_each_collection

        filtered = [filtered[i] for i in first_of_each_collection(filtered)]

    if demote_ids and not keep_internal:
        # Stable partition: fresh films keep their order, then the demoted
        # ones in theirs.
        filtered = [m for m in filtered if m.get("imdb_id") not in demote_ids] + [
            m for m in filtered if m.get("imdb_id") in demote_ids
        ]

    # The best-ranked films, in order. Variety between visits comes from the
    # score noise in semantic_search() and from demoting what was just shown.
    #
    # This used to draw `limit` films at random from the top 3 x limit, with
    # weights falling only to 0.3. Measured on the graded benchmark as actually
    # served, that cost a third of the page's quality (NDCG@6 0.43 against
    # 0.62 without the draw) to show 24 different films over ten visits where
    # 13 is plenty. It also ran on the unified strategy's candidate pool,
    # handing the re-ranker a random 60 of the top 180 films.
    return filtered[:limit]
