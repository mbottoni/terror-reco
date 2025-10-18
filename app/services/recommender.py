from __future__ import annotations

from math import log
from typing import Any

from .omdb_client import get_omdb_client
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
    kind: str = "movie",  # "movie" | "series" | "both"
    english_only: bool = False,
    pages: int = 3,
) -> list[dict[str, Any]]:
    client = await get_omdb_client()

    mood_norm = (mood or "").strip().lower()
    queries = [
        f"{mood_norm} horror",
        "horror",
        "scary horror",
        "supernatural horror",
        "slasher horror",
        "zombie horror",
    ]

    types: list[str]
    if kind == "both":
        types = ["movie", "series"]
    else:
        types = [kind]

    # Collect IDs across queries/types/pages
    ids: list[str] = []
    for q in queries:
        for t in types:
            for page in range(1, max(1, pages) + 1):
                res = await client.search_titles(q, page=page, type_=t)
                for item in res or []:
                    imdb_id = item.get("imdbID")
                    if isinstance(imdb_id, str):
                        ids.append(imdb_id)
    ids = list(dict.fromkeys(ids))

    # Fetch details, filter to horror, year range, language
    details: list[dict[str, Any]] = []
    for imdb_id in ids:
        d = await client.get_by_id(imdb_id, plot_full=True)
        if not d:
            continue
        genre = (d.get("Genre") or "").lower()
        if "horror" not in genre:
            continue
        year_str = d.get("Year") or ""
        try:
            year_int = int(str(year_str)[:4]) if year_str else None
        except Exception:
            year_int = None
        if min_year is not None and (year_int is None or year_int < min_year):
            continue
        if max_year is not None and (year_int is None or year_int > max_year):
            continue
        if english_only:
            lang = (d.get("Language") or "").lower()
            if "english" not in lang:
                continue
        poster = d.get("Poster")
        poster_url = poster if poster and poster != "N/A" else None
        details.append(
            {
                "title": d.get("Title"),
                "overview": d.get("Plot") or "",
                "poster_url": poster_url,
                "release_date": d.get("Released"),
                "vote_average": (
                    float(d.get("imdbRating") or 0)
                    if (d.get("imdbRating") and d.get("imdbRating") != "N/A")
                    else None
                ),
                "_score": _score_popularity(d),
            }
        )
        if len(details) >= max(limit * 8, 60):
            break

    if not details:
        return []

    ranked = sorted(details, key=lambda x: x.get("_score", 0.0), reverse=True)
    pool = ranked[: max(10, limit * 3)]
    if len(pool) <= limit:
        return [{k: v for k, v in m.items() if k != "_score"} for m in pool[:limit]]
    import random

    chosen = random.sample(pool, k=limit)
    return [{k: v for k, v in m.items() if k != "_score"} for m in chosen]
