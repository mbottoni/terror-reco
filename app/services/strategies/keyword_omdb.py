from __future__ import annotations

import random
from math import log
from typing import Any

from ..omdb_client import get_omdb_client

# No hardcoded mood-to-keyword mapping.
# The main recommendation pipeline (recommend_movies_advanced) uses
# sentence-transformer embeddings against a pre-built corpus.
# This strategy's _expand_queries is only used for the lightweight
# KeywordOMDbStrategy (live OMDb search fallback).

_STOP_WORDS = frozenset(
    {
        "a", "an", "and", "the", "in", "of", "with", "for", "on", "to",
        "lots", "very", "that", "this", "but", "not", "its", "my", "me",
    }
)


def _normalize(text: str) -> str:
    return text.strip().lower()


def _expand_queries(mood: str) -> list[str]:
    """Generate OMDb title-search queries from a mood description.

    Simply splits the mood text into content words and adjacent pairs.
    No hardcoded keyword mapping -- the heavy lifting is done by the
    corpus-based semantic search in :mod:`app.services.corpus`.
    """
    words = _normalize(mood).split()
    content = [w for w in words if w not in _STOP_WORDS and len(w) >= 3]

    queries: list[str] = []

    # Individual content words (each may match a movie title)
    for w in content:
        queries.append(w)

    # Adjacent pairs (e.g. "found footage", "body horror")
    for i in range(len(content) - 1):
        queries.append(f"{content[i]} {content[i + 1]}")

    # Minimal fallback
    if not queries:
        queries = ["horror", "thriller", "supernatural"]

    # De-duplicate preserving order
    seen: set[str] = set()
    return [q for q in queries if q not in seen and not seen.add(q)]  # type: ignore[func-returns-value]


def _na(val: Any) -> str | None:
    """Return None for OMDb 'N/A' sentinel values."""
    if val is None or val == "N/A":
        return None
    return str(val)


def _build_movie(d: dict[str, Any], score: float | None = None) -> dict[str, Any]:
    """Build a normalised movie dict from a raw OMDb detail response."""
    poster = d.get("Poster")
    poster_url = poster if poster and poster != "N/A" else None
    rating_str = d.get("imdbRating") or ""
    movie: dict[str, Any] = {
        "imdb_id": d.get("imdbID"),
        "title": d.get("Title"),
        "overview": d.get("Plot") or "",
        "poster_url": poster_url,
        "release_date": _na(d.get("Released")),
        "year": d.get("Year"),
        "vote_average": (
            float(rating_str) if rating_str and rating_str != "N/A" else None
        ),
        "genre": d.get("Genre"),
        "director": _na(d.get("Director")),
        "actors": _na(d.get("Actors")),
        "writer": _na(d.get("Writer")),
        "runtime": _na(d.get("Runtime")),
        "language": _na(d.get("Language")),
        "country": _na(d.get("Country")),
        "rated": _na(d.get("Rated")),
        "awards": _na(d.get("Awards")),
    }
    if score is not None:
        movie["_score"] = score
    return movie


def _score_omdb(detail: dict[str, Any]) -> float:
    rating_str = (detail.get("imdbRating") or "0").replace("N/A", "0")
    votes_str = (detail.get("imdbVotes") or "0").replace(",", "")
    try:
        rating = float(rating_str)
    except ValueError:
        rating = 0.0
    try:
        votes = int(votes_str)
    except ValueError:
        votes = 0
    return rating * (1 + log(1 + votes))


class KeywordOMDbStrategy:
    async def recommend(self, mood: str, limit: int = 5) -> list[dict[str, Any]]:
        client = await get_omdb_client()

        ids: list[str] = []
        for q in _expand_queries(mood):
            if len(ids) >= 150:
                break
            res = await client.search_titles(q, page=1)
            for item in res or []:
                imdb_id = item.get("imdbID")
                if isinstance(imdb_id, str):
                    ids.append(imdb_id)
            # also try page 2 for generic queries
            if q != f"{_normalize(mood)} horror" and len(ids) < 150:
                res2 = await client.search_titles(q, page=2)
                for item in res2 or []:
                    imdb_id = item.get("imdbID")
                    if isinstance(imdb_id, str):
                        ids.append(imdb_id)

        ids = list(dict.fromkeys(ids))[:150]

        details: list[dict[str, Any]] = []
        for imdb_id in ids:
            d = await client.get_by_id(imdb_id)
            if not d:
                continue
            genre = (d.get("Genre") or "").lower()
            if "horror" not in genre:
                continue
            details.append(_build_movie(d, score=_score_omdb(d)))
            if len(details) >= max(limit * 6, 30):
                break

        # If still no details, do a final generic horror fetch
        if not details:
            res = await client.search_titles("horror", page=1)
            for item in res or []:
                imdb_id = item.get("imdbID")
                if not isinstance(imdb_id, str):
                    continue
                d = await client.get_by_id(imdb_id)
                if not d:
                    continue
                genre = (d.get("Genre") or "").lower()
                if "horror" not in genre:
                    continue
                details.append(_build_movie(d, score=_score_omdb(d)))
                if len(details) >= limit:
                    break

        if not details:
            return []

        details_sorted = sorted(details, key=lambda x: x.get("_score", 0.0), reverse=True)
        # Add a bit of variety: sample from the top pool
        pool = details_sorted[: max(10, limit * 3)]
        if len(pool) <= limit:
            return [{k: v for k, v in m.items() if k != "_score"} for m in pool[:limit]]
        chosen = random.sample(pool, k=limit)
        return [{k: v for k, v in m.items() if k != "_score"} for m in chosen]
