"""Async TMDB API client used for corpus discovery.

TMDB is used instead of OMDb for *discovery* because OMDb's ``s=`` parameter
searches titles only -- there is no way to browse by genre, which is why the
previous corpus was dominated by films with the word "horror" in the title.
TMDB's ``/discover/movie`` supports real genre filtering.

OMDb is still used afterwards to enrich each film with IMDb rating/votes,
Metascore and awards (see :mod:`app.services.omdb_client`).
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

import httpx

from ..settings import get_settings

# TMDB genre IDs (verified against /genre/movie/list).
GENRE_HORROR = 27
GENRE_THRILLER = 53
GENRE_MYSTERY = 9648

# TMDB allows ~40 req/s. We throttle well below that to stay a good citizen.
_DEFAULT_RATE_LIMIT = 10.0
_MAX_RETRIES = 5


class TMDBError(RuntimeError):
    """Raised when TMDB cannot be reached or is misconfigured."""


class TMDBClient:
    """Rate-limited, retrying TMDB client.

    Unlike the old corpus builder, this client never gives up after a fixed
    number of consecutive errors -- transient failures are retried with
    exponential backoff so a rate-limit blip cannot silently truncate a build.
    """

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        *,
        rate_limit: float = _DEFAULT_RATE_LIMIT,
    ) -> None:
        settings = get_settings()
        self._base_url = settings.TMDB_BASE_URL.rstrip("/")
        self._bearer = settings.TMDB_BEARER_TOKEN
        self._api_key = settings.TMDB_API_KEY
        if not self._bearer and not self._api_key:
            raise TMDBError(
                "No TMDB credentials configured. Set TMDB_BEARER_TOKEN "
                "(preferred) or TMDB_API_KEY in your environment / .env file."
            )
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=5.0))
        self._min_interval = 1.0 / rate_limit if rate_limit > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next_slot = 0.0

    # -- internals ---------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        headers = {"accept": "application/json"}
        if self._bearer:
            headers["Authorization"] = f"Bearer {self._bearer}"
        return headers

    async def _throttle(self) -> None:
        """Space requests out to respect the configured rate limit."""
        if self._min_interval <= 0:
            return
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            wait = self._next_slot - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = loop.time()
            self._next_slot = now + self._min_interval

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        merged: dict[str, Any] = dict(params or {})
        # v3 api_key is only needed when no bearer token is configured.
        if not self._bearer and self._api_key:
            merged["api_key"] = self._api_key

        url = f"{self._base_url}/{path.lstrip('/')}"
        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            await self._throttle()
            try:
                resp = await self._client.get(url, params=merged, headers=self._headers())
            except httpx.HTTPError as exc:  # network-level failure
                last_exc = exc
                await self._backoff(attempt)
                continue

            if resp.status_code == 429:
                # Honour Retry-After when TMDB provides it.
                retry_after = resp.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else None
                await self._backoff(attempt, override=delay)
                continue

            if resp.status_code == 404:
                return {}

            if resp.status_code >= 500:
                last_exc = httpx.HTTPStatusError(
                    f"TMDB {resp.status_code}", request=resp.request, response=resp
                )
                await self._backoff(attempt)
                continue

            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            return data

        raise TMDBError(f"TMDB request failed after {_MAX_RETRIES} attempts: {path}") from last_exc

    @staticmethod
    async def _backoff(attempt: int, *, override: float | None = None) -> None:
        delay = override if override is not None else (2.0**attempt) + random.random()
        await asyncio.sleep(min(delay, 60.0))

    # -- public API --------------------------------------------------------
    async def discover_movies(
        self,
        *,
        genre: int = GENRE_HORROR,
        page: int = 1,
        sort_by: str = "vote_count.desc",
        min_votes: int = 50,
        release_date_gte: str | None = None,
        release_date_lte: str | None = None,
    ) -> list[dict[str, Any]]:
        """Browse movies by genre.

        Note ``sort_by`` defaults to ``vote_count.desc`` rather than
        ``popularity.desc``: TMDB's popularity is a *live trending* metric, so
        sorting by it produces a corpus that changes week to week and skews
        heavily modern. Vote count is stable and approximates the canon.
        """
        params: dict[str, Any] = {
            "with_genres": genre,
            "sort_by": sort_by,
            "vote_count.gte": min_votes,
            "include_adult": "false",
            "page": page,
        }
        if release_date_gte:
            params["primary_release_date.gte"] = release_date_gte
        if release_date_lte:
            params["primary_release_date.lte"] = release_date_lte

        data = await self._get("/discover/movie", params)
        results = data.get("results") if isinstance(data, dict) else None
        return list(results or [])

    async def search_movie(self, title: str, *, year: int | None = None) -> list[dict[str, Any]]:
        """Search movies by title (used to force-seed the evaluation gold set)."""
        params: dict[str, Any] = {"query": title, "include_adult": "false"}
        if year is not None:
            params["primary_release_year"] = year
        data = await self._get("/search/movie", params)
        results = data.get("results") if isinstance(data, dict) else None
        return list(results or [])

    async def get_movie(self, tmdb_id: int) -> dict[str, Any]:
        """Fetch full movie details.

        ``/discover`` does not return ``imdb_id``, so a detail call is required
        for every film regardless. We batch the sub-resources we need via
        ``append_to_response`` so one request yields imdb_id, crew, cast and
        certification instead of four.
        """
        return await self._get(
            f"/movie/{tmdb_id}",
            {"append_to_response": "credits,external_ids,release_dates,keywords"},
        )

    async def get_keywords(self, tmdb_id: int) -> list[str]:
        """Fetch a film's keyword tags.

        These carry the tone/subgenre vocabulary ("isolation", "paranoia",
        "transformation") that plot summaries lack, which is what the
        evaluation baseline identified as the retrieval gap.
        """
        data = await self._get(f"/movie/{tmdb_id}/keywords")
        return [k.get("name", "") for k in (data.get("keywords") or []) if k.get("name")]

    async def aclose(self) -> None:
        await self._client.aclose()


async def get_tmdb_client(**kwargs: Any) -> TMDBClient:
    return TMDBClient(**kwargs)
