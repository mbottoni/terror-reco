"""Where to watch: a snapshot of streaming availability per film and country.

Fetched offline from TMDB (which sources it from JustWatch) by
``scripts/fetch_watch_providers.py`` and stored in ``data/watch_providers.json``,
so, like everything else on the request path, a lookup is a dictionary access
with no network call.

Two things this data is not. It is not current: availability changes weekly
and this is whatever was true on ``fetched_at``, which is why every answer
carries that date. And it is not global: only the regions in ``REGIONS`` are
stored, and a visitor's region is a guess from their browser's language.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROVIDERS_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "watch_providers.json"

# Stored regions. Kept short on purpose: each one adds to a file that ships in
# the image, and a region nobody visits from is dead weight.
REGIONS: tuple[str, ...] = ("US", "GB", "CA", "AU", "BR", "DE", "FR", "ES")
DEFAULT_REGION = "US"

# A bare language tag says little about where someone is, so only the common,
# unambiguous ones are mapped. Everything else falls back to DEFAULT_REGION.
_LANGUAGE_REGION = {"pt": "BR", "de": "DE", "fr": "FR", "es": "ES"}

_CACHE: dict[str, Any] = {}


def load_providers() -> dict[str, Any]:
    """The snapshot, cached until the file changes. ``{}`` when not fetched."""
    try:
        mtime = PROVIDERS_FILE.stat().st_mtime
    except OSError:
        return {}
    if _CACHE.get("mtime") != mtime:
        try:
            data = json.loads(PROVIDERS_FILE.read_text())
        except (OSError, json.JSONDecodeError):
            data = {}
        _CACHE.update(mtime=mtime, data=data if isinstance(data, dict) else {})
    cached: dict[str, Any] = _CACHE["data"]
    return cached


def region_from_accept_language(header: str | None) -> str:
    """Best guess at a stored region from an ``Accept-Language`` header.

    Uses the first tag that names a stored region outright (``en-GB`` -> GB),
    then the first whose language implies one (``pt`` -> BR). Order follows
    the header, ignoring ``q`` weights: browsers already list by preference.
    """
    tags = [part.split(";")[0].strip() for part in (header or "").split(",")]
    for tag in tags:
        pieces = tag.replace("_", "-").split("-")
        if len(pieces) > 1 and pieces[-1].upper() in REGIONS:
            return pieces[-1].upper()
    for tag in tags:
        language = tag.replace("_", "-").split("-")[0].lower()
        if language in _LANGUAGE_REGION:
            return _LANGUAGE_REGION[language]
    return DEFAULT_REGION


def providers_for(imdb_id: str, region: str) -> dict[str, Any] | None:
    """Availability of one film in one region, or None when nothing is known.

    None covers three cases the caller treats alike: no snapshot, a film that
    is not in it, and a film with no listed provider in that region.
    """
    snapshot = load_providers()
    film = (snapshot.get("films") or {}).get(imdb_id) or {}
    entry = film.get(region) if region in REGIONS else None
    if not entry or not any(entry.get(kind) for kind in ("stream", "rent", "buy")):
        return None
    return {
        "region": region,
        "as_of": str(snapshot.get("fetched_at") or "")[:10],
        "stream": list(entry.get("stream") or []),
        "rent": list(entry.get("rent") or []),
        "buy": list(entry.get("buy") or []),
        # TMDB's own page for this film and region, which lists every
        # provider with deep links. Derived, so it is not stored per region.
        "link": (
            f"https://www.themoviedb.org/movie/{film['tmdb']}/watch?locale={region}"
            if film.get("tmdb")
            else None
        ),
    }
