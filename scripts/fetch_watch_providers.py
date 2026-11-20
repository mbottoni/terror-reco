#!/usr/bin/env python3
"""Snapshot where each corpus film can be watched, per region.

Writes ``data/watch_providers.json``. Availability goes stale within weeks,
so this is meant to be re-run (``make providers``) and the file committed; the
app shows the snapshot date next to the data.

Usage (needs TMDB credentials in the environment):
    python scripts/fetch_watch_providers.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.corpus import load_corpus  # noqa: E402
from app.services.providers import PROVIDERS_FILE, REGIONS  # noqa: E402
from app.services.tmdb_client import get_tmdb_client  # noqa: E402

# TMDB lists the same service several times ("Netflix", "Netflix Standard
# with Ads", "Shudder Amazon Channel"). The reseller and tier variants add
# noise without telling anyone something new.
_NOISE = (" amazon channel", " apple tv channel", " with ads", " roku premium channel")
MAX_PER_KIND = 6


def _names(entries: list[dict[str, Any]] | None) -> list[str]:
    ordered = sorted(entries or [], key=lambda e: e.get("display_priority", 999))
    names: list[str] = []
    for entry in ordered:
        name = str(entry.get("provider_name") or "").strip()
        if not name or any(noise in name.lower() for noise in _NOISE) or name in names:
            continue
        names.append(name)
    return names[:MAX_PER_KIND]


def shape_region(raw: dict[str, Any]) -> dict[str, Any] | None:
    """One region's TMDB payload reduced to what the page shows."""
    shaped = {
        "stream": _names((raw.get("flatrate") or []) + (raw.get("free") or [])),
        "rent": _names(raw.get("rent")),
        "buy": _names(raw.get("buy")),
    }
    return shaped if (shaped["stream"] or shaped["rent"] or shaped["buy"]) else None


async def main() -> int:
    corpus = load_corpus()
    if not corpus:
        print("Corpus is empty. Run `make corpus` first.")
        return 1

    client = await get_tmdb_client()
    films: dict[str, dict[str, Any]] = {}
    failed = 0
    try:
        for i, movie in enumerate(corpus, 1):
            imdb_id, tmdb_id = movie.get("imdb_id"), movie.get("tmdb_id")
            if not imdb_id or not tmdb_id:
                continue
            try:
                results = await client.get_watch_providers(int(tmdb_id))
            except Exception as exc:  # noqa: BLE001 - keep going; report at the end
                failed += 1
                print(f"  ! {movie.get('title')}: {exc}")
                continue
            regions = {
                region: shaped
                for region in REGIONS
                if (shaped := shape_region(results.get(region) or {})) is not None
            }
            if regions:
                # The TMDB id is kept once per film: the "where to watch" link
                # is derived from it, which is far smaller than storing the
                # same long URL for every region.
                films[imdb_id] = {"tmdb": int(tmdb_id), **regions}
            if i % 100 == 0:
                print(f"  {i}/{len(corpus)}")
    finally:
        await client.aclose()

    snapshot = {
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": "TMDB watch providers (data by JustWatch)",
        "regions": list(REGIONS),
        "films": films,
    }
    PROVIDERS_FILE.write_text(json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")))
    size_kb = PROVIDERS_FILE.stat().st_size / 1024
    print(f"{len(films)} of {len(corpus)} films have a provider somewhere; failed {failed}")
    print(f"wrote {PROVIDERS_FILE.name}, {size_kb:.0f} KB")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
