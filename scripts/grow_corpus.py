"""Add the next batch of horror films to the corpus.

Unlike ``build_corpus.py``, which rebuilds the catalogue from nothing, this
adds to it: the next most-voted horror films on TMDB that the corpus does not
hold yet. It is meant to run unattended on a schedule, so it is built around
the free API limits:

* **TMDB** has no daily quota, only a rate limit (about 40 requests a second;
  the client stays far below it). Cost per run: one request per page scanned
  and two per film (details, watch providers).
* **OMDb**'s free key allows 1,000 requests a day. Every film costs one, so a
  run never spends more than ``--omdb-budget`` (default 200), and stops
  cleanly if OMDb reports the limit anyway. Films it could not afford are
  simply candidates again tomorrow.

A film is added only when it has an IMDb id, an overview, a poster and an
IMDb rating with at least ``--min-imdb-votes`` votes. Anything else is
recorded in ``data/corpus_growth.json`` with the reason, and not asked about
again for 90 days (vote counts grow, so a rejection is not for ever).

New films get **predicted** tags (``scripts/tag_predictor.py``): nobody has
read them, and they are listed under ``predicted`` in ``data/film_tags.json``
so a later pass can replace them.

    python scripts/grow_corpus.py --add 40
    python scripts/grow_corpus.py --add 40 --dry-run     # report, write nothing
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

GROWTH_FILE = ROOT / "data" / "corpus_growth.json"
PROVIDERS_FILE = ROOT / "data" / "watch_providers.json"

# OMDb's free tier. The default budget leaves room for a manual rebuild or a
# second run on the same day.
OMDB_DAILY_LIMIT = 1000
DEFAULT_OMDB_BUDGET = 200
RETRY_REJECTED_AFTER_DAYS = 90
MAX_PAGES_SCANNED = 150  # 3,000 films deep; TMDB itself stops at page 500
HISTORY_KEPT = 60
MIN_RUNTIME_MINUTES = 60
# Not "music": Climax (2018) is filed under it. Runtime keeps music videos out.
UNWANTED_GENRES = ("family", "documentary")


class OmdbLimitReached(RuntimeError):
    """OMDb answered that today's quota is spent."""


def load_growth() -> dict[str, Any]:
    if GROWTH_FILE.exists():
        data: dict[str, Any] = json.loads(GROWTH_FILE.read_text())
        return data
    return {
        "about": (
            "State of scripts/grow_corpus.py: films it declined (and why), and a log of "
            "its runs. Committed so a scheduled run remembers what the last one did."
        ),
        "rejected": {},
        "runs": [],
    }


def votes_of(record: dict[str, Any]) -> int:
    raw = str(record.get("imdbVotes") or "0").replace(",", "")
    return int(raw) if raw.isdigit() else 0


def rejection_reason(record: dict[str, Any], *, min_imdb_votes: int) -> str | None:
    """Why *record* does not belong in the corpus, or None if it does."""
    if not (record.get("overview") or "").strip():
        return "no overview"
    if not record.get("poster_url"):
        return "no poster"
    if record.get("rating_source") != "imdb":
        return "no IMDb rating"
    if votes_of(record) < min_imdb_votes:
        return f"under {min_imdb_votes} IMDb votes"
    genres = str(record.get("genre") or "").lower()
    if "horror" not in genres:
        return "not listed as horror"
    # The best-rated list is full of things that are not films to sit down to:
    # music videos, shorts, and children's animation filed under horror.
    minutes = re.search(r"\d+", str(record.get("runtime") or ""))
    if minutes and int(minutes.group()) < MIN_RUNTIME_MINUTES:
        return f"shorter than {MIN_RUNTIME_MINUTES} minutes"
    for unwanted in UNWANTED_GENRES:
        if unwanted in genres:
            return f"listed as {unwanted}"
    return None


def still_rejected(entry: dict[str, Any], today: date) -> bool:
    try:
        when = date.fromisoformat(str(entry.get("on")))
    except ValueError:
        return False
    return today - when < timedelta(days=RETRY_REJECTED_AFTER_DAYS)


# Two ways of finding the next films, taken in turn. Vote count alone keeps
# adding recent wide releases (the first hundred it found were all from 2000 or
# later); the best-rated list is where older and non-English films are.
DISCOVERY_ORDERS = ("vote_count.desc", "vote_average.desc")
MIN_TMDB_VOTES_FOR_RATED = 300  # a 9.0 from forty voters is not acclaim


async def _walk(
    client: Any, *, sort_by: str, min_votes: int, known: set[int], skip: set[int], want: int
) -> tuple[list[int], int]:
    found: list[int] = []
    pages = 0
    for page in range(1, MAX_PAGES_SCANNED + 1):
        results = await client.discover_movies(page=page, sort_by=sort_by, min_votes=min_votes)
        pages = page
        if not results:
            break
        for row in results:
            tmdb_id = int(row["id"])
            if tmdb_id not in known and tmdb_id not in skip and tmdb_id not in found:
                found.append(tmdb_id)
        if len(found) >= want:
            break
    return found[:want], pages


async def discover_candidates(
    client: Any,
    *,
    known: set[int],
    skip: set[int],
    want: int,
    min_tmdb_votes: int,
    orders: tuple[str, ...] = DISCOVERY_ORDERS,
) -> tuple[list[int], int]:
    """The next *want* TMDB ids that are neither known nor skipped, alternating
    between the most-voted and the best-rated horror films."""
    lists: list[list[int]] = []
    pages = 0
    for order in orders:
        floor = (
            max(min_tmdb_votes, MIN_TMDB_VOTES_FOR_RATED) if "average" in order else min_tmdb_votes
        )
        found, walked = await _walk(
            client, sort_by=order, min_votes=floor, known=known, skip=skip, want=want
        )
        lists.append(found)
        pages += walked
    merged: list[int] = []
    for rank in range(max((len(found) for found in lists), default=0)):
        for found in lists:
            if rank < len(found) and found[rank] not in merged:
                merged.append(found[rank])
    return merged[:want], pages


async def enrich(omdb: Any, record: dict[str, Any]) -> dict[str, Any]:
    from app.services.corpus import apply_omdb_enrichment

    data = await omdb.get_by_id(record["imdb_id"], plot_full=True)
    if str(data.get("Response")) == "False" and "limit" in str(data.get("Error", "")).lower():
        raise OmdbLimitReached(str(data.get("Error")))
    return apply_omdb_enrichment(record, data)


async def fetch_providers(client: Any, tmdb_id: int) -> dict[str, Any] | None:
    from fetch_watch_providers import REGIONS, shape_region

    results = await client.get_watch_providers(tmdb_id)
    regions = {
        region: shaped
        for region in REGIONS
        if (shaped := shape_region(results.get(region) or {})) is not None
    }
    return {"tmdb": tmdb_id, **regions} if regions else None


async def grow(args: argparse.Namespace) -> int:
    from app.services.corpus import load_corpus, map_tmdb_to_corpus, save_corpus
    from app.services.omdb_client import get_omdb_client
    from app.services.tmdb_client import get_tmdb_client
    from app.settings import get_settings

    settings = get_settings()
    if not settings.OMDB_API_KEY:
        print("OMDB_API_KEY is not set: films cannot be given an IMDb rating. Nothing added.")
        return 1

    today = datetime.now(UTC).date()
    corpus = load_corpus()
    growth = load_growth()
    known_tmdb = {int(m["tmdb_id"]) for m in corpus if m.get("tmdb_id")}
    known_imdb = {str(m["imdb_id"]) for m in corpus}
    skip = {int(k) for k, v in growth["rejected"].items() if still_rejected(v, today)}

    budget = min(args.omdb_budget, OMDB_DAILY_LIMIT)
    # Some candidates will be declined; look at a few more than are wanted,
    # but never more than OMDb can be asked about today.
    want = min(budget, int(args.add * 1.6) + 5)

    tmdb = await get_tmdb_client()
    omdb = await get_omdb_client()
    added: list[dict[str, Any]] = []
    providers: dict[str, dict[str, Any]] = {}
    rejected_now: dict[str, dict[str, Any]] = {}
    omdb_calls = 0
    stopped = ""
    try:
        candidates, pages = await discover_candidates(
            tmdb,
            known=known_tmdb,
            skip=skip,
            want=want,
            min_tmdb_votes=args.min_tmdb_votes,
            orders={
                "both": DISCOVERY_ORDERS,
                "votes": DISCOVERY_ORDERS[:1],
                "rating": DISCOVERY_ORDERS[1:],
            }[args.order],
        )
        print(f"{len(corpus)} films in the corpus; {len(candidates)} candidates from {pages} pages")
        for tmdb_id in candidates:
            if len(added) >= args.add:
                break
            if omdb_calls >= budget:
                stopped = f"OMDb budget of {budget} requests spent"
                break
            detail = await tmdb.get_movie(tmdb_id)
            record = (
                map_tmdb_to_corpus(detail, image_base=settings.TMDB_IMAGE_BASE) if detail else None
            )
            title = (detail or {}).get("title") or str(tmdb_id)
            if record is None:
                rejected_now[str(tmdb_id)] = {"title": title, "why": "no IMDb id or overview"}
                continue
            if record["imdb_id"] in known_imdb:
                rejected_now[str(tmdb_id)] = {
                    "title": title,
                    "why": "IMDb id already in the corpus",
                }
                continue
            try:
                omdb_calls += 1
                record = await enrich(omdb, record)
            except OmdbLimitReached as exc:
                stopped = f"OMDb says: {exc}"
                break
            except Exception as exc:  # noqa: BLE001 - one bad answer must not end the run
                print(f"  ! OMDb {record['imdb_id']} ({title}): {exc}")
                continue
            why = rejection_reason(record, min_imdb_votes=args.min_imdb_votes)
            if why:
                rejected_now[str(tmdb_id)] = {"title": title, "why": why}
                continue
            added.append(record)
            known_imdb.add(record["imdb_id"])
            try:
                shaped = await fetch_providers(tmdb, tmdb_id)
                if shaped:
                    providers[record["imdb_id"]] = shaped
            except Exception as exc:  # noqa: BLE001 - where-to-watch is optional
                print(f"  ! providers {title}: {exc}")
    finally:
        await tmdb.aclose()
        await omdb.aclose()

    for record in added:
        print(
            f"  + {record['title']} ({record.get('year')})  IMDb {record.get('vote_average')}, "
            f"{votes_of(record):,} votes"
        )
    by_reason: dict[str, int] = {}
    for entry in rejected_now.values():
        by_reason[entry["why"]] = by_reason.get(entry["why"], 0) + 1
    print(
        f"added {len(added)}, declined {len(rejected_now)} {by_reason or ''}, "
        f"OMDb requests {omdb_calls}/{budget}" + (f"; stopped early: {stopped}" if stopped else "")
    )

    if args.dry_run:
        print("dry run: nothing written")
        return 0

    tagged = 0
    if added:
        save_corpus(corpus + added)
        if providers and PROVIDERS_FILE.exists():
            snapshot = json.loads(PROVIDERS_FILE.read_text())
            snapshot["films"].update(providers)
            PROVIDERS_FILE.write_text(
                json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
            )
        if not args.no_tags:
            from tag_predictor import tag_new_films

            tagged = tag_new_films([str(r["imdb_id"]) for r in added])
            print(f"predicted tags for {tagged} films")

    for key, entry in rejected_now.items():
        growth["rejected"][key] = {**entry, "on": today.isoformat()}
    growth["runs"] = (
        growth["runs"]
        + [
            {
                "on": today.isoformat(),
                "added": len(added),
                "declined": len(rejected_now),
                "omdb_requests": omdb_calls,
                "corpus_size": len(corpus) + len(added),
                **({"stopped": stopped} if stopped else {}),
            }
        ]
    )[-HISTORY_KEPT:]
    GROWTH_FILE.write_text(json.dumps(growth, indent=1, ensure_ascii=False) + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--add", type=int, default=40, help="films to add at most")
    parser.add_argument("--min-tmdb-votes", type=int, default=100, help="floor for discovery")
    parser.add_argument("--min-imdb-votes", type=int, default=5000, help="floor for inclusion")
    parser.add_argument(
        "--omdb-budget", type=int, default=DEFAULT_OMDB_BUDGET, help="OMDb requests at most"
    )
    parser.add_argument(
        "--order",
        choices=("both", "votes", "rating"),
        default="both",
        help="which TMDB lists to draw from",
    )
    parser.add_argument("--no-tags", action="store_true", help="skip tag prediction")
    parser.add_argument("--dry-run", action="store_true", help="report without writing")
    return asyncio.run(grow(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
