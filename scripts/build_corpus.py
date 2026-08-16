#!/usr/bin/env python3
"""Build the horror movie corpus offline.

Discovery runs against TMDB (real genre browsing) and enrichment against OMDb
(IMDb rating/votes, Metascore, awards).  The build is *resumable*: progress is
checkpointed after every stage and every batch, so a rate-limit or a crash
costs only the outstanding work, never what already completed.

Usage
-----
    python scripts/build_corpus.py --target 500
    python scripts/build_corpus.py --target 500 --resume
    python scripts/build_corpus.py --validate-only

Stages
------
A  gold      force-seed the evaluation gold set (so offline eval is meaningful)
B  canon     most-voted horror films (the canon)
C  decades   decade-stratified fill (prevents an all-modern corpus)
D  hydrate   TMDB detail fetch -> corpus records
E  enrich    OMDb overlay (IMDb rating/votes, Metascore, awards)
F  finalize  write corpus + report validation
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.corpus import (  # noqa: E402
    CORPUS_DIR,
    apply_omdb_enrichment,
    load_corpus,
    map_tmdb_to_corpus,
    save_corpus,
)
from app.services.tmdb_client import GENRE_HORROR, TMDBClient  # noqa: E402
from app.settings import get_settings  # noqa: E402

STATE_FILE = CORPUS_DIR / ".corpus_build_state.json"
EVAL_NOTEBOOK = PROJECT_ROOT / "notebooks" / "1-evaluation.py"

# Decade buckets for stratified discovery.  Without this, sorting by vote
# count yields an overwhelmingly post-2000 corpus.
DECADES: list[tuple[str, str, str]] = [
    ("pre-1970", "1900-01-01", "1969-12-31"),
    ("1970s", "1970-01-01", "1979-12-31"),
    ("1980s", "1980-01-01", "1989-12-31"),
    ("1990s", "1990-01-01", "1999-12-31"),
    ("2000s", "2000-01-01", "2009-12-31"),
    ("2010s", "2010-01-01", "2019-12-31"),
    ("2020s", "2020-01-01", "2029-12-31"),
]


# ------------------------------------------------------------------- state
def load_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return {"stages": {}, "tmdb_ids": [], "records": {}, "enriched": []}
    with open(STATE_FILE) as f:
        state: dict[str, Any] = json.load(f)
    return state


def save_state(state: dict[str, Any]) -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f)
    tmp.replace(STATE_FILE)


# --------------------------------------------------------------- gold titles
def gold_titles() -> list[str]:
    """Extract the evaluation gold titles from the marimo notebook.

    Parsed via AST rather than duplicated here so the seed list cannot drift
    from what the evaluation framework actually scores against.
    """
    if not EVAL_NOTEBOOK.exists():
        print(f"  ! {EVAL_NOTEBOOK.name} not found; skipping gold seeding")
        return []
    tree = ast.parse(EVAL_NOTEBOOK.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "TEST_SET":
                cases = ast.literal_eval(node.value)
                titles = {t.lower().strip() for case in cases for t in case.get("gold", [])}
                return sorted(titles)
    return []


def _norm_title(s: str) -> str:
    """Lowercase and strip punctuation/spacing so "vs." matches "vs"."""
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _title_matches(candidate: str, wanted: str) -> bool:
    c, w = _norm_title(candidate), _norm_title(wanted)
    return bool(w) and (c == w or c.startswith(w) or w in c)


# ------------------------------------------------------------------- stages
async def stage_gold(client: TMDBClient, state: dict[str, Any]) -> None:
    """Force-seed the evaluation gold set."""
    titles = gold_titles()
    print(f"[A] Gold seeding: {len(titles)} titles")
    ids: set[int] = set(state["tmdb_ids"])
    misses: list[str] = []

    for i, title in enumerate(titles, 1):
        results = await client.search_movie(title)
        if not results:
            misses.append(title)
            continue
        # Prefer an exact-ish title match carrying a horror-adjacent genre.
        genre_ok = {GENRE_HORROR, 53, 9648}
        best = next(
            (
                r
                for r in results
                if _title_matches(r.get("title", ""), title)
                and genre_ok & set(r.get("genre_ids") or [])
            ),
            None,
        )
        if best is None:
            best = next(
                (r for r in results if _title_matches(r.get("title", ""), title)),
                results[0],
            )
            misses.append(f"{title} -> {best.get('title')} ({best.get('release_date', '?')[:4]})")
        ids.add(int(best["id"]))
        if i % 25 == 0:
            print(f"    {i}/{len(titles)} searched, {len(ids)} ids")
            state["tmdb_ids"] = sorted(ids)
            save_state(state)

    state["tmdb_ids"] = sorted(ids)
    state["stages"]["gold"] = True
    state["gold_misses"] = misses
    save_state(state)
    print(f"    collected {len(ids)} ids; {len(misses)} low-confidence/missed")
    for m in misses[:10]:
        print(f"      ? {m}")


async def stage_canon(client: TMDBClient, state: dict[str, Any], pages: int) -> None:
    """Most-voted horror films -- approximates the canon and is stable."""
    print(f"[B] Canon: {pages} pages of vote_count.desc")
    ids: set[int] = set(state["tmdb_ids"])
    for page in range(1, pages + 1):
        results = await client.discover_movies(page=page, min_votes=200)
        if not results:
            break
        ids.update(int(r["id"]) for r in results)
    state["tmdb_ids"] = sorted(ids)
    state["stages"]["canon"] = True
    save_state(state)
    print(f"    {len(ids)} ids after canon")


async def stage_decades(client: TMDBClient, state: dict[str, Any], target: int) -> None:
    """Round-robin across decades until the id pool reaches *target*."""
    print(f"[C] Decade fill toward {target} ids")
    ids: set[int] = set(state["tmdb_ids"])
    page = 1
    while len(ids) < target and page <= 10:
        for label, gte, lte in DECADES:
            if len(ids) >= target:
                break
            results = await client.discover_movies(
                page=page, min_votes=50, release_date_gte=gte, release_date_lte=lte
            )
            before = len(ids)
            ids.update(int(r["id"]) for r in results)
            if page == 1:
                print(f"    {label}: +{len(ids) - before}")
        page += 1
    state["tmdb_ids"] = sorted(ids)
    state["stages"]["decades"] = True
    save_state(state)
    print(f"    {len(ids)} ids after decade fill")


async def stage_hydrate(client: TMDBClient, state: dict[str, Any], target: int) -> None:
    """Fetch TMDB details and map onto corpus records."""
    image_base = get_settings().TMDB_IMAGE_BASE
    todo = [i for i in state["tmdb_ids"] if str(i) not in state["records"]]
    print(f"[D] Hydrate: {len(todo)} to fetch ({len(state['records'])} cached)")

    dropped = 0
    for n, tmdb_id in enumerate(todo, 1):
        if len(state["records"]) >= target:
            break
        detail = await client.get_movie(int(tmdb_id))
        record = map_tmdb_to_corpus(detail, image_base=image_base) if detail else None
        if record is None:
            dropped += 1
        else:
            state["records"][str(tmdb_id)] = record
        if n % 50 == 0:
            print(f"    {n}/{len(todo)} fetched, {len(state['records'])} usable")
            save_state(state)

    state["stages"]["hydrate"] = True
    save_state(state)
    print(f"    {len(state['records'])} records ({dropped} dropped: no imdb_id or no overview)")


def _merge_with_existing(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fill empty fields in *records* from the corpus already on disk.

    The checkpoint is a parallel copy of the corpus, and it goes stale: records
    hydrated before a field existed keep lacking it, so writing the checkpoint
    straight out silently *deletes* anything added to the corpus afterwards.
    That is exactly how a ``--resume`` run wiped the TMDB keywords behind the
    +61% NDCG gain -- the corpus regressed with no error and no failing test.

    Merging by imdb_id makes the write additive: the checkpoint can add films
    and fields, but never remove a value the corpus already had.
    """
    existing = {m.get("imdb_id"): m for m in load_corpus() if m.get("imdb_id")}
    if not existing:
        return records

    restored = 0
    for record in records:
        prior = existing.get(record.get("imdb_id"))
        if not prior:
            continue
        for key, value in prior.items():
            if value in (None, "", [], {}):
                continue
            if record.get(key) in (None, "", [], {}):
                record[key] = value
                restored += 1
    if restored:
        print(f"  merged {restored} field(s) preserved from the existing corpus")
    return records


async def refresh_keywords() -> int:
    """Backfill TMDB keywords onto an already-built corpus.

    Corpora built before keywords were added to the pipeline have no tone or
    subgenre vocabulary, which the evaluation baseline showed is what makes
    mood queries fail. This tops them up without a full rebuild.
    """
    corpus = load_corpus()
    if not corpus:
        print("Corpus is empty; nothing to refresh.")
        return 1

    client = TMDBClient()
    updated = missing = 0
    try:
        for n, movie in enumerate(corpus, 1):
            tmdb_id = movie.get("tmdb_id")
            if not tmdb_id:
                missing += 1
                continue
            keywords = await client.get_keywords(int(tmdb_id))
            if keywords:
                movie["keywords"] = ", ".join(keywords)
                updated += 1
            if n % 100 == 0:
                print(f"    {n}/{len(corpus)} refreshed ({updated} with keywords)")
    finally:
        await client.aclose()

    save_corpus(corpus)

    # Write the keywords back into the checkpoint too. Updating only the corpus
    # left the checkpoint stale, so the next --resume run overwrote them.
    state = load_state()
    if state.get("records"):
        by_id = {m.get("imdb_id"): m for m in corpus}
        for record in state["records"].values():
            prior = by_id.get(record.get("imdb_id"))
            if prior and prior.get("keywords"):
                record["keywords"] = prior["keywords"]
        save_state(state)

    covered = sum(1 for m in corpus if m.get("keywords"))
    print(f"  keywords on {covered}/{len(corpus)} films ({missing} lacked a tmdb_id)")

    from app.services.corpus import get_corpus_embeddings

    embs = get_corpus_embeddings(corpus)
    print(f"  re-embedded with composed text: {embs.shape}")
    return 0


async def stage_enrich(state: dict[str, Any]) -> None:
    """Overlay OMDb fields.  Best-effort -- failures leave TMDB data intact."""
    from app.services.omdb_client import get_omdb_client

    settings = get_settings()
    if not settings.OMDB_API_KEY:
        print("[E] Enrich: skipped (no OMDB_API_KEY)")
        state["stages"]["enrich"] = True
        save_state(state)
        return

    client = await get_omdb_client()
    done = set(state["enriched"])
    todo = [k for k in state["records"] if k not in done]
    print(f"[E] Enrich via OMDb: {len(todo)} to fetch ({len(done)} cached)")

    failures = 0
    for n, key in enumerate(todo, 1):
        record = state["records"][key]
        imdb_id = record.get("imdb_id")
        if not imdb_id:
            continue
        try:
            omdb = await client.get_by_id(imdb_id, plot_full=True)
            state["records"][key] = apply_omdb_enrichment(record, omdb)
        except Exception as exc:  # noqa: BLE001 - enrichment is best-effort
            failures += 1
            if failures <= 3:
                print(f"    ! OMDb {imdb_id}: {exc}")
        done.add(key)
        if n % 50 == 0:
            print(f"    {n}/{len(todo)} enriched")
            state["enriched"] = sorted(done)
            save_state(state)

    state["enriched"] = sorted(done)
    state["stages"]["enrich"] = True
    save_state(state)
    await client.aclose()
    print(f"    enrichment complete ({failures} failures, tolerated)")


# --------------------------------------------------------------- validation
def validate(corpus: list[dict[str, Any]]) -> bool:
    print("\n=== Validation ===")
    ok = True

    total = len(corpus)
    with_overview = sum(1 for m in corpus if (m.get("overview") or "").strip())
    imdb_ids = [m.get("imdb_id") for m in corpus if m.get("imdb_id")]
    dupes = [k for k, v in Counter(imdb_ids).items() if v > 1]
    imdb_rated = sum(1 for m in corpus if m.get("rating_source") == "imdb")

    def check(label: str, passed: bool, detail: str) -> None:
        nonlocal ok
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}: {detail}")
        if not passed:
            ok = False

    check("corpus size", total >= 480, f"{total} films")
    check("overviews present", with_overview >= 480, f"{with_overview}/{total}")
    check("imdb_id present", len(imdb_ids) == total, f"{len(imdb_ids)}/{total}")
    check("no duplicate imdb_id", not dupes, f"{len(dupes)} duplicates")

    years = [int(m["year"]) for m in corpus if (m.get("year") or "").isdigit()]
    spread = Counter(y // 10 * 10 for y in years)
    top_decade, top_count = (spread.most_common(1) or [(0, 0)])[0]
    share = top_count / total if total else 1.0
    check("decade spread", share <= 0.40, f"{top_decade}s is {share:.0%} of corpus")

    titles = {(m.get("title") or "").lower().strip() for m in corpus}
    gold = gold_titles()
    hits = [g for g in gold if any(_title_matches(t, g) for t in titles)]
    check("gold titles present", len(hits) >= int(len(gold) * 0.85), f"{len(hits)}/{len(gold)}")

    print(f"  [INFO] IMDb-scale ratings: {imdb_rated}/{total}")
    print(f"  [INFO] decade histogram: {dict(sorted(spread.items()))}")

    missing_gold = [g for g in gold if g not in hits]
    if missing_gold:
        print(f"  [INFO] gold titles absent ({len(missing_gold)}): {missing_gold[:15]}")
    return ok


# --------------------------------------------------------------------- main
async def build(target: int, resume: bool, canon_pages: int, skip_embed: bool) -> int:
    state = (
        load_state()
        if resume
        else {
            "stages": {},
            "tmdb_ids": [],
            "records": {},
            "enriched": [],
        }
    )
    stages = state["stages"]

    # Raising --target on an already-complete build used to do nothing: every
    # stage was flagged done, so the run skipped straight to writing the same
    # corpus back out. Re-open the stages that gather films when the target
    # has grown.
    if len(state["records"]) < target and stages.get("hydrate"):
        print(
            f"  target {target} exceeds the {len(state['records'])} cached records; "
            "re-opening discovery and hydration"
        )
        stages.pop("decades", None)
        stages.pop("hydrate", None)
        stages.pop("enrich", None)

    client = TMDBClient()

    try:
        if not stages.get("gold"):
            await stage_gold(client, state)
        if not stages.get("canon"):
            await stage_canon(client, state, canon_pages)
        if not stages.get("decades"):
            await stage_decades(client, state, target)
        if not stages.get("hydrate"):
            await stage_hydrate(client, state, target)
    finally:
        await client.aclose()

    if not stages.get("enrich"):
        await stage_enrich(state)

    corpus = _merge_with_existing(list(state["records"].values()))
    corpus.sort(key=lambda m: (m.get("title") or "").lower())
    save_corpus(corpus)
    print(f"\n[F] Wrote {len(corpus)} films to data/horror_corpus.json")

    if not skip_embed:
        # Warm the embedding cache here rather than letting the first user
        # request pay the encode cost -- the whole point of an offline build.
        from app.services.corpus import get_corpus_embeddings

        embs = get_corpus_embeddings(corpus)
        print(f"[F] Embeddings ready: {embs.shape}")

    return 0 if validate(corpus) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the horror movie corpus")
    parser.add_argument("--target", type=int, default=500, help="target corpus size")
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
    parser.add_argument("--canon-pages", type=int, default=15, help="pages of vote_count.desc")
    parser.add_argument("--validate-only", action="store_true", help="validate existing corpus")
    parser.add_argument("--skip-embed", action="store_true", help="skip embedding generation")
    parser.add_argument(
        "--refresh-keywords",
        action="store_true",
        help="backfill TMDB keywords onto an existing corpus and re-embed",
    )
    args = parser.parse_args()

    if args.validate_only:
        return 0 if validate(load_corpus()) else 1
    if args.refresh_keywords:
        return asyncio.run(refresh_keywords())
    return asyncio.run(build(args.target, args.resume, args.canon_pages, args.skip_embed))


if __name__ == "__main__":
    raise SystemExit(main())
