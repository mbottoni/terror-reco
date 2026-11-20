#!/usr/bin/env python3
"""Add TMDB franchise ("collection") membership to the existing corpus.

Sequels sit next to each other in embedding space, so two entries from one
franchise can both land in a six-film page. TMDB records which collection a
film belongs to; the corpus builder now keeps it (``map_tmdb_to_corpus``), and
this script backfills the films already in ``data/horror_corpus.json`` without
a full rebuild.

Only two fields are added (``collection_id``, ``collection``). The text that
gets embedded does not change, so cached embeddings stay valid.

Usage (needs TMDB credentials in the environment):
    python scripts/add_collections.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.corpus import CORPUS_FILE, collection_fields, load_corpus  # noqa: E402
from app.services.tmdb_client import get_tmdb_client  # noqa: E402


async def main() -> int:
    corpus = load_corpus()
    if not corpus:
        print("Corpus is empty. Run `make corpus` first.")
        return 1

    client = await get_tmdb_client()
    updated = failed = 0
    try:
        for i, movie in enumerate(corpus, 1):
            tmdb_id = movie.get("tmdb_id")
            if not tmdb_id:
                continue
            try:
                detail: dict[str, Any] = await client.get_movie(int(tmdb_id))
            except Exception as exc:  # noqa: BLE001 - keep going; report at the end
                failed += 1
                print(f"  ! {movie.get('title')}: {exc}")
                continue
            movie.update(collection_fields(detail))
            updated += 1
            if i % 100 == 0:
                print(f"  {i}/{len(corpus)}")
    finally:
        await client.aclose()

    # Written directly rather than through save_corpus(): that deletes the
    # embedding cache, and nothing that gets embedded has changed.
    CORPUS_FILE.write_text(json.dumps(corpus, indent=2, ensure_ascii=False))
    in_collection = sum(1 for m in corpus if m.get("collection_id"))
    print(
        f"updated {updated}, failed {failed}; {in_collection} of {len(corpus)} are in a collection"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
