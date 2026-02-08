"""Horror movie corpus for semantic recommendation.

Maintains a local cache of horror movies fetched broadly from OMDb.
All matching is done via sentence-transformer embeddings at query time --
no hardcoded mood-to-movie mappings.

Workflow
--------
1. ``build_corpus()``          -- fetch broadly from OMDb  (run once, cached to disk)
2. ``load_corpus()``           -- load cached corpus
3. ``get_corpus_embeddings()`` -- compute / load plot embeddings
4. ``semantic_search()``       -- embed arbitrary user text, cosine-rank against corpus
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Broad discovery terms for OMDb title search.
#
# These are NOT mood keywords.  They simply cast a wide net to build a
# diverse corpus of horror movies.  OMDb's ``s=`` parameter searches
# by *title*, so we use words that commonly appear in horror movie titles.
# ---------------------------------------------------------------------------
DISCOVERY_TERMS: list[str] = [
    # Common title words
    "horror", "dead", "evil", "night", "blood", "dark",
    "ghost", "devil", "hell", "curse", "haunted", "terror",
    "scream", "nightmare", "death", "kill", "fear",
    "fright", "tomb", "grave", "shadow",
    # Creatures & archetypes
    "zombie", "vampire", "demon", "witch", "alien", "werewolf",
    "creature", "monster", "dracula", "frankenstein", "mummy",
    # Iconic franchises & well-known titles
    "halloween", "saw", "conjuring", "exorcist", "friday 13",
    "omen", "purge", "insidious", "paranormal", "sinister",
    "hereditary", "babadook", "poltergeist", "candyman",
    "hellraiser", "chucky", "jaws", "cloverfield", "psycho",
    "ring", "grudge", "it", "us",
    # Subgenre & thematic
    "slasher", "possession", "haunting", "survival",
    "massacre", "cannibal", "asylum", "cabin", "ritual",
    "annihilation", "midsommar", "descent",
]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_DIR = _PROJECT_ROOT / "data"
CORPUS_FILE = CORPUS_DIR / "horror_corpus.json"
EMBEDDINGS_FILE = CORPUS_DIR / "corpus_embeddings.npy"


# ---------------------------------------------------------------------- build
def _save_corpus(corpus: list[dict[str, Any]]) -> None:
    """Persist corpus to disk and invalidate stale embeddings."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CORPUS_FILE, "w") as f:
        json.dump(corpus, f, indent=2)
    if EMBEDDINGS_FILE.exists():
        EMBEDDINGS_FILE.unlink()


async def build_corpus(
    pages: int = 2,
    max_details: int = 800,
    delay: float = 0.12,
) -> list[dict[str, Any]]:
    """Fetch a broad, diverse set of horror movies from OMDb.

    Searches OMDb with each *DISCOVERY_TERMS* entry, fetches full plot
    details, filters to the horror genre, and deduplicates by title.
    Results are cached to :data:`CORPUS_FILE` for reuse.

    Parameters
    ----------
    pages:
        How many search-result pages to fetch per discovery term.
    max_details:
        Cap on the number of detail requests (to stay within OMDb daily limits).
    delay:
        Seconds to wait between detail requests (rate-limit courtesy).
    """
    from .omdb_client import get_omdb_client

    client = await get_omdb_client()

    # Load any previously-built corpus so we can extend it
    existing = load_corpus()
    existing_ids: set[str] = {m["imdb_id"] for m in existing if "imdb_id" in m}

    # 1. Collect unique IMDb IDs via broad title searches
    raw_ids: list[str] = []
    total_queries = len(DISCOVERY_TERMS) * pages
    done = 0

    for term in DISCOVERY_TERMS:
        for page in range(1, pages + 1):
            done += 1
            try:
                results = await client.search_titles(term, page=page, type_="movie")
            except Exception as exc:
                print(f"  Search error ({term} p{page}): {exc}")
                continue
            for item in results or []:
                imdb_id = item.get("imdbID")
                if isinstance(imdb_id, str):
                    raw_ids.append(imdb_id)
            if done % 20 == 0:
                print(
                    f"  Corpus search: {done}/{total_queries} queries, "
                    f"{len(raw_ids)} raw IDs"
                )

    unique_ids = list(dict.fromkeys(raw_ids))
    # Skip IDs we already have
    new_ids = [i for i in unique_ids if i not in existing_ids]
    print(
        f"  {len(unique_ids)} unique IDs ({len(new_ids)} new, "
        f"{len(existing_ids)} already cached). Fetching details..."
    )

    # 2. Fetch full details, filtering to horror genre
    corpus: list[dict[str, Any]] = list(existing)
    seen_titles: set[str] = {m.get("title", "").lower().strip() for m in existing}
    consecutive_errors = 0
    fetched = 0

    for _i, imdb_id in enumerate(new_ids):
        if fetched >= max_details:
            print(f"  Reached max_details cap ({max_details}). Stopping.")
            break

        try:
            d = await client.get_by_id(imdb_id, plot_full=True)
            consecutive_errors = 0
        except Exception as exc:
            consecutive_errors += 1
            if consecutive_errors >= 5:
                print(
                    f"  {consecutive_errors} consecutive errors -- "
                    f"likely rate-limited. Stopping. Last error: {exc}"
                )
                break
            print(f"  Detail error ({imdb_id}): {exc}")
            continue

        fetched += 1

        if not d:
            continue
        genre = (d.get("Genre") or "").lower()
        if "horror" not in genre:
            continue

        title = d.get("Title") or ""
        key = title.lower().strip()
        if key in seen_titles:
            continue
        seen_titles.add(key)

        poster = d.get("Poster")
        poster_url = poster if poster and poster != "N/A" else None
        rating_str = d.get("imdbRating") or ""

        corpus.append(
            {
                "imdb_id": imdb_id,
                "title": title,
                "overview": d.get("Plot") or "",
                "poster_url": poster_url,
                "release_date": d.get("Released"),
                "year": d.get("Year"),
                "vote_average": (
                    float(rating_str) if rating_str and rating_str != "N/A" else None
                ),
                "genre": d.get("Genre"),
                "imdbVotes": d.get("imdbVotes"),
                "Metascore": d.get("Metascore"),
            }
        )

        if (fetched) % 50 == 0:
            print(
                f"  Detail progress: {fetched}/{len(new_ids)} fetched, "
                f"{len(corpus)} horror movies so far"
            )
            # Save periodically in case we get interrupted
            _save_corpus(corpus)

        # Small delay to be polite to the API
        if delay > 0:
            await asyncio.sleep(delay)

    print(f"  Corpus complete: {len(corpus)} horror movies")
    _save_corpus(corpus)
    return corpus


# ----------------------------------------------------------------------- load
def load_corpus() -> list[dict[str, Any]]:
    """Load the cached corpus from disk.  Returns ``[]`` if not built yet."""
    if not CORPUS_FILE.exists():
        return []
    with open(CORPUS_FILE) as f:
        data: list[dict[str, Any]] = json.load(f)
    return data


# ----------------------------------------------------------------- embeddings
def get_corpus_embeddings(corpus: list[dict[str, Any]]) -> np.ndarray:
    """Load or compute sentence-transformer embeddings for corpus plots.

    Embeddings are cached to :data:`EMBEDDINGS_FILE` so subsequent calls
    are a fast numpy load.
    """
    if EMBEDDINGS_FILE.exists():
        embs: np.ndarray = np.load(EMBEDDINGS_FILE)
        if embs.shape[0] == len(corpus):
            return embs

    from .unified_recommender import _embed_sbert, _normalize_text

    texts = [_normalize_text(m.get("overview") or "") for m in corpus]
    print(f"  Computing embeddings for {len(texts)} movies...")
    embs = _embed_sbert(texts)

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_FILE, embs)
    return embs


# -------------------------------------------------------------- semantic search
def semantic_search(
    query: str,
    corpus: list[dict[str, Any]],
    corpus_embeddings: np.ndarray,
    top_k: int = 60,
) -> list[dict[str, Any]]:
    """Rank corpus movies by semantic similarity to *any* arbitrary text.

    Uses sentence-transformer cosine similarity.  No hardcoded keywords.

    Parameters
    ----------
    query:
        Free-form text (mood description, plot synopsis, vibes, anything).
    corpus:
        The full list of horror movies (from :func:`load_corpus`).
    corpus_embeddings:
        Pre-computed plot embeddings (from :func:`get_corpus_embeddings`).
    top_k:
        How many results to return.

    Returns
    -------
    list[dict]
        Movies sorted by descending semantic similarity, each dict
        augmented with ``_semantic_score``.
    """
    from .unified_recommender import _embed_sbert, _normalize_text

    q_emb = _embed_sbert([_normalize_text(query)])  # (1, dim)
    sims = (q_emb @ corpus_embeddings.T).ravel()  # (n_corpus,)

    top_idx = np.argsort(-sims)[:top_k]

    results: list[dict[str, Any]] = []
    for idx in top_idx:
        movie = dict(corpus[int(idx)])
        movie["_semantic_score"] = float(sims[idx])
        results.append(movie)
    return results
