"""Horror movie corpus for semantic recommendation.

The corpus is built **offline** by ``scripts/build_corpus.py`` (discovery via
TMDB, enrichment via OMDb) and cached to disk.  At request time this module
only loads it and runs vector search -- it never performs network I/O.

Workflow
--------
1. ``scripts/build_corpus.py``   -- build/refresh the corpus (offline, resumable)
2. ``load_corpus()``             -- load cached corpus
3. ``get_corpus_embeddings()``   -- compute / load plot embeddings
4. ``semantic_search()``         -- embed arbitrary user text, cosine-rank against corpus
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_DIR = _PROJECT_ROOT / "data"
CORPUS_FILE = CORPUS_DIR / "horror_corpus.json"
EMBEDDINGS_FILE = CORPUS_DIR / "corpus_embeddings.npy"
EMBEDDINGS_META_FILE = CORPUS_DIR / "corpus_embeddings.meta.json"

# Fields every corpus record must carry (consumed by templates and scorers).
REQUIRED_FIELDS = ("imdb_id", "title", "overview")


class CorpusNotBuiltError(RuntimeError):
    """Raised when a recommendation is requested before the corpus exists."""


# ----------------------------------------------------------------- load/save
def load_corpus() -> list[dict[str, Any]]:
    """Load the cached corpus from disk.  Returns ``[]`` if not built yet."""
    if not CORPUS_FILE.exists():
        return []
    with open(CORPUS_FILE) as f:
        data: list[dict[str, Any]] = json.load(f)
    return data


def save_corpus(corpus: list[dict[str, Any]]) -> None:
    """Persist corpus to disk and invalidate stale embeddings."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CORPUS_FILE, "w") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)
    for stale in (EMBEDDINGS_FILE, EMBEDDINGS_META_FILE):
        if stale.exists():
            stale.unlink()


# ------------------------------------------------------------------- mapping
def _first(values: list[str], limit: int | None = None) -> str | None:
    """Join non-empty strings, or return None when there are none."""
    cleaned = [v for v in values if v]
    if not cleaned:
        return None
    if limit is not None:
        cleaned = cleaned[:limit]
    return ", ".join(cleaned)


def _us_certification(detail: dict[str, Any]) -> str | None:
    """Extract the US certification (the analogue of OMDb's ``Rated``)."""
    blocks = (detail.get("release_dates") or {}).get("results") or []
    for block in blocks:
        if block.get("iso_3166_1") != "US":
            continue
        for release in block.get("release_dates") or []:
            cert = (release.get("certification") or "").strip()
            if cert:
                return cert
    return None


def map_tmdb_to_corpus(detail: dict[str, Any], *, image_base: str) -> dict[str, Any] | None:
    """Map a TMDB movie-detail payload onto the corpus record schema.

    Returns ``None`` when the film lacks the fields the pipeline depends on
    (an IMDb id to key feedback/dedup on, and an overview to embed).

    The output schema is byte-for-byte the one the previous OMDb-built corpus
    produced, so templates and scorers need no changes.
    """
    external = detail.get("external_ids") or {}
    imdb_id = external.get("imdb_id") or detail.get("imdb_id")
    overview = (detail.get("overview") or "").strip()
    if not imdb_id or not overview:
        return None

    credits = detail.get("credits") or {}
    crew = credits.get("crew") or []
    cast = credits.get("cast") or []

    directors = [c.get("name", "") for c in crew if c.get("job") == "Director"]
    writers = [c.get("name", "") for c in crew if c.get("job") in ("Writer", "Screenplay", "Story")]
    actors = [c.get("name", "") for c in cast]

    poster_path = detail.get("poster_path")
    release_date = (detail.get("release_date") or "").strip()
    runtime = detail.get("runtime")

    return {
        "imdb_id": imdb_id,
        "tmdb_id": detail.get("id"),
        "title": detail.get("title") or detail.get("original_title") or "",
        "overview": overview,
        "poster_url": f"{image_base.rstrip('/')}{poster_path}" if poster_path else None,
        "release_date": release_date or None,
        "year": release_date[:4] if release_date else None,
        # Placeholder on the IMDb scale; overwritten by OMDb enrichment below.
        "vote_average": float(detail["vote_average"]) if detail.get("vote_average") else None,
        "rating_source": "tmdb",
        "genre": _first([g.get("name", "") for g in detail.get("genres") or []]),
        "director": _first(directors),
        "actors": _first(actors, limit=4),
        "writer": _first(writers, limit=3),
        # OMDb rendered runtime as "121 min"; keep that so templates are unchanged.
        "runtime": f"{runtime} min" if runtime else None,
        "language": _first(
            [lang.get("english_name", "") for lang in detail.get("spoken_languages") or []]
        ),
        "country": _first([c.get("name", "") for c in detail.get("production_countries") or []]),
        "rated": _us_certification(detail),
        "awards": None,
        "imdbVotes": None,
        "Metascore": None,
    }


def apply_omdb_enrichment(record: dict[str, Any], omdb: dict[str, Any]) -> dict[str, Any]:
    """Overlay OMDb-only fields onto a TMDB-derived corpus record.

    ``vote_average`` is deliberately sourced from OMDb's ``imdbRating``: the
    ``min_rating`` filter and the popularity scorer (``rating * log(votes)``)
    both assume IMDb semantics, and the vote count comes from OMDb.  Blending
    TMDB ratings with IMDb vote counts would silently corrupt that signal.
    """
    if not omdb:
        return record

    rating_str = omdb.get("imdbRating") or ""
    if rating_str and rating_str != "N/A":
        try:
            record["vote_average"] = float(rating_str)
            record["rating_source"] = "imdb"
        except ValueError:
            pass

    for src, dst in (("imdbVotes", "imdbVotes"), ("Metascore", "Metascore"), ("Awards", "awards")):
        val = omdb.get(src)
        if val and val != "N/A":
            record[dst] = val

    # OMDb full plots are often longer than TMDB overviews; longer text gives
    # the embedding model more to work with.
    plot = (omdb.get("Plot") or "").strip()
    if plot and plot != "N/A" and len(plot) > len(record.get("overview") or ""):
        record["overview"] = plot

    return record


# ----------------------------------------------------------------- embeddings
def corpus_fingerprint(corpus: list[dict[str, Any]]) -> str:
    """Content hash of the exact text that gets embedded.

    Keyed on content rather than ``len(corpus)`` so that editing plot text
    without changing the film count still invalidates the cache.
    """
    h = hashlib.sha256()
    for movie in corpus:
        h.update((movie.get("imdb_id") or "").encode())
        h.update(b"\x00")
        h.update((movie.get("overview") or "").encode())
        h.update(b"\x00")
    return h.hexdigest()


def get_corpus_embeddings(corpus: list[dict[str, Any]]) -> np.ndarray:
    """Load or compute sentence-transformer embeddings for corpus plots."""
    fingerprint = corpus_fingerprint(corpus)

    if EMBEDDINGS_FILE.exists() and EMBEDDINGS_META_FILE.exists():
        try:
            with open(EMBEDDINGS_META_FILE) as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            meta = {}
        if meta.get("fingerprint") == fingerprint:
            cached: np.ndarray = np.load(EMBEDDINGS_FILE)
            if cached.shape[0] == len(corpus):
                return cached

    from .unified_recommender import _embed_sbert, _normalize_text

    texts = [_normalize_text(m.get("overview") or "") for m in corpus]
    print(f"  Computing embeddings for {len(texts)} movies...")
    embs = _embed_sbert(texts)

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_FILE, embs)
    with open(EMBEDDINGS_META_FILE, "w") as f:
        json.dump({"fingerprint": fingerprint, "count": len(corpus)}, f)
    return embs


# -------------------------------------------------------------- semantic search
def semantic_search(
    query: str,
    corpus: list[dict[str, Any]],
    corpus_embeddings: np.ndarray,
    top_k: int = 60,
    temperature: float = 1.0,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Rank corpus movies by semantic similarity to *any* arbitrary text.

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
    temperature:
        Controls randomness.  0 = fully deterministic.  1 = default variety.
        Noise is proportional to the score spread in the top candidates, so
        irrelevant movies never leak in.
    seed:
        Optional RNG seed.  Together with ``temperature=0`` this makes the
        search reproducible, which offline evaluation depends on.
    """
    from .unified_recommender import _embed_sbert, _normalize_text

    q_emb = _embed_sbert([_normalize_text(query)])  # (1, dim)
    sims = (q_emb @ corpus_embeddings.T).ravel()  # (n_corpus,)

    if temperature > 0:
        # Fetch a wider pool, then perturb scores within the relevant band
        pool_k = min(len(sims), top_k * 3)
        pool_idx = np.argsort(-sims)[:pool_k]
        pool_sims = sims[pool_idx]

        spread = float(pool_sims.max() - pool_sims.min()) if len(pool_sims) > 1 else 0.0
        noise_scale = spread * 0.08 * temperature
        noise = np.random.default_rng(seed).normal(0, noise_scale, size=len(pool_sims))
        perturbed = pool_sims + noise.astype(np.float32)

        reranked = np.argsort(-perturbed)[:top_k]
        top_idx = pool_idx[reranked]
    else:
        top_idx = np.argsort(-sims)[:top_k]

    results: list[dict[str, Any]] = []
    for idx in top_idx:
        movie = dict(corpus[int(idx)])
        movie["_semantic_score"] = float(sims[idx])
        results.append(movie)
    return results
