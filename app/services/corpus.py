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
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_DIR = _PROJECT_ROOT / "data"
CORPUS_FILE = CORPUS_DIR / "horror_corpus.json"
EMBEDDINGS_FILE = CORPUS_DIR / "corpus_embeddings.npy"
EMBEDDINGS_META_FILE = CORPUS_DIR / "corpus_embeddings.meta.json"
TAGS_FILE = CORPUS_DIR / "film_tags.json"

# Fields every corpus record must carry (consumed by templates and scorers).
REQUIRED_FIELDS = ("imdb_id", "title", "overview")


class CorpusNotBuiltError(RuntimeError):
    """Raised when a recommendation is requested before the corpus exists."""


# ----------------------------------------------------------------- load/save
def load_film_tags() -> dict[str, list[str]]:
    """Descriptive tags per film (``imdb_id`` -> tags), or ``{}`` if absent.

    Kept in a file of their own, not in the corpus JSON: they are written by a
    model from a fixed vocabulary, not fetched from TMDB, and a corpus rebuild
    must not be able to wipe them (which is how the keywords were once lost).
    """
    if not TAGS_FILE.exists():
        return {}
    try:
        with open(TAGS_FILE) as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read %s; continuing without tags", TAGS_FILE)
        return {}
    tags: dict[str, list[str]] = raw.get("tags", {})
    return tags


def load_corpus() -> list[dict[str, Any]]:
    """Load the cached corpus from disk.  Returns ``[]`` if not built yet.

    Each film's tags are attached as a comma-separated ``tags`` field, the same
    shape as ``keywords``.
    """
    if not CORPUS_FILE.exists():
        return []
    with open(CORPUS_FILE) as f:
        data: list[dict[str, Any]] = json.load(f)
    tags = load_film_tags()
    for movie in data:
        own = tags.get(movie.get("imdb_id") or "")
        if own:
            movie["tags"] = ", ".join(own)
    return data


def save_corpus(corpus: list[dict[str, Any]]) -> None:
    """Persist corpus to disk and invalidate stale embeddings."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    # ``tags`` is attached at load time from its own file; it is not corpus data.
    records = [{k: v for k, v in movie.items() if k != "tags"} for movie in corpus]
    with open(CORPUS_FILE, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    for stale in (EMBEDDINGS_FILE, EMBEDDINGS_META_FILE):
        if stale.exists():
            stale.unlink()


# ------------------------------------------------------------------- mapping
def collection_fields(detail: dict[str, Any]) -> dict[str, Any]:
    """Franchise membership from a TMDB movie payload.

    ``collection_id`` is what results are de-duplicated on; the name is kept
    for humans reading the corpus. Both are None for a standalone film.
    """
    collection = detail.get("belongs_to_collection") or {}
    return {
        "collection_id": collection.get("id"),
        "collection": collection.get("name"),
    }


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
        **collection_fields(detail),
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
        # Tone/subgenre vocabulary absent from plot summaries; see embedding_text()
        "keywords": _first(
            [k.get("name", "") for k in (detail.get("keywords") or {}).get("keywords") or []]
        ),
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
def embedding_text(movie: dict[str, Any]) -> str:
    """Compose the text that represents a film in embedding space.

    Deliberately more than the plot.  Users describe *tone and subgenre*
    ("campy", "folk horror", "body horror"), but plot summaries are written
    in *narrative* language and contain almost none of that vocabulary --
    measured in docs/evaluation-baseline.md, where every mood scoring 0.000
    NDCG was a tone/subgenre query whose gold films were present in the
    corpus but unreachable.

    TMDB keywords supply exactly that missing register ("isolation",
    "paranoia", "transformation"), so they lead the composed document.
    """
    parts = [
        movie.get("title") or "",
        movie.get("genre") or "",
        movie.get("keywords") or "",
        movie.get("overview") or "",
    ]
    return " ".join(p.strip() for p in parts if p and p.strip())


# Share of each film's stored vector that comes from its keywords alone.
#
# One concatenated document is mean-pooled, so a 60-word overview outweighs a
# 6-word keyword list by token count, and the keywords are where the subgenre
# and tone vocabulary lives. Encoding the keywords a second time, alone, and
# mixing that vector in gives them a voice of their own. Measured on the graded
# benchmark (docs/evaluation-baseline.md): +0.038 NDCG@6 for retrieval, 95%
# interval [+0.014, +0.064]. Fields scored separately (max or mean of title,
# keywords, overview) were far worse; the concatenated document must stay the
# larger part.
#
# The mix is linear, so it is done once here and stored: a query's cosine
# against the mixed vector is the blend of its cosines against the two parts.
# Nothing downstream needs to know there were two.
KEYWORD_VECTOR_SHARE = 0.3
EMBEDDING_RECIPE = f"concat+tags,keywords:{KEYWORD_VECTOR_SHARE}"


def keyword_text(movie: dict[str, Any]) -> str:
    """The text of a film's second vector: its tags, then its TMDB keywords.

    Empty when the film has neither. The tags are deliberately *not* added to
    ``embedding_text()``: putting them in the long document was measured and
    lost most of the gain, for the reason this second vector exists at all.
    """
    parts = [(movie.get("tags") or "").strip(), (movie.get("keywords") or "").strip()]
    return ", ".join(p for p in parts if p)


def mix_keyword_vectors(
    document_vecs: np.ndarray, keyword_vecs: np.ndarray, has_keywords: np.ndarray
) -> np.ndarray:
    """Blend each document vector with its keyword vector and re-normalise.

    A film without keywords keeps its document vector unchanged. The result is
    unit length, which the taste vector, "more like this" and MMR all assume.
    """
    keyword_part = np.where(has_keywords[:, None], keyword_vecs, document_vecs)
    mixed = (1.0 - KEYWORD_VECTOR_SHARE) * document_vecs + KEYWORD_VECTOR_SHARE * keyword_part
    norms = np.linalg.norm(mixed, axis=1, keepdims=True)
    result: np.ndarray = (mixed / np.where(norms == 0, 1.0, norms)).astype(np.float32)
    return result


def corpus_fingerprint(corpus: list[dict[str, Any]]) -> str:
    """Content hash of the exact text that gets embedded.

    Keyed on the composed embedding text rather than ``len(corpus)``, so both
    editing a plot and changing how the document is composed invalidate the
    cache automatically.
    """
    h = hashlib.sha256()
    # How the vectors are built from that text is part of what is cached.
    h.update(EMBEDDING_RECIPE.encode())
    h.update(b"\x00")
    for movie in corpus:
        h.update((movie.get("imdb_id") or "").encode())
        h.update(b"\x00")
        h.update(embedding_text(movie).encode())
        h.update(b"\x00")
        h.update(keyword_text(movie).encode())
        h.update(b"\x00")
    return h.hexdigest()


def load_cached_embeddings(corpus: list[dict[str, Any]]) -> np.ndarray | None:
    """Return cached embeddings for *corpus*, or None if there is no valid cache.

    Never computes and never writes.  Callers that are incidental to a request
    (personalisation, say) must use this rather than
    :func:`get_corpus_embeddings`, so they can neither block a request on a
    full re-encode nor overwrite the cache with a partial corpus.
    """
    if not (EMBEDDINGS_FILE.exists() and EMBEDDINGS_META_FILE.exists()):
        return None
    try:
        with open(EMBEDDINGS_META_FILE) as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if meta.get("fingerprint") != corpus_fingerprint(corpus):
        return None
    cached: np.ndarray = np.load(EMBEDDINGS_FILE)
    return cached if cached.shape[0] == len(corpus) else None


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

    texts = [_normalize_text(embedding_text(m)) for m in corpus]
    logger.info("Computing embeddings for %d movies", len(texts))
    has_keywords = np.array([bool(keyword_text(m)) for m in corpus], dtype=bool)
    keyword_texts = [
        _normalize_text(keyword_text(m)) if has else text
        for m, has, text in zip(corpus, has_keywords, texts, strict=True)
    ]
    embs = mix_keyword_vectors(_embed_sbert(texts), _embed_sbert(keyword_texts), has_keywords)

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_FILE, embs)
    with open(EMBEDDINGS_META_FILE, "w") as f:
        json.dump({"fingerprint": fingerprint, "count": len(corpus)}, f)
    return embs


# -------------------------------------------------------------- semantic search
def resolve_query(
    query: str, corpus: list[dict[str, Any]], corpus_embeddings: np.ndarray
) -> tuple[np.ndarray, frozenset[str]]:
    """The vector to search with, shape ``(1, dim)``, and films to leave out.

    Ordinarily the embedded query and nothing excluded. When the query names
    corpus films, their vectors (see ``title_reference``) and those films.
    """
    from .title_reference import find_title_references, reference_vector
    from .unified_recommender import embed_query

    reference = find_title_references(query, corpus)
    if reference is None or corpus_embeddings.shape[0] != len(corpus):
        return embed_query(query), frozenset()  # (1, dim), cached per process
    vector = reference_vector(reference, corpus_embeddings, embed_query)
    return vector, reference.exclude_ids


def semantic_search(
    query: str,
    corpus: list[dict[str, Any]],
    corpus_embeddings: np.ndarray,
    top_k: int = 60,
    temperature: float = 1.0,
    seed: int | None = None,
    query_vector: np.ndarray | None = None,
    exclude_ids: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Rank corpus movies by semantic similarity to *any* arbitrary text.

    A query that names a corpus film ("like Hereditary") is searched with that
    film's vector and the film is left out; see ``title_reference``. A caller
    that searches a filtered slice of the corpus resolves the query against
    the whole corpus first and passes ``query_vector`` and ``exclude_ids``,
    because the named film may not be in the slice.

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
    if query_vector is None:
        query_vector, exclude_ids = resolve_query(query, corpus, corpus_embeddings)
    q_emb = query_vector
    sims = (q_emb @ corpus_embeddings.T).ravel()  # (n_corpus,)
    named: list[int] = []
    if exclude_ids:
        # Below every real cosine, so the noise step cannot bring them back.
        named = [i for i, m in enumerate(corpus) if str(m.get("imdb_id") or "") in exclude_ids]
        sims[named] = -2.0
        top_k = max(0, min(top_k, len(corpus) - len(named)))
    available = len(sims) - len(named)

    if temperature > 0:
        # Fetch a wider pool, then perturb scores within the relevant band
        # Never reaching into the masked films, whose scores would stretch
        # the spread the noise is scaled by.
        pool_k = min(available, top_k * 3)
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
        # Row index into `corpus_embeddings`. Downstream re-ranking can slice
        # the cached matrix instead of re-encoding text it already has vectors
        # for -- that re-encode cost ~2.4s per unified request.
        movie["_embedding_row"] = int(idx)
        results.append(movie)
    return results


_RUNTIME_CACHE: dict[str, Any] = {}


def get_corpus_and_embeddings() -> tuple[list[dict[str, Any]], np.ndarray]:
    """Corpus + embeddings, cached for the life of the process.

    The corpus is built offline and immutable at runtime, but every request
    used to re-read 740 KB of JSON and re-hash all 500 records to check the
    embedding fingerprint.  Keyed on the file's mtime so a rebuild is still
    picked up without a restart.
    """
    mtime = CORPUS_FILE.stat().st_mtime if CORPUS_FILE.exists() else 0.0
    if _RUNTIME_CACHE.get("mtime") != mtime:
        corpus = load_corpus()
        _RUNTIME_CACHE.update(
            mtime=mtime,
            corpus=corpus,
            embeddings=get_corpus_embeddings(corpus) if corpus else np.zeros((0, 0)),
        )
    corpus_out: list[dict[str, Any]] = _RUNTIME_CACHE["corpus"]
    embeddings_out: np.ndarray = _RUNTIME_CACHE["embeddings"]
    return corpus_out, embeddings_out


def embeddings_for(items: list[dict[str, Any]], corpus_embeddings: np.ndarray) -> np.ndarray | None:
    """Slice cached embeddings for *items* using their ``_embedding_row`` hints.

    Returns None when any item lacks a usable hint, so callers fall back to
    encoding rather than silently ranking against mismatched vectors.
    """
    rows: list[int] = []
    for item in items:
        row = item.get("_embedding_row")
        if not isinstance(row, int) or not 0 <= row < corpus_embeddings.shape[0]:
            return None
        rows.append(row)
    if not rows:
        return None
    return corpus_embeddings[rows]
