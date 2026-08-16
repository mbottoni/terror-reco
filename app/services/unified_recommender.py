from __future__ import annotations

from collections import Counter
from math import isfinite, log
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional in prod
    SentenceTransformer = None  # type: ignore[assignment,misc]

_SentenceTransformer: Any = SentenceTransformer  # keep as Any to avoid unreachable

_MODEL_CACHE: dict[str, Any] = {}

# Local model cache directory (avoids re-downloading from HuggingFace).
# Falls back to the default HF cache if the directory doesn't exist.
_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


def _normalize_text(s: str | None) -> str:
    return (s or "").strip().lower()


def _get_sbert(model_name: str = "sentence-transformers/all-mpnet-base-v2") -> Any:
    if _SentenceTransformer is None:
        return None
    if model_name not in _MODEL_CACHE:
        cache_folder = str(_MODELS_DIR) if _MODELS_DIR.is_dir() else None
        _MODEL_CACHE[model_name] = _SentenceTransformer(
            model_name,
            cache_folder=cache_folder,
        )
    return _MODEL_CACHE[model_name]


def _embed_sbert(texts: list[str]) -> np.ndarray:
    model = _get_sbert()
    if model is None:
        # Fallback: zeros; caller should handle low-signal gracefully
        return np.zeros((len(texts), 1), dtype=np.float32)
    vecs = model.encode(texts, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    result: np.ndarray = (a @ b.T).astype(np.float32)
    return result


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = float(np.min(x)), float(np.max(x))
    if not isfinite(lo) or not isfinite(hi) or abs(hi - lo) < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def _popularity(detail: dict[str, Any]) -> float:
    rating = float(detail.get("vote_average") or 0.0)
    votes_str = (detail.get("imdbVotes") or detail.get("imdb_votes_raw") or "0").replace(",", "")
    metascore_str = detail.get("Metascore") or detail.get("metascore_raw") or "0"
    try:
        votes = int(votes_str)
    except Exception:
        votes = 0
    try:
        metascore = int(metascore_str)
    except Exception:
        metascore = 0
    return rating * (1 + log(1 + votes)) + 0.02 * metascore


# Common English words carry no retrieval signal but appear in nearly every
# plot, so the old overlap ratio was largely measuring how many stopwords a
# query contained.
_STOP_WORDS = frozenset(
    """a an the and or but of in on at to for with from by as is are was were be been being
    it its this that these those he she they them his her their who whom which what when
    where why how all any both each few more most other some such no nor not only own same
    so than too very can will just about into over after before between during above below
    up down out off again further then once""".split()
)


def _tokenize(text: str) -> list[str]:
    return [w for w in _normalize_text(text).split() if w not in _STOP_WORDS and len(w) > 2]


def _item_text(item: dict[str, Any]) -> str:
    """Text used for lexical matching -- includes keywords when present."""
    parts = [
        item.get("title") or "",
        item.get("keywords") or "",
        item.get("genre") or "",
        item.get("overview") or "",
    ]
    return " ".join(p for p in parts if p)


def _bm25_scores(mood: str, items: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> Any:
    """Okapi BM25 of *mood* against each item's text.

    Replaces a raw ``|query ∩ doc| / |query|`` overlap ratio, which had no
    stopword filtering, no term-frequency saturation and no inverse document
    frequency -- so a query's common words dominated the score.
    """
    query = _tokenize(mood)
    docs = [_tokenize(_item_text(it)) for it in items]
    n = len(docs)
    scores = np.zeros(n, dtype=np.float32)
    if not query or n == 0:
        return scores

    avg_len = sum(len(d) for d in docs) / n or 1.0
    doc_counters = [Counter(d) for d in docs]
    # Document frequency per query term
    df = {term: sum(1 for c in doc_counters if term in c) for term in set(query)}

    for term in query:
        n_q = df.get(term, 0)
        if n_q == 0:
            continue
        # BM25 idf, floored at 0 so terms in most documents cannot go negative
        idf = max(0.0, log((n - n_q + 0.5) / (n_q + 0.5) + 1.0))
        for i, counter in enumerate(doc_counters):
            freq = counter.get(term, 0)
            if not freq:
                continue
            norm = 1 - b + b * (len(docs[i]) / avg_len)
            scores[i] += idf * (freq * (k1 + 1)) / (freq + k1 * norm)
    return scores


def _lexical_sim_matrix(items: list[dict[str, Any]]) -> np.ndarray:
    """Pairwise Jaccard over title+overview tokens (fallback when no embeddings)."""
    token_sets = [
        set((_normalize_text(it.get("title")) + " " + _normalize_text(it.get("overview"))).split())
        for it in items
    ]
    n = len(items)
    matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = token_sets[i], token_sets[j]
            union = len(si | sj)
            val = (len(si & sj) / union) if union else 0.0
            matrix[i, j] = matrix[j, i] = val
    return matrix


def _mmr(
    items: list[dict[str, Any]],
    sims: np.ndarray,
    k: int,
    lambda_: float,
    embeddings: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Maximal Marginal Relevance: balance relevance against redundancy.

    Redundancy is measured on the sentence-transformer embeddings when they
    are available.  The previous implementation used Jaccard overlap of
    title+overview *word sets*, which measured vocabulary reuse rather than
    semantic similarity, and rebuilt those sets inside an O(k*n) loop.  The
    embedding path computes one pairwise matrix up front instead.
    """
    n = len(items)
    if n <= k:
        return items

    if embeddings is not None and embeddings.shape[0] == n:
        # Embeddings are L2-normalised, so the dot product is cosine similarity.
        sim_matrix = (embeddings @ embeddings.T).astype(np.float32)
    else:
        sim_matrix = _lexical_sim_matrix(items)

    selected: list[int] = [int(np.argmax(sims))]
    candidates = set(range(n)) - {selected[0]}

    while len(selected) < k and candidates:
        idx = np.fromiter(candidates, dtype=np.int64)
        sel = np.asarray(selected, dtype=np.int64)
        # Redundancy of each candidate = its closest already-selected item.
        max_sim = sim_matrix[np.ix_(idx, sel)].max(axis=1)
        scores = lambda_ * sims[idx] - (1.0 - lambda_) * max_sim
        best = int(idx[int(np.argmax(scores))])
        selected.append(best)
        candidates.remove(best)

    return [items[i] for i in selected]


def recommend_unified_semantic(
    *,
    mood: str,
    items: list[dict[str, Any]],
    limit: int = 6,
    diversity_lambda: float = 0.7,
    weights: dict[str, float] | None = None,
    seed: int | None = None,
    temperature: float = 1.0,
) -> list[dict[str, Any]]:
    if not items:
        return []
    # Embed the SAME composed document the corpus embeddings use. Embedding
    # `overview` alone here silently reverted the semantic signal to the
    # plot-only text, discarding the keyword/genre composition that the
    # retrieval step depends on -- so unified scored worse than the plain
    # semantic search feeding it.
    from .corpus import embedding_text

    docs = [_normalize_text(mood)] + [_normalize_text(embedding_text(m)) for m in items]
    embs = _embed_sbert(docs)
    mood_vec, plot_vecs = embs[0:1], embs[1:]
    sem = _cosine(mood_vec, plot_vecs).ravel()
    sem = _minmax(sem)

    kw = _minmax(_bm25_scores(mood, items))
    pop = np.array([_popularity(it) for it in items], dtype=np.float32)
    pop = _minmax(pop)

    rec = np.zeros(len(items), dtype=np.float32)
    years: list[int | None] = []
    for it in items:
        y = it.get("year") or it.get("release_date") or ""
        try:
            y_int = int(str(y)[:4])
        except Exception:
            y_int = None
        years.append(y_int)
    valid = [y for y in years if isinstance(y, int)]
    if valid:
        y_arr = np.array([y if isinstance(y, int) else min(valid) for y in years], dtype=np.int32)
        rec = _minmax(y_arr.astype(np.float32))

    w = {"semantic": 0.45, "keyword": 0.20, "popularity": 0.20, "recency": 0.05}
    if weights:
        w.update(weights)
    blended = (
        w["semantic"] * sem + w["keyword"] * kw + w["popularity"] * pop + w["recency"] * rec
    ).astype(np.float32)

    # Add controlled noise to blended scores so results vary per request.
    # Noise scale is proportional to score spread, keeping top picks near
    # the top while shuffling the mid-band.  temperature=0 disables both
    # noise sources, making the call reproducible for evaluation.
    rng = np.random.default_rng(seed)
    if temperature > 0:
        spread = float(blended.max() - blended.min()) if len(blended) > 1 else 0.0
        noise = rng.normal(0, spread * 0.06 * temperature, size=len(blended)).astype(np.float32)
        blended_noisy = blended + noise
        # Slightly jitter diversity lambda so MMR selection also varies
        lambda_ = max(0.3, min(1.0, diversity_lambda + rng.uniform(-0.08, 0.08)))
    else:
        blended_noisy = blended
        lambda_ = diversity_lambda

    order = np.argsort(-blended_noisy)
    pool_idx = order[: max(10, limit * 5)]
    pool = [items[i] for i in pool_idx]
    pool_scores = blended_noisy[pool_idx]
    pool_embs = plot_vecs[pool_idx]

    selected = _mmr(pool, sims=pool_scores, k=limit, lambda_=lambda_, embeddings=pool_embs)
    return selected
