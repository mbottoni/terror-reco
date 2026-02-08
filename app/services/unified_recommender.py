from __future__ import annotations

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
            model_name, cache_folder=cache_folder,
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


def _facet_proxy(mood: str, item: dict[str, Any]) -> float:
    q = set(_normalize_text(mood).split())
    t = set(
        (_normalize_text(item.get("title")) + " " + _normalize_text(item.get("overview"))).split()
    )
    if not q or not t:
        return 0.0
    inter = len(q & t)
    return inter / max(1, len(q))


def _mmr(
    items: list[dict[str, Any]], sims: np.ndarray, k: int, lambda_: float
) -> list[dict[str, Any]]:
    n = len(items)
    if n <= k:
        return items
    selected: list[int] = []
    candidates = set(range(n))
    first = int(np.argmax(sims))
    selected.append(first)
    candidates.remove(first)

    def _item_sim(i: int, j: int) -> float:
        ti = (
            _normalize_text(items[i].get("title")) + " " + _normalize_text(items[i].get("overview"))
        )
        tj = (
            _normalize_text(items[j].get("title")) + " " + _normalize_text(items[j].get("overview"))
        )
        si, sj = set(ti.split()), set(tj.split())
        if not si or not sj:
            return 0.0
        inter = len(si & sj)
        union = len(si | sj)
        return inter / union if union else 0.0

    while len(selected) < k and candidates:
        best_c = None
        best_score = -1e9
        for c in list(candidates):
            max_sim = 0.0
            for s in selected:
                max_sim = max(max_sim, _item_sim(c, s))
            score = lambda_ * float(sims[c]) - (1.0 - lambda_) * max_sim
            if score > best_score:
                best_score = score
                best_c = c
        if best_c is None:
            break
        selected.append(best_c)
        candidates.remove(best_c)
    return [items[i] for i in selected]


def recommend_unified_semantic(
    *,
    mood: str,
    items: list[dict[str, Any]],
    limit: int = 6,
    diversity_lambda: float = 0.7,
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    if not items:
        return []
    plots = [_normalize_text(mood)] + [_normalize_text(m.get("overview") or "") for m in items]
    embs = _embed_sbert(plots)
    mood_vec, plot_vecs = embs[0:1], embs[1:]
    sem = _cosine(mood_vec, plot_vecs).ravel()
    sem = _minmax(sem)

    kw = np.array([_facet_proxy(mood, it) for it in items], dtype=np.float32)
    kw = _minmax(kw)
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

    order = np.argsort(-blended)
    pool_idx = order[: max(10, limit * 5)]
    pool = [items[i] for i in pool_idx]
    pool_scores = blended[pool_idx]

    selected = _mmr(pool, sims=pool_scores, k=limit, lambda_=diversity_lambda)
    return selected
