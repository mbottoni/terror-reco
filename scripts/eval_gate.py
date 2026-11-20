"""A ranking regression gate that runs in CI without the model.

``make eval`` needs the sentence-transformer and takes minutes, so nothing in
CI noticed when a corpus rebuild dropped every ``keywords`` field and NDCG fell
by about 0.19. This module freezes a small slice of the real evaluation so a
plain pytest run can replay it:

* a few hundred corpus films (each dev-split mood's gold films plus its
  hardest distractors, the top of the real semantic ranking),
* their embeddings and the mood embeddings, precomputed with the real model,
* a hash of the exact text each vector was computed from.

The test (``tests/test_eval_gate.py``) feeds those vectors to the production
ranking code with ``temperature=0`` and asserts NDCG@6 has not collapsed.

The text hashes are the tripwire. Change ``embedding_text()``, or lose a field
in the corpus, and the stored vectors no longer describe what production would
embed -- the test fails and says to rebuild the fixture. Rebuilding is a
deliberate act: it re-records the baseline, so do it next to a ``make eval``.

Only the dev half of the moods is used, leaving the other half as a holdout.

Usage (needs the model; run after any intended change to corpus text):
    python scripts/eval_gate.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EVAL_NOTEBOOK = ROOT / "notebooks" / "1-evaluation.py"
FIXTURE_DIR = ROOT / "tests" / "fixtures"
FIXTURE_META = FIXTURE_DIR / "eval_gate.json"
FIXTURE_VECS = FIXTURE_DIR / "eval_gate.npz"

K = 6
POOL_K = 60
DISTRACTORS_PER_MOOD = 40
# How far below the recorded baseline NDCG may fall before the gate fails.
# The honest interval on a real change is about +/-0.06 (issue #32), so this
# is set to catch a collapse, not to police small movements.
TOLERANCE = 0.10


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def gold_cases() -> list[dict[str, Any]]:
    """The notebook's TEST_SET, read by AST so it cannot drift from the eval."""
    tree = ast.parse(EVAL_NOTEBOOK.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "TEST_SET" for t in node.targets
        ):
            cases: list[dict[str, Any]] = ast.literal_eval(node.value)
            return cases
    raise RuntimeError(f"TEST_SET not found in {EVAL_NOTEBOOK}")


def dev_split(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Same alternating split as ``scripts/run_eval.py --holdout dev``."""
    return [case for i, case in enumerate(cases) if i % 2 == 0]


def _normalize_title(title: str) -> str:
    title = title.lower().strip()
    for prefix in ("the ", "a ", "an "):
        if title.startswith(prefix):
            title = title[len(prefix) :]
    return title


def title_match(candidate: str, gold: list[str]) -> bool:
    """The notebook's matching rule, reproduced so the gate scores like the eval."""
    c = _normalize_title(candidate)
    for g in gold:
        gn = _normalize_title(g)
        if c == gn or c in gn or gn in c:
            return True
    return False


def ndcg_at_k(ranked_ids: list[str], gold_ids: set[str], n_gold: int, k: int = K) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, mid in enumerate(ranked_ids[:k]) if mid in gold_ids)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(n_gold, k)))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(
    moods: list[dict[str, Any]], films: list[dict[str, Any]], film_vecs: np.ndarray
) -> dict[str, float]:
    """Mean NDCG@6 of both production strategies over *moods*, deterministically.

    Calls the real ``semantic_search`` and ``recommend_unified_semantic``; the
    caller decides where query vectors come from (the real model when building
    the fixture, stored vectors in the test).
    """
    from app.services.corpus import semantic_search
    from app.services.unified_recommender import recommend_unified_semantic

    semantic: list[float] = []
    unified: list[float] = []
    for case in moods:
        gold_ids = set(case["gold_ids"])
        pool = semantic_search(case["mood"], films, film_vecs, top_k=POOL_K, temperature=0.0)
        semantic.append(ndcg_at_k([m["imdb_id"] for m in pool[:K]], gold_ids, case["n_gold"]))
        ranked = recommend_unified_semantic(
            mood=case["mood"], items=pool, limit=K, temperature=0.0, seed=0
        )
        unified.append(ndcg_at_k([m["imdb_id"] for m in ranked], gold_ids, case["n_gold"]))
    return {"semantic": float(np.mean(semantic)), "unified": float(np.mean(unified))}


def build() -> int:
    from app.services import corpus as corpus_mod
    from app.services.corpus import embedding_text, get_corpus_embeddings, load_corpus
    from app.services.unified_recommender import _embed_sbert, _normalize_text

    corpus = load_corpus()
    if not corpus:
        print("Corpus is empty. Run `make corpus` first.")
        return 1
    embeddings = get_corpus_embeddings(corpus)

    cases = dev_split(gold_cases())
    mood_vecs = _embed_sbert([_normalize_text(case["mood"]) for case in cases])

    keep: set[int] = set()
    moods: list[dict[str, Any]] = []
    for case, mood_vec in zip(cases, mood_vecs, strict=True):
        gold_rows = [i for i, m in enumerate(corpus) if title_match(m["title"], case["gold"])]
        top_rows = np.argsort(-(embeddings @ mood_vec))[:DISTRACTORS_PER_MOOD]
        keep.update(gold_rows)
        keep.update(int(i) for i in top_rows)
        moods.append(
            {
                "mood": case["mood"],
                "gold_ids": sorted(corpus[i]["imdb_id"] for i in gold_rows),
                "n_gold": len(case["gold"]),
            }
        )

    rows = sorted(keep)
    films = [corpus[i] for i in rows]
    # float16 halves the file; cosine rankings at this scale are unaffected,
    # and the baseline below is computed from the stored precision anyway.
    film_vecs16 = embeddings[rows].astype(np.float16)
    mood_vecs16 = mood_vecs.astype(np.float16)
    film_vecs = film_vecs16.astype(np.float32)

    # Score through the same substitutions the test makes, so the recorded
    # baseline is exactly what the test will measure.
    lookup = {
        text_sha(_normalize_text(case["mood"])): vec.astype(np.float32)
        for case, vec in zip(cases, mood_vecs16, strict=True)
    }
    with stored_vectors(corpus_mod, films, film_vecs, lookup):
        baseline = evaluate(moods, films, film_vecs)

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FIXTURE_VECS, film_vecs=film_vecs16, mood_vecs=mood_vecs16)
    FIXTURE_META.write_text(
        json.dumps(
            {
                "model": "sentence-transformers/all-mpnet-base-v2",
                "recipe": corpus_mod.EMBEDDING_RECIPE,
                "k": K,
                "tolerance": TOLERANCE,
                "baseline": {name: round(score, 4) for name, score in baseline.items()},
                "moods": moods,
                "films": [
                    {
                        "imdb_id": m["imdb_id"],
                        "text_sha": text_sha(_normalize_text(embedding_text(m))),
                    }
                    for m in films
                ],
            },
            indent=1,
        )
        + "\n"
    )
    size_kb = FIXTURE_VECS.stat().st_size / 1024
    print(f"{len(films)} films, {len(moods)} moods, vectors {size_kb:.0f} KB")
    print(
        f"baseline ndcg@{K}: semantic {baseline['semantic']:.4f}, unified {baseline['unified']:.4f}"
    )
    return 0


class stored_vectors:
    """Run the production ranking code on stored vectors instead of the model.

    Query (and, on a cache miss, document) texts are looked up by hash; an
    unknown text raises, because silently returning zeros would make a broken
    fixture look like a ranking regression.
    """

    def __init__(
        self,
        corpus_mod: Any,
        films: list[dict[str, Any]],
        film_vecs: np.ndarray,
        lookup: dict[str, np.ndarray],
    ) -> None:
        import app.services.unified_recommender as unified_mod

        self._corpus_mod = corpus_mod
        self._unified_mod = unified_mod
        self._films = films
        self._film_vecs = film_vecs
        self._lookup = lookup
        self._saved: tuple[Any, Any] | None = None

    def _embed(self, texts: list[str]) -> np.ndarray:
        try:
            vecs = np.stack([self._lookup[text_sha(text)] for text in texts])
        except KeyError as exc:
            raise KeyError(
                "eval-gate fixture has no vector for a text the ranker asked to embed"
            ) from exc
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        normalised: np.ndarray = (vecs / np.where(norms == 0, 1.0, norms)).astype(np.float32)
        return normalised

    def __enter__(self) -> stored_vectors:
        self._saved = (
            self._unified_mod._embed_sbert,
            self._corpus_mod.get_corpus_and_embeddings,
        )
        self._unified_mod._embed_sbert = self._embed
        self._corpus_mod.get_corpus_and_embeddings = lambda: (self._films, self._film_vecs)
        return self

    def __exit__(self, *exc: object) -> None:
        assert self._saved is not None
        self._unified_mod._embed_sbert, self._corpus_mod.get_corpus_and_embeddings = self._saved


if __name__ == "__main__":
    raise SystemExit(build())
