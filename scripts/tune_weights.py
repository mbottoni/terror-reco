#!/usr/bin/env python3
"""Grid-search the unified recommender's blend weights.

Reuses ``compute_signals()`` from the production path rather than
reimplementing the blend, so the weights this produces actually apply to the
running system.  Signals are computed once per mood and the grid is then pure
numpy, which is what makes an exhaustive search cheap.

IMPORTANT: with only 15 moods there is no meaningful holdout, so these
weights are tuned on the same set they are scored against.  Treat the number
as an upper bound, not an estimate of generalisation.

Usage:
    python scripts/tune_weights.py [--lambda 0.7] [--top 15]
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EVAL_NOTEBOOK = PROJECT_ROOT / "notebooks" / "1-evaluation.py"

SEM_VALS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
KW_VALS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
POP_VALS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
REC_VALS = [0.00, 0.05, 0.10]


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune unified blend weights")
    parser.add_argument("--lambda", dest="lam", type=float, default=0.7, help="MMR lambda")
    parser.add_argument("--top", type=int, default=15, help="configurations to print")
    parser.add_argument("--k", type=int, default=6)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("eval_nb", EVAL_NOTEBOOK)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load evaluation notebook")
    nb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nb)
    test_set = nb.gold_test_set.run()[1]["TEST_SET"]
    score_pipeline = nb.define_metrics.run()[1]["score_pipeline"]

    from app.services.corpus import get_corpus_embeddings, load_corpus, semantic_search
    from app.services.unified_recommender import (
        DEFAULT_WEIGHTS,
        _mmr,
        blend_signals,
        compute_signals,
    )

    corpus = load_corpus()
    if not corpus:
        print("Corpus is empty. Run `make corpus` first.")
        return 1
    embeddings = get_corpus_embeddings(corpus)

    # Precompute signals per mood once; the grid is then pure arithmetic.
    print(f"Precomputing signals for {len(test_set)} moods...")
    per_mood: list[tuple[list[dict[str, Any]], dict[str, np.ndarray], np.ndarray, list[str]]] = []
    for entry in test_set:
        mood = entry["mood"]
        pool = [
            {k: v for k, v in m.items() if not k.startswith("_")}
            for m in semantic_search(mood, corpus, embeddings, top_k=60, temperature=0.0)
        ]
        signals, item_embs = compute_signals(mood, pool)
        per_mood.append((pool, signals, item_embs, entry["gold"]))

    combos = [
        {"semantic": s, "keyword": k, "popularity": p, "recency": r}
        for s, k, p, r in itertools.product(SEM_VALS, KW_VALS, POP_VALS, REC_VALS)
        if abs(s + k + p + r - 1.0) < 1e-6
    ]
    print(f"Evaluating {len(combos)} weight combinations at lambda={args.lam}...\n")

    def score(weights: dict[str, float]) -> float:
        ndcgs = []
        for pool, signals, item_embs, gold in per_mood:
            blended = blend_signals(signals, weights)
            order = np.argsort(-blended)[: max(10, args.k * 5)]
            sel = _mmr(
                [pool[i] for i in order],
                sims=blended[order],
                k=args.k,
                lambda_=args.lam,
                embeddings=item_embs[order],
            )
            ndcgs.append(score_pipeline([m["title"] for m in sel], gold, args.k)["ndcg@k"])
        return float(np.mean(ndcgs))

    results = sorted(((score(w), w) for w in combos), key=lambda t: -t[0])

    baseline = score(DEFAULT_WEIGHTS)
    print(f"{'ndcg@' + str(args.k):>8}  sem   kw    pop   rec")
    print("-" * 40)
    for ndcg, w in results[: args.top]:
        print(
            f"{ndcg:>8.4f}  {w['semantic']:.2f}  {w['keyword']:.2f}  "
            f"{w['popularity']:.2f}  {w['recency']:.2f}"
        )
    print("-" * 40)
    d = DEFAULT_WEIGHTS
    print(
        f"{baseline:>8.4f}  {d['semantic']:.2f}  {d['keyword']:.2f}  "
        f"{d['popularity']:.2f}  {d['recency']:.2f}   <-- current default"
    )
    best_ndcg, best_w = results[0]
    print(f"\nbest improves on default by {best_ndcg - baseline:+.4f}")
    print("Tuned on all 15 moods with no holdout -- treat as an upper bound.")
    body = ", ".join(f'"{k}": {v}' for k, v in best_w.items())
    print("\nUNIFIED_WEIGHTS='{" + body + "}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
