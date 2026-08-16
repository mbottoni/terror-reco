#!/usr/bin/env python3
"""Headless runner for the evaluation framework in notebooks/1-evaluation.py.

Imports the notebook's *own* metric functions (via marimo's ``Cell.run()``)
rather than reimplementing them, so these numbers are exactly what the
notebook reports -- they cannot drift apart.

Reports three things:

A. **Deterministic semantic** -- ``semantic_search(temperature=0)``, top-6 by
   cosine. Reproducible, so it is the number to track across changes.
B. **Full unified pipeline** -- repeated N times with mean +/- stddev, because
   the pipeline adds noise at three independent stages and a single run is not
   a measurement.
C. **Gold coverage ceiling** -- how many gold titles are in the corpus at all.
   Without this you cannot tell a ranking failure from a missing film.

Usage:
    python scripts/run_eval.py [--runs 10] [--k 6]
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EVAL_NOTEBOOK = PROJECT_ROOT / "notebooks" / "1-evaluation.py"
METRICS = ("hit_rate@k", "precision@k", "ndcg@k", "mrr")


def load_notebook() -> Any:
    spec = importlib.util.spec_from_file_location("eval_nb", EVAL_NOTEBOOK)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {EVAL_NOTEBOOK}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the recommendation evaluation baseline")
    parser.add_argument("--runs", type=int, default=10, help="stochastic repetitions")
    parser.add_argument("--k", type=int, default=6, help="cutoff for @k metrics")
    args = parser.parse_args()

    nb = load_notebook()
    # marimo Cell.run() -> (display_output, defs); we want the defs.
    test_set = nb.gold_test_set.run()[1]["TEST_SET"]
    metric_defs = nb.define_metrics.run()[1]
    evaluate_ranker = metric_defs["evaluate_ranker"]
    score_pipeline = metric_defs["score_pipeline"]
    title_match = metric_defs["title_match"]

    from app.services.corpus import get_corpus_embeddings, load_corpus, semantic_search
    from app.services.unified_recommender import recommend_unified_semantic

    corpus = load_corpus()
    if not corpus:
        print("Corpus is empty. Run `make corpus` first.")
        return 1
    embeddings = get_corpus_embeddings(corpus)
    print(f"corpus: {len(corpus)} films, embeddings {embeddings.shape}, k={args.k}\n")

    def build_pools(temperature: float) -> dict[str, list[dict[str, Any]]]:
        pools: dict[str, list[dict[str, Any]]] = {}
        for entry in test_set:
            mood = entry["mood"]
            candidates = semantic_search(
                mood, corpus, embeddings, top_k=60, temperature=temperature
            )
            pools[mood] = [
                {k: v for k, v in m.items() if not k.startswith("_")} for m in candidates
            ]
        return pools

    # -- A: deterministic -----------------------------------------------------
    det_pools = build_pools(temperature=0.0)
    semantic_only = evaluate_ranker(
        lambda mood, items: items[: args.k], det_pools, test_set, k=args.k
    )

    # -- B: stochastic --------------------------------------------------------
    runs = [
        evaluate_ranker(
            lambda mood, items: recommend_unified_semantic(mood=mood, items=items, limit=args.k),
            build_pools(temperature=1.0),
            test_set,
            k=args.k,
        )
        for _ in range(args.runs)
    ]

    print("=" * 68)
    print(f"BASELINE  ({len(corpus)} films, {len(test_set)} moods, k={args.k})")
    print("=" * 68)
    print(f"{'metric':<14}{'A: semantic (det.)':>20}{'B: unified (n=' + str(args.runs) + ')':>26}")
    print("-" * 68)
    for m in METRICS:
        vals = [r[m] for r in runs]
        print(
            f"{m:<14}{semantic_only[m]:>20.4f}"
            f"{statistics.mean(vals):>19.4f} +/- {statistics.pstdev(vals):.4f}"
        )

    # -- C: coverage ceiling --------------------------------------------------
    titles = [m["title"] for m in corpus]
    print("\n" + "=" * 68)
    print("PER-MOOD  (A, deterministic)          ndcg   gold-in-corpus")
    print("-" * 68)
    total_gold = total_present = 0
    for entry in test_set:
        mood, gold = entry["mood"], entry["gold"]
        present = [g for g in gold if any(title_match(t, {g}) for t in titles)]
        total_gold += len(gold)
        total_present += len(present)
        ranked = [it["title"] for it in det_pools[mood][: args.k]]
        ndcg = score_pipeline(ranked, gold, args.k)["ndcg@k"]
        # A zero score with full coverage is a ranking failure, not a data gap.
        flag = "  <-- ranking failure" if ndcg == 0 and len(present) >= len(gold) - 2 else ""
        print(f"  {mood[:38]:<40}{ndcg:>6.3f}   {len(present):>2}/{len(gold):<3}{flag}")
    print("-" * 68)
    print(f"  {'gold coverage ceiling':<40}         {total_present}/{total_gold}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
