#!/usr/bin/env python3
"""Headless runner for the evaluation framework in notebooks/1-evaluation.py.

Imports the notebook's *own* metric functions (via marimo's ``Cell.run()``)
rather than reimplementing them, so these numbers are exactly what the
notebook reports -- they cannot drift apart.

Reports five things:

A. **Deterministic semantic** -- ``semantic_search(temperature=0)``, top-6 by
   cosine. Reproducible, so it is the number to track across changes.
B. **Full unified pipeline** -- repeated N times with mean +/- stddev, because
   the pipeline adds noise at three independent stages and a single run is not
   a measurement.
C. **Paired A-vs-B comparison** -- per-mood deltas with a bootstrap confidence
   interval. Comparing two independent means throws away the fact that both
   arms saw the *same* moods; most of the variance is between-mood difficulty,
   which pairing cancels. A comparison whose CI straddles zero is not a result,
   and this is what says so out loud.
D. **Retrieval vs ranking** -- recall@60 of the candidate pool, and the best
   NDCG@6 that pool allows. Every metric here is measured at k=6 after both
   stages, so a mood scoring 0.000 could be a retriever that never found the
   film or a ranker that buried it. These two numbers tell those apart.
E. **Gold coverage ceiling** -- how many gold titles are in the corpus at all.
   Without this you cannot tell a ranking failure from a missing film.

Every run is seeded, so two invocations with the same arguments produce the
same numbers.

Usage:
    python scripts/run_eval.py [--runs 10] [--k 6] [--seed 0]
                              [--holdout dev|test|all]
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EVAL_NOTEBOOK = PROJECT_ROOT / "notebooks" / "1-evaluation.py"
METRICS = ("hit_rate@k", "precision@k", "ndcg@k", "mrr")
POOL_K = 60  # candidates retrieved before re-ranking; matches the request path
BOOTSTRAP_ITERS = 5000


def load_notebook() -> Any:
    spec = importlib.util.spec_from_file_location("eval_nb", EVAL_NOTEBOOK)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {EVAL_NOTEBOOK}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------- stats
def bootstrap_ci(
    values: list[float], rng: np.random.Generator, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of *values*.

    Resamples **moods**, not runs. With 15 moods the dominant source of
    uncertainty is which moods happen to be in the test set, and a CI over
    runs alone reports a precision the benchmark does not have.
    """
    if len(values) < 2:
        return (float("nan"), float("nan"))
    arr = np.asarray(values, dtype=np.float64)
    idx = rng.integers(0, len(arr), size=(BOOTSTRAP_ITERS, len(arr)))
    means = arr[idx].mean(axis=1)
    return (
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
    )


def per_mood_scores(
    ranker_fn: Callable[[str, list[dict[str, Any]]], list[dict[str, Any]]],
    pools: dict[str, list[dict[str, Any]]],
    test_set: list[dict[str, Any]],
    k: int,
    score_pipeline: Any,
) -> dict[str, dict[str, float]]:
    """Score every mood separately, keeping the per-mood breakdown.

    ``evaluate_ranker`` in the notebook averages immediately; pairing needs
    the individual scores, so this reuses the notebook's ``score_pipeline``
    and skips only the averaging step.
    """
    out: dict[str, dict[str, float]] = {}
    for entry in test_set:
        mood, gold = entry["mood"], entry["gold"]
        items = pools.get(mood, [])
        if not items:
            continue
        ranked = ranker_fn(mood, items)
        out[mood] = score_pipeline([it.get("title", "") for it in ranked], gold, k)
    return out


def mean_of(scores: dict[str, dict[str, float]], metric: str) -> float:
    return float(np.mean([s[metric] for s in scores.values()])) if scores else 0.0


def split_test_set(test_set: list[dict[str, Any]], holdout: str) -> list[dict[str, Any]]:
    """Deterministic dev/test split, so tuning has something to be checked against.

    `docs/evaluation-baseline.md` rejected the tuned weights partly because they
    were selected on all 15 moods with no holdout. Alternating by index keeps
    both halves comparable in difficulty without needing a shuffle seed.
    """
    if holdout == "dev":
        return [e for i, e in enumerate(test_set) if i % 2 == 0]
    if holdout == "test":
        return [e for i, e in enumerate(test_set) if i % 2 == 1]
    return test_set


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the recommendation evaluation baseline")
    parser.add_argument("--runs", type=int, default=10, help="stochastic repetitions")
    parser.add_argument("--k", type=int, default=6, help="cutoff for @k metrics")
    parser.add_argument("--seed", type=int, default=0, help="base seed; runs use seed+i")
    parser.add_argument(
        "--holdout",
        choices=("all", "dev", "test"),
        default="all",
        help="evaluate on half the moods, so a tuned result can be checked on the other half",
    )
    args = parser.parse_args()

    nb = load_notebook()
    # marimo Cell.run() -> (display_output, defs); we want the defs.
    test_set = nb.gold_test_set.run()[1]["TEST_SET"]
    test_set = split_test_set(test_set, args.holdout)
    metric_defs = nb.define_metrics.run()[1]
    score_pipeline = metric_defs["score_pipeline"]
    title_match = metric_defs["title_match"]

    from app.services.corpus import get_corpus_embeddings, load_corpus, semantic_search
    from app.services.unified_recommender import recommend_unified_semantic

    corpus = load_corpus()
    if not corpus:
        print("Corpus is empty. Run `make corpus` first.")
        return 1
    embeddings = get_corpus_embeddings(corpus)
    print(
        f"corpus: {len(corpus)} films, embeddings {embeddings.shape}, k={args.k}, "
        f"moods={len(test_set)} ({args.holdout}), seed={args.seed}\n"
    )

    def build_pools(temperature: float, seed: int | None) -> dict[str, list[dict[str, Any]]]:
        pools: dict[str, list[dict[str, Any]]] = {}
        for entry in test_set:
            mood = entry["mood"]
            candidates = semantic_search(
                mood, corpus, embeddings, top_k=POOL_K, temperature=temperature, seed=seed
            )
            pools[mood] = [
                {k: v for k, v in m.items() if not k.startswith("_")} for m in candidates
            ]
        return pools

    # -- A: deterministic -----------------------------------------------------
    det_pools = build_pools(temperature=0.0, seed=None)
    semantic_scores = per_mood_scores(
        lambda mood, items: items[: args.k], det_pools, test_set, args.k, score_pipeline
    )

    # -- B: stochastic --------------------------------------------------------
    # Seeded per run so the whole report is reproducible; the noise is still
    # there, it is just the *same* noise between invocations.
    unified_runs: list[dict[str, dict[str, float]]] = []
    for i in range(args.runs):
        run_seed = args.seed + i
        pools = build_pools(temperature=1.0, seed=run_seed)

        # `seed=s` binds the run's seed at definition time; reading the loop
        # variable would give every run the last seed and quietly destroy the
        # pairing this section exists to provide.
        def unified(mood: str, items: list[dict[str, Any]], s: int = run_seed) -> Any:
            return recommend_unified_semantic(mood=mood, items=items, limit=args.k, seed=s)

        unified_runs.append(
            per_mood_scores(
                unified,
                pools,
                test_set,
                args.k,
                score_pipeline,
            )
        )

    print("=" * 78)
    print(f"A/B  ({len(corpus)} films, {len(test_set)} moods, k={args.k})")
    print("=" * 78)
    print(f"{'metric':<14}{'A: semantic (det.)':>20}{'B: unified (n=' + str(args.runs) + ')':>26}")
    print("-" * 78)
    for m in METRICS:
        vals = [mean_of(run, m) for run in unified_runs]
        print(
            f"{m:<14}{mean_of(semantic_scores, m):>20.4f}"
            f"{statistics.mean(vals):>19.4f} +/- {statistics.pstdev(vals):.4f}"
        )

    # -- C: paired comparison -------------------------------------------------
    # Per mood, B's mean across runs minus A's score. Both arms saw the same
    # moods, so the (large) between-mood difficulty difference cancels instead
    # of being counted as noise in both arms.
    rng = np.random.default_rng(args.seed)
    print("\n" + "=" * 78)
    print("PAIRED  B - A, per mood, with a 95% bootstrap CI over moods")
    print("-" * 78)
    print(f"{'metric':<14}{'mean delta':>14}{'95% CI':>24}{'verdict':>24}")
    print("-" * 78)
    for m in METRICS:
        deltas = [
            float(np.mean([run[mood][m] for run in unified_runs if mood in run]))
            - semantic_scores[mood][m]
            for mood in semantic_scores
            if any(mood in run for run in unified_runs)
        ]
        lo, hi = bootstrap_ci(deltas, rng)
        # A CI containing zero means the sign of the effect is not established.
        verdict = "indistinguishable" if lo <= 0.0 <= hi else "B wins" if lo > 0 else "A wins"
        print(
            f"{m:<14}{statistics.mean(deltas):>+14.4f}"
            f"{'[' + format(lo, '+.4f') + ', ' + format(hi, '+.4f') + ']':>24}{verdict:>24}"
        )
    print(
        "\n  An effect smaller than this CI is not measurable with 15 synthetic moods,\n"
        "  however many times it is re-run. Widen the benchmark, do not chase it."
    )

    # -- D: retrieval vs ranking ----------------------------------------------
    # Everything above is measured at k=6 after retrieval AND ranking, so a
    # 0.000 is ambiguous. recall@60 is what retrieval alone achieved; the
    # ceiling is the best NDCG@6 obtainable from that pool by any re-ranker.
    print("\n" + "=" * 78)
    print("RETRIEVAL vs RANKING  (A, deterministic)")
    print("-" * 78)
    print(f"  {'mood':<40}{'ndcg':>7}{'ceil':>7}{'rec@' + str(POOL_K):>8}{'in corpus':>11}")
    print("-" * 78)
    corpus_titles = [m["title"] for m in corpus]
    total_recall: list[float] = []
    total_lost_to_retrieval = 0
    total_lost_to_ranking = 0
    for entry in test_set:
        mood, gold = entry["mood"], entry["gold"]
        pool_titles = [it["title"] for it in det_pools[mood]]
        in_corpus = [g for g in gold if any(title_match(t, {g}) for t in corpus_titles)]
        in_pool = [g for g in in_corpus if any(title_match(t, {g}) for t in pool_titles)]
        recall = len(in_pool) / len(in_corpus) if in_corpus else float("nan")
        if in_corpus:
            total_recall.append(recall)

        ndcg = score_pipeline(pool_titles[: args.k], gold, args.k)["ndcg@k"]
        # Best achievable: the gold films the pool actually contains, ranked first.
        ceiling = score_pipeline(in_pool[: args.k], gold, args.k)["ndcg@k"]
        total_lost_to_retrieval += len(in_corpus) - len(in_pool)
        total_lost_to_ranking += 1 if (ceiling > 0 and ndcg == 0) else 0

        print(
            f"  {mood[:38]:<40}{ndcg:>7.3f}{ceiling:>7.3f}"
            f"{recall:>8.2f}{str(len(in_corpus)) + '/' + str(len(gold)):>11}"
        )
    print("-" * 78)
    mean_recall = float(np.mean(total_recall)) if total_recall else 0.0
    print(f"  {'mean recall@' + str(POOL_K):<40}{mean_recall:>22.3f}")
    print(
        f"  gold films in the corpus but missed by retrieval: {total_lost_to_retrieval}\n"
        f"  moods where the pool held an answer and ranking buried it: "
        f"{total_lost_to_ranking}"
    )
    print(
        "\n  A mood with ceil=0.000 is a RETRIEVAL failure: no re-ranker, blend weight\n"
        "  or cross-encoder can fix it, because the answer is not in the pool.\n"
        "  A mood with ceil>0 and ndcg=0.000 is a RANKING failure and is worth tuning."
    )

    # -- E: coverage ceiling --------------------------------------------------
    total_gold = sum(len(e["gold"]) for e in test_set)
    total_present = sum(
        len([g for g in e["gold"] if any(title_match(t, {g}) for t in corpus_titles)])
        for e in test_set
    )
    print("\n" + "=" * 78)
    print(f"  {'gold coverage ceiling':<40}{str(total_present) + '/' + str(total_gold):>22}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
