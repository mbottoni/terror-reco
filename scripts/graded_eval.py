#!/usr/bin/env python3
"""A wider benchmark scored from pooled, graded relevance judgments.

The original benchmark (``make eval``) is 15 moods with hand-written lists of
famous films. Two limits made it unable to decide anything (issue #32):

* **Mood sampling.** With 15 moods the honest interval on NDCG@6 is about
  +/-0.06, wider than any ranking change worth making.
* **False negatives.** A film that fits a mood but is not on its list counts
  as a miss. "Body horror" returns *The Brood* and *Re-Animator* and scores 0.

This module addresses both in the standard way: more moods (``eval/moods.json``)
and *pooling*. Run several rankers, take the union of what they return for a
mood, and judge that pool once, on a three-point scale:

    2  squarely fits: a film a horror fan would name for this mood
    1  defensible: fits in part, or fits a looser reading of the mood
    0  does not fit

The judgments live in ``eval/judgments.json``. A film a ranker returns that was
never judged is a **hole**: it scores 0 and is counted in the report. A ranker
with many holes has not been measured; add its results to the pool
(``pool --weights ...``), judge the new films, and score again.

Usage:
    python scripts/graded_eval.py pool   [-o FILE] [--weights JSON]
    python scripts/graded_eval.py score  [--runs 5] [--compare JSON]
    python scripts/graded_eval.py nulls
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_gate import gold_cases, title_match  # noqa: E402

MOODS_FILE = ROOT / "eval" / "moods.json"
JUDGMENTS_FILE = ROOT / "eval" / "judgments.json"

K = 6
POOL_K = 60  # candidates retrieved before re-ranking; matches the request path
POOL_DEPTH = 12  # how deep each deterministic ranker contributes to the pool
BOOTSTRAP_ITERS = 5000
KEYWORD_ONLY = {"semantic": 0.0, "keyword": 1.0, "popularity": 0.0, "recency": 0.0}

Ranker = Callable[[str], list[dict[str, Any]]]


# ----------------------------------------------------------------- data files
def load_moods() -> dict[str, list[str]]:
    data: dict[str, list[str]] = json.loads(MOODS_FILE.read_text())
    return {key: list(data.get(key, [])) for key in ("hand", "pooled", "null", "title")}


def load_judgments(path: Path = JUDGMENTS_FILE) -> dict[str, dict[str, int]]:
    """``{mood: {imdb_id: grade}}``. Empty when nothing has been judged yet."""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {
        mood: {row["imdb_id"]: int(row["grade"]) for row in rows}
        for mood, rows in raw.get("judgments", {}).items()
    }


# -------------------------------------------------------------------- metrics
def gain(grade: int) -> float:
    """Exponential gain, the usual choice: a 2 is worth three 1s."""
    return float(2**grade - 1)


def graded_ndcg(ranked_ids: list[str], grades: dict[str, int], k: int = K) -> float:
    """NDCG@k against graded judgments. Unjudged films gain nothing.

    The ideal ordering is built from every judged film for the mood, so a
    ranker is measured against the best page the pool is known to contain.
    """
    dcg = sum(gain(grades.get(mid, 0)) / math.log2(i + 2) for i, mid in enumerate(ranked_ids[:k]))
    ideal = sorted((gain(g) for g in grades.values()), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def precision_at(ranked_ids: list[str], grades: dict[str, int], floor: int, k: int = K) -> float:
    """Share of the top k judged at *floor* or better."""
    top = ranked_ids[:k]
    return sum(1 for mid in top if grades.get(mid, 0) >= floor) / k if top else 0.0


def holes(ranked_ids: list[str], grades: dict[str, int], k: int = K) -> int:
    return sum(1 for mid in ranked_ids[:k] if mid not in grades)


def bootstrap_ci(values: list[float], seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap for the mean, resampling moods."""
    if len(values) < 2:
        return (float("nan"), float("nan"))
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(BOOTSTRAP_ITERS, len(arr)))
    means = arr[idx].mean(axis=1)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


# -------------------------------------------------------------------- rankers
def _ids(films: list[dict[str, Any]]) -> list[str]:
    return [str(f.get("imdb_id")) for f in films]


def make_rankers(
    corpus: list[dict[str, Any]], embeddings: np.ndarray
) -> dict[str, Callable[..., list[dict[str, Any]]]]:
    """The production ranking functions, wrapped to take a mood."""
    from app.services.corpus import semantic_search
    from app.services.unified_recommender import recommend_unified_semantic

    def candidates(mood: str, temperature: float, seed: int | None) -> list[dict[str, Any]]:
        found = semantic_search(
            mood, corpus, embeddings, top_k=POOL_K, temperature=temperature, seed=seed
        )
        # The private ``_embedding_row`` hints are kept, as on the request path:
        # without them the re-ranker encodes all 60 candidates again per call.
        return found

    def semantic(mood: str, limit: int = K) -> list[dict[str, Any]]:
        return candidates(mood, 0.0, None)[:limit]

    def unified(
        mood: str,
        limit: int = K,
        weights: dict[str, float] | None = None,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> list[dict[str, Any]]:
        pool = candidates(mood, temperature, seed if temperature > 0 else None)
        ranked: list[dict[str, Any]] = recommend_unified_semantic(
            mood=mood, items=pool, limit=limit, weights=weights, temperature=temperature, seed=seed
        )
        return ranked

    def served(mood: str, strategy: str, seed: int) -> list[dict[str, Any]]:
        """Exactly what a visitor gets: the request path's calls, randomness on.

        The arms above call the ranking functions directly. This one goes
        through ``recommend_movies_advanced`` as the route does, so anything
        that function adds (it once drew the page at random from the top 18)
        is measured and not assumed away.
        """
        from app.services.recommender import recommend_movies_advanced

        if strategy == "semantic":
            return recommend_movies_advanced(mood=mood, limit=K, seed=seed, temperature=1.0)
        pool = recommend_movies_advanced(
            mood=mood, limit=POOL_K, seed=seed, temperature=1.0, keep_internal=True
        )
        ranked: list[dict[str, Any]] = recommend_unified_semantic(
            mood=mood, items=pool, limit=K, seed=seed, temperature=1.0
        )
        return ranked

    return {"semantic": semantic, "unified": unified, "served": served}


def load_ranking_inputs() -> tuple[list[dict[str, Any]], np.ndarray]:
    from app.services.corpus import get_corpus_embeddings, load_corpus

    corpus = load_corpus()
    if not corpus:
        raise SystemExit("Corpus is empty. Run `make corpus` first.")
    return corpus, get_corpus_embeddings(corpus)


# ----------------------------------------------------------------------- pool
def build_pool(weights: dict[str, float] | None = None) -> dict[str, list[dict[str, Any]]]:
    """Films to judge per mood: the union of several rankers' results.

    Rankers of different kinds are pooled on purpose (dense, blended, keyword
    only, and the noisy production path), so the judgments are not just a
    record of what today's ranker likes. The hand-written gold films are added
    for the moods that have them.
    """
    corpus, embeddings = load_ranking_inputs()
    rankers = make_rankers(corpus, embeddings)
    moods = load_moods()
    hand_gold = {case["mood"]: case["gold"] for case in gold_cases()}
    by_id = {str(m["imdb_id"]): m for m in corpus}

    pool: dict[str, list[dict[str, Any]]] = {}
    for mood in moods["hand"] + moods["pooled"]:
        ids: list[str] = []
        ids += _ids(rankers["semantic"](mood, POOL_DEPTH))
        ids += _ids(rankers["unified"](mood, POOL_DEPTH))
        ids += _ids(rankers["unified"](mood, 8, weights=KEYWORD_ONLY))
        for seed in range(5):
            ids += _ids(rankers["unified"](mood, K, temperature=1.0, seed=seed))
        if weights:
            ids += _ids(rankers["unified"](mood, POOL_DEPTH, weights=weights))
        ids += [
            str(m["imdb_id"]) for m in corpus if title_match(m["title"], hand_gold.get(mood, []))
        ]
        pool[mood] = [by_id[i] for i in dict.fromkeys(ids) if i in by_id]
    return pool


def cmd_pool(args: argparse.Namespace) -> int:
    weights = json.loads(args.weights) if args.weights else None
    judged = load_judgments()
    out: dict[str, list[dict[str, Any]]] = {}
    for mood, films in build_pool(weights).items():
        todo = [f for f in films if str(f["imdb_id"]) not in judged.get(mood, {})]
        if todo:
            out[mood] = [
                {
                    "imdb_id": f["imdb_id"],
                    "title": f["title"],
                    "year": f.get("year"),
                    "genre": f.get("genre"),
                    "keywords": f.get("keywords"),
                    "overview": (f.get("overview") or "")[:400],
                }
                for f in sorted(todo, key=lambda f: str(f["title"]))
            ]
    total = sum(len(v) for v in out.values())
    Path(args.output).write_text(json.dumps(out, indent=1) + "\n")
    print(f"{total} unjudged films across {len(out)} moods -> {args.output}")
    return 0


# ---------------------------------------------------------------------- score
def score_arm(
    ranker: Ranker, moods: list[str], judgments: dict[str, dict[str, int]]
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for mood in moods:
        grades = judgments[mood]
        ids = _ids(ranker(mood))
        out[mood] = {
            "ndcg": graded_ndcg(ids, grades),
            "p_fit": precision_at(ids, grades, 1),
            "p_square": precision_at(ids, grades, 2),
            "holes": float(holes(ids, grades)),
        }
    return out


def _mean(scores: dict[str, dict[str, float]], key: str) -> float:
    return float(np.mean([s[key] for s in scores.values()])) if scores else 0.0


def _average_runs(runs: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    return {
        mood: {key: float(np.mean([run[mood][key] for run in runs])) for key in runs[0][mood]}
        for mood in runs[0]
    }


def _print_paired(
    label: str, a: dict[str, dict[str, float]], b: dict[str, dict[str, float]]
) -> None:
    deltas = [b[mood]["ndcg"] - a[mood]["ndcg"] for mood in a]
    lo, hi = bootstrap_ci(deltas)
    verdict = "indistinguishable" if lo <= 0.0 <= hi else "second wins" if lo > 0 else "first wins"
    print(f"  {label:<34}{np.mean(deltas):>+9.4f}   [{lo:+.4f}, {hi:+.4f}]   {verdict}")


def cmd_score(args: argparse.Namespace) -> int:
    judgments = load_judgments()
    all_moods = load_moods()
    moods = [m for m in all_moods["hand"] + all_moods["pooled"] if judgments.get(m)]
    if not moods:
        print(f"No judgments in {JUDGMENTS_FILE}. Run `pool`, judge the films, then score.")
        return 1
    corpus, embeddings = load_ranking_inputs()
    rankers = make_rankers(corpus, embeddings)

    arms: dict[str, dict[str, dict[str, float]]] = {
        "semantic (det.)": score_arm(rankers["semantic"], moods, judgments),
        "unified (det.)": score_arm(rankers["unified"], moods, judgments),
        f"unified (noisy, n={args.runs})": _average_runs(
            [
                score_arm(
                    lambda mood, s=seed: rankers["unified"](mood, temperature=1.0, seed=s),
                    moods,
                    judgments,
                )
                for seed in range(args.runs)
            ]
        ),
    }
    for strategy in ("semantic", "unified"):
        arms[f"{strategy}, as served (n={args.runs})"] = _average_runs(
            [
                score_arm(
                    lambda mood, s=seed, st=strategy: rankers["served"](mood, st, s),
                    moods,
                    judgments,
                )
                for seed in range(args.runs)
            ]
        )
    compare = json.loads(args.compare) if args.compare else None
    if compare:
        arms[f"variant {args.compare}"] = score_arm(
            lambda mood: rankers["unified"](mood, weights=compare), moods, judgments
        )

    judged = sum(len(judgments[m]) for m in moods)
    print(f"{len(moods)} moods, {judged} judged films, k={K}, corpus {len(corpus)} films\n")
    print(f"  {'arm':<34}{'ndcg@6':>8}{'  95% CI':>20}{'fits@6':>9}{'square@6':>10}{'holes':>7}")
    print("  " + "-" * 86)
    for name, scores in arms.items():
        lo, hi = bootstrap_ci([s["ndcg"] for s in scores.values()])
        total_holes = sum(s["holes"] for s in scores.values())
        print(
            f"  {name[:33]:<34}{_mean(scores, 'ndcg'):>8.4f}"
            f"{'[' + format(lo, '.4f') + ', ' + format(hi, '.4f') + ']':>20}"
            f"{_mean(scores, 'p_fit'):>9.3f}{_mean(scores, 'p_square'):>10.3f}{total_holes:>7.1f}"
        )
    print(
        "\n  fits@6: share of the page judged 1 or 2.  square@6: share judged 2.\n"
        "  holes: unjudged films in the top 6, summed over moods; they score 0.\n"
        "  An arm with more than a few holes has not been measured: pool and judge it first."
    )

    names = list(arms)
    print("\n  PAIRED differences in ndcg@6 (second minus first), 95% bootstrap CI over moods")
    print("  " + "-" * 86)
    _print_paired("semantic -> unified (det.)", arms[names[0]], arms[names[1]])
    _print_paired("unified det. -> noisy", arms[names[1]], arms[names[2]])
    _print_paired("as served: semantic -> unified", arms[names[3]], arms[names[4]])
    if compare:
        _print_paired("unified (det.) -> variant", arms[names[1]], arms[names[5]])

    if args.per_mood:
        print(f"\n  {'mood':<52}{'semantic':>9}{'unified':>9}{'judged':>8}{'2s':>5}")
        print("  " + "-" * 86)
        for mood in moods:
            twos = sum(1 for g in judgments[mood].values() if g == 2)
            print(
                f"  {mood[:50]:<52}{arms[names[0]][mood]['ndcg']:>9.3f}"
                f"{arms[names[1]][mood]['ndcg']:>9.3f}{len(judgments[mood]):>8}{twos:>5}"
            )
    return 0


# ---------------------------------------------------------------------- nulls
def cmd_titles(args: argparse.Namespace) -> int:
    """Queries that name a film, under three treatments of the title.

    ``off`` is the plain text search. ``exclude only`` still searches with the
    query's text but leaves the named film and its franchise out, which is
    worth something by itself here because the named film is graded 0.
    ``film vector`` is what ships. Comparing the last two isolates what the
    film's vector adds beyond not repeating the film back.
    """
    import app.services.title_reference as tr

    judgments = load_judgments()
    queries = [q for q in load_moods()["title"] if judgments.get(q)]
    if not queries:
        print("No judged title queries.")
        return 1
    corpus, embeddings = load_ranking_inputs()
    rankers = make_rankers(corpus, embeddings)
    shipped = (tr.ENABLED, tr.USE_FILM_VECTOR, tr.FILM_SHARE, tr.KEYWORDS_FROM_FILM)

    def arms_for(label: str) -> dict[str, dict[str, dict[str, float]]]:
        return {
            f"semantic (det.), {label}": score_arm(rankers["semantic"], queries, judgments),
            f"unified (det.), {label}": score_arm(rankers["unified"], queries, judgments),
            f"unified as served, {label}": _average_runs(
                [
                    score_arm(
                        lambda q, s=seed: rankers["served"](q, "unified", s), queries, judgments
                    )
                    for seed in range(args.runs)
                ]
            ),
        }

    arms: dict[str, dict[str, dict[str, float]]] = {}
    try:
        tr.ENABLED = False
        arms.update(arms_for("off"))
        tr.ENABLED, tr.USE_FILM_VECTOR = True, False
        arms.update(arms_for("exclude only"))
        tr.USE_FILM_VECTOR = True
        arms.update(arms_for("film vector"))
        sweep: dict[str, dict[str, dict[str, float]]] = {}
        if args.sweep:
            for share in (0.5, 0.6, 0.7, 0.85, 1.0):
                for from_film in (True, False):
                    tr.FILM_SHARE, tr.KEYWORDS_FROM_FILM = share, from_film
                    name = f"unified share={share} kw_from_film={from_film}"
                    sweep[name] = score_arm(rankers["unified"], queries, judgments)
    finally:
        tr.ENABLED, tr.USE_FILM_VECTOR, tr.FILM_SHARE, tr.KEYWORDS_FROM_FILM = shipped

    judged = sum(len(judgments[q]) for q in queries)
    print(f"{len(queries)} title queries, {judged} judged films, k={K}\n")
    print(f"  {'arm':<36}{'ndcg@6':>8}{'fits@6':>9}{'square@6':>10}{'holes':>7}")
    print("  " + "-" * 70)
    for name, scores in {**arms, **sweep}.items():
        total_holes = sum(s["holes"] for s in scores.values())
        print(
            f"  {name[:35]:<36}{_mean(scores, 'ndcg'):>8.4f}{_mean(scores, 'p_fit'):>9.3f}"
            f"{_mean(scores, 'p_square'):>10.3f}{total_holes:>7.1f}"
        )
    print("\n  PAIRED differences in ndcg@6 (second minus first), 95% bootstrap CI over queries")
    print("  " + "-" * 86)
    for kind in ("semantic (det.)", "unified (det.)", "unified as served"):
        _print_paired(
            f"{kind[:17]}: off -> excl.", arms[f"{kind}, off"], arms[f"{kind}, exclude only"]
        )
        _print_paired(
            f"{kind[:17]}: excl. -> film",
            arms[f"{kind}, exclude only"],
            arms[f"{kind}, film vector"],
        )
    if args.per_mood:
        print()
        for q in queries:
            row = "  ".join(
                f"{arms[f'unified (det.), {label}'][q]['ndcg']:.2f}"
                for label in ("off", "exclude only", "film vector")
            )
            print(f"  {q[:44]:<46}{row}")
    return 0


def cmd_nulls(args: argparse.Namespace) -> int:
    """How confident does the retriever look on queries with no right answer?

    The app always returns a full page. Whether it *could* tell a query it
    cannot serve from one it can depends on whether the best cosine score
    separates the two groups; this prints that, and decides nothing.
    """
    from app.services.unified_recommender import embed_query

    corpus, embeddings = load_ranking_inputs()
    moods = load_moods()

    def top_scores(query: str) -> tuple[float, float]:
        sims = np.sort(embeddings @ np.asarray(embed_query(query)).reshape(-1))[::-1]
        return float(sims[0]), float(sims[:K].mean())

    real = {m: top_scores(m) for m in moods["hand"] + moods["pooled"]}
    null = {m: top_scores(m) for m in moods["null"]}
    for label, group in (("horror moods", real), ("null queries", null)):
        best = [v[0] for v in group.values()]
        print(
            f"{label:<14} n={len(group):<3} best cosine: min {min(best):.3f}  "
            f"median {float(np.median(best)):.3f}  max {max(best):.3f}"
        )
    print()
    for query, (best, mean6) in sorted(null.items(), key=lambda kv: -kv[1][0]):
        print(f"  null  {query[:44]:<46}{best:>7.3f}{mean6:>8.3f}")
    for query, (best, mean6) in sorted(real.items(), key=lambda kv: kv[1][0])[:8]:
        print(f"  real  {query[:44]:<46}{best:>7.3f}{mean6:>8.3f}")
    lowest_real = min(v[0] for v in real.values())
    highest_null = max(v[0] for v in null.values())
    overlap = sum(1 for v in null.values() if v[0] >= lowest_real)
    print(
        f"\n  lowest real {lowest_real:.3f}, highest null {highest_null:.3f}: "
        f"{overlap} of {len(null)} null queries score at least as high as the weakest real mood."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    pool = sub.add_parser("pool", help="write the films that still need judging")
    pool.add_argument("-o", "--output", default="eval/to_judge.json")
    pool.add_argument("--weights", help="JSON blend weights of a variant to add to the pool")
    pool.set_defaults(func=cmd_pool)
    score = sub.add_parser("score", help="score the rankers against the judgments")
    score.add_argument("--runs", type=int, default=5, help="noisy repetitions")
    score.add_argument("--compare", help="JSON blend weights of a variant to compare")
    score.add_argument("--per-mood", action="store_true")
    score.set_defaults(func=cmd_score)
    titles = sub.add_parser("titles", help="score the queries that name a film")
    titles.add_argument("--runs", type=int, default=5, help="noisy repetitions")
    titles.add_argument("--sweep", action="store_true", help="also vary the blend settings")
    titles.add_argument("--per-mood", action="store_true")

    nulls = sub.add_parser("nulls", help="retrieval confidence on queries with no answer")
    titles.set_defaults(func=cmd_titles)
    nulls.set_defaults(func=cmd_nulls)
    args = parser.parse_args()
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
