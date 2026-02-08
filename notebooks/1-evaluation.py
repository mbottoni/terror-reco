"""Evaluation Framework for TerrorReco Recommendations."""

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")

with app.setup:
    import asyncio
    import json
    import math
    import sys
    import time
    from pathlib import Path

    import marimo as mo
    import numpy as np

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")

    CACHE_DIR = PROJECT_ROOT / "notebooks" / "cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE = CACHE_DIR / "candidate_pools.json"

    mo.md("## 1 - Evaluation Framework for TerrorReco Recommendations")


@app.cell
def gold_test_set():
    TEST_SET = [
        {
            "mood": "slow-burn psychological dread",
            "gold": [
                "the witch", "hereditary", "the babadook", "it follows",
                "the lighthouse", "midsommar", "rosemary's baby", "the shining", "black swan",
            ],
        },
        {
            "mood": "campy fun with lots of blood",
            "gold": [
                "evil dead", "braindead", "dead alive", "re-animator",
                "the return of the living dead", "army of darkness",
                "tucker and dale vs evil", "shaun of the dead", "bad taste",
            ],
        },
        {
            "mood": "cosmic Lovecraftian isolation",
            "gold": [
                "the thing", "color out of space", "annihilation",
                "in the mouth of madness", "the void", "from beyond",
                "the mist", "event horizon", "underwater",
            ],
        },
        {
            "mood": "haunted house with dark secrets",
            "gold": [
                "the haunting", "the others", "the conjuring", "insidious",
                "poltergeist", "the amityville horror", "hill house",
                "the innocents", "the changeling",
            ],
        },
        {
            "mood": "slasher with suspense and a masked killer",
            "gold": [
                "scream", "halloween", "friday the 13th",
                "a nightmare on elm street", "the texas chain saw massacre",
                "black christmas", "you're next", "happy death day",
            ],
        },
        {
            "mood": "zombie apocalypse survival",
            "gold": [
                "dawn of the dead", "28 days later", "train to busan",
                "world war z", "night of the living dead",
                "zombieland", "cargo", "the girl with all the gifts",
            ],
        },
        {
            "mood": "demonic possession and exorcism",
            "gold": [
                "the exorcist", "the conjuring", "the rite",
                "the last exorcism", "insidious", "the nun",
                "evil dead", "deliver us from evil",
            ],
        },
        {
            "mood": "creepy kids and childhood fears",
            "gold": [
                "the omen", "children of the corn", "the ring", "orphan",
                "the sixth sense", "the others", "goodnight mommy", "the innocents",
            ],
        },
        {
            "mood": "found footage realistic terror",
            "gold": [
                "the blair witch project", "paranormal activity", "rec", "[rec]",
                "cloverfield", "as above, so below", "creep",
                "the last exorcism", "grave encounters",
            ],
        },
        {
            "mood": "survival horror isolated in nature",
            "gold": [
                "the descent", "the ritual", "backcountry", "eden lake",
                "the ruins", "wrong turn", "the forest", "crawl", "prey",
            ],
        },
        {
            "mood": "vampire gothic romance",
            "gold": [
                "interview with the vampire", "let the right one in",
                "only lovers left alive", "bram stoker's dracula", "nosferatu",
                "a girl walks home alone at night", "the hunger",
                "what we do in the shadows",
            ],
        },
        {
            "mood": "body horror and grotesque transformation",
            "gold": [
                "the fly", "the thing", "videodrome", "tusk", "society",
                "slither", "from beyond", "possessor", "titane",
            ],
        },
        {
            "mood": "eerie folk horror pagan rituals",
            "gold": [
                "the wicker man", "midsommar", "the witch", "apostle",
                "kill list", "a field in england", "the ritual", "hagazussa",
            ],
        },
        {
            "mood": "home invasion and paranoia",
            "gold": [
                "the strangers", "you're next", "funny games", "don't breathe",
                "hush", "inside", "the purge", "us",
            ],
        },
        {
            "mood": "sci-fi horror in space",
            "gold": [
                "alien", "aliens", "event horizon", "pandorum",
                "life", "sunshine", "the thing", "underwater",
            ],
        },
    ]

    mo.md(
        f"### Gold Test Set\n\n"
        f"**{len(TEST_SET)}** mood descriptions, each with curated gold titles."
    )
    return (TEST_SET,)


@app.cell
async def load_or_fetch_cache(TEST_SET):
    """Load cached candidate pools or fetch from OMDb if missing."""

    def _load_cache():
        if CACHE_FILE.exists():
            with open(CACHE_FILE) as f:
                return json.load(f)
        return None

    def _save_cache(data):
        with open(CACHE_FILE, "w") as f:
            json.dump(data, f, indent=2)

    cached = _load_cache()
    if cached and len(cached) >= len(TEST_SET):
        pools = cached
        mo.md(
            f"Loaded **{len(pools)}** cached candidate pools from disk.  \n"
            f"Delete `{CACHE_FILE.relative_to(PROJECT_ROOT)}` to re-fetch."
        )
    else:
        from app.services.recommender import recommend_movies_advanced

        pools = {}
        t0 = time.time()
        print(f"Fetching {len(TEST_SET)} candidate pools from OMDb API...")

        for idx, entry in enumerate(TEST_SET, 1):
            mood_str = entry["mood"]
            lap = time.time()
            pools[mood_str] = await recommend_movies_advanced(
                mood=mood_str, limit=60, pages=3, kind="movie", english_only=False,
            )
            n = len(pools[mood_str])
            print(f"  [{idx}/{len(TEST_SET)}] {mood_str} -> {n} movies ({time.time() - lap:.1f}s)")

        elapsed = time.time() - t0
        _save_cache(pools)

        total_movies = sum(len(v) for v in pools.values())
        print(f"Done in {elapsed:.1f}s. {total_movies} total movies cached.")
        mo.md(
            f"Fetched **{len(pools)}** pools in **{elapsed:.1f}s** "
            f"({total_movies} total movies). Saved to cache."
        )
    return (pools,)


@app.cell
def define_metrics():
    """Scoring functions for ranked recommendation lists."""

    def normalize_title(t):
        t = t.lower().strip()
        for prefix in ("the ", "a ", "an "):
            if t.startswith(prefix):
                t = t[len(prefix):]
        return t

    def title_match(candidate, gold_set):
        c = normalize_title(candidate)
        for g in gold_set:
            gn = normalize_title(g)
            if c == gn or c in gn or gn in c:
                return True
        return False

    def hit_rate_at_k(ranked_titles, gold, k=6):
        gold_set = set(gold)
        return 1.0 if any(title_match(t, gold_set) for t in ranked_titles[:k]) else 0.0

    def precision_at_k(ranked_titles, gold, k=6):
        gold_set = set(gold)
        return sum(1 for t in ranked_titles[:k] if title_match(t, gold_set)) / k

    def ndcg_at_k(ranked_titles, gold, k=6):
        gold_set = set(gold)
        dcg = sum(
            1.0 / math.log2(i + 2)
            for i, t in enumerate(ranked_titles[:k])
            if title_match(t, gold_set)
        )
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
        return dcg / idcg if idcg > 0 else 0.0

    def mrr(ranked_titles, gold):
        gold_set = set(gold)
        for i, t in enumerate(ranked_titles):
            if title_match(t, gold_set):
                return 1.0 / (i + 1)
        return 0.0

    def score_pipeline(ranked_titles, gold, k=6):
        return {
            "hit_rate@k": hit_rate_at_k(ranked_titles, gold, k),
            "precision@k": precision_at_k(ranked_titles, gold, k),
            "ndcg@k": ndcg_at_k(ranked_titles, gold, k),
            "mrr": mrr(ranked_titles, gold),
        }

    def evaluate_ranker(ranker_fn, pools_dict, test_set, k=6):
        all_scores = []
        for entry in test_set:
            mood, gold = entry["mood"], entry["gold"]
            items = pools_dict.get(mood, [])
            if not items:
                continue
            ranked = ranker_fn(mood, items)
            titles = [it.get("title", "") for it in ranked]
            all_scores.append(score_pipeline(titles, gold, k))
        if not all_scores:
            return {"hit_rate@k": 0, "precision@k": 0, "ndcg@k": 0, "mrr": 0}
        return {key: float(np.mean([s[key] for s in all_scores])) for key in all_scores[0]}

    mo.md(
        "### Evaluation Metrics\n\n"
        "Defined: **Hit Rate@K**, **Precision@K**, **NDCG@K**, **MRR**  \n"
        "Plus `evaluate_ranker()` to score any ranking function over the full test set."
    )
    return evaluate_ranker, score_pipeline


@app.cell
def baseline_eval(TEST_SET, evaluate_ranker, pools):
    """Measure the current unified recommender as a baseline."""
    from app.services.unified_recommender import recommend_unified_semantic

    def baseline_ranker(mood, items):
        return recommend_unified_semantic(mood=mood, items=items, limit=6)

    baseline_scores = evaluate_ranker(baseline_ranker, pools, TEST_SET, k=6)

    rows = "| Metric | Score |\n|--------|-------|\n"
    for metric, val in baseline_scores.items():
        rows += f"| {metric} | {val:.4f} |\n"

    mo.md(
        "### Baseline: Unified Recommender (default weights)\n\n"
        "Weights: semantic=0.45, keyword=0.20, popularity=0.20, recency=0.05  \n"
        "MMR lambda=0.7, limit=6\n\n" + rows
    )
    return (baseline_scores,)


@app.cell
def per_mood_breakdown(TEST_SET, pools, score_pipeline):
    def per_mood_breakdown(pools, TEST_SET):
        """Show per-mood evaluation for the baseline."""
        from app.services.unified_recommender import recommend_unified_semantic

        rows = "| Mood | Hit@6 | P@6 | NDCG@6 | MRR | Top-3 Titles |\n"
        rows += "|------|-------|-----|--------|-----|---------------|\n"

        for entry in TEST_SET:
            mood, gold = entry["mood"], entry["gold"]
            items = pools.get(mood, [])
            if not items:
                continue
            ranked = recommend_unified_semantic(mood=mood, items=items, limit=6)
            titles = [it.get("title", "") for it in ranked]
            scores = score_pipeline(titles, gold, k=6)
            top3 = ", ".join(titles[:3])
            rows += (
                f"| {mood[:40]} | {scores['hit_rate@k']:.2f} | "
                f"{scores['precision@k']:.2f} | {scores['ndcg@k']:.2f} | "
                f"{scores['mrr']:.2f} | {top3} |\n"
            )

        return mo.md("### Per-Mood Breakdown (Baseline)\n\n" + rows)

    per_mood_breakdown(pools, TEST_SET)
    return


@app.cell
def summary(baseline_scores):
    ndcg = baseline_scores.get("ndcg@k", 0)
    hr = baseline_scores.get("hit_rate@k", 0)
    m = baseline_scores.get("mrr", 0)
    mo.md(
        "### Summary\n\n"
        "This notebook provides:\n\n"
        "1. A **gold test set** of 15 horror mood descriptions with curated movie lists\n"
        "2. **Cached OMDb pools** so subsequent notebooks skip API calls\n"
        "3. **Metric functions** (`evaluate_ranker`) reusable in notebooks 2-4\n"
        "4. **Baseline scores** to beat:\n\n"
        f"   - NDCG@6 = **{ndcg:.4f}**\n"
        f"   - Hit Rate@6 = **{hr:.4f}**\n"
        f"   - MRR = **{m:.4f}**\n"
    )
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
