"""Evaluation Framework for TerrorReco Recommendations."""

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")

with app.setup:
    import math
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")

    mo.md("## 1 - Evaluation Framework for TerrorReco Recommendations")


@app.cell
def gold_test_set():
    TEST_SET = [
        {
            "mood": "slow-burn psychological dread",
            "gold": [
                "the witch",
                "hereditary",
                "the babadook",
                "it follows",
                "the lighthouse",
                "midsommar",
                "rosemary's baby",
                "the shining",
                "black swan",
            ],
        },
        {
            "mood": "campy fun with lots of blood",
            "gold": [
                "evil dead",
                "braindead",
                "dead alive",
                "re-animator",
                "the return of the living dead",
                "army of darkness",
                "tucker and dale vs evil",
                "shaun of the dead",
                "bad taste",
            ],
        },
        {
            "mood": "cosmic Lovecraftian isolation",
            "gold": [
                "the thing",
                "color out of space",
                "annihilation",
                "in the mouth of madness",
                "the void",
                "from beyond",
                "the mist",
                "event horizon",
                "underwater",
            ],
        },
        {
            "mood": "haunted house with dark secrets",
            "gold": [
                "the haunting",
                "the others",
                "the conjuring",
                "insidious",
                "poltergeist",
                "the amityville horror",
                "hill house",
                "the innocents",
                "the changeling",
            ],
        },
        {
            "mood": "slasher with suspense and a masked killer",
            "gold": [
                "scream",
                "halloween",
                "friday the 13th",
                "a nightmare on elm street",
                "the texas chain saw massacre",
                "black christmas",
                "you're next",
                "happy death day",
            ],
        },
        {
            "mood": "zombie apocalypse survival",
            "gold": [
                "dawn of the dead",
                "28 days later",
                "train to busan",
                "world war z",
                "night of the living dead",
                "zombieland",
                "cargo",
                "the girl with all the gifts",
            ],
        },
        {
            "mood": "demonic possession and exorcism",
            "gold": [
                "the exorcist",
                "the conjuring",
                "the rite",
                "the last exorcism",
                "insidious",
                "the nun",
                "evil dead",
                "deliver us from evil",
            ],
        },
        {
            "mood": "creepy kids and childhood fears",
            "gold": [
                "the omen",
                "children of the corn",
                "the ring",
                "orphan",
                "the sixth sense",
                "the others",
                "goodnight mommy",
                "the innocents",
            ],
        },
        {
            "mood": "found footage realistic terror",
            "gold": [
                "the blair witch project",
                "paranormal activity",
                "rec",
                "[rec]",
                "cloverfield",
                "as above, so below",
                "creep",
                "the last exorcism",
                "grave encounters",
            ],
        },
        {
            "mood": "survival horror isolated in nature",
            "gold": [
                "the descent",
                "the ritual",
                "backcountry",
                "eden lake",
                "the ruins",
                "wrong turn",
                "the forest",
                "crawl",
                "prey",
            ],
        },
        {
            "mood": "vampire gothic romance",
            "gold": [
                "interview with the vampire",
                "let the right one in",
                "only lovers left alive",
                "bram stoker's dracula",
                "nosferatu",
                "a girl walks home alone at night",
                "the hunger",
                "what we do in the shadows",
            ],
        },
        {
            "mood": "body horror and grotesque transformation",
            "gold": [
                "the fly",
                "the thing",
                "videodrome",
                "tusk",
                "society",
                "slither",
                "from beyond",
                "possessor",
                "titane",
            ],
        },
        {
            "mood": "eerie folk horror pagan rituals",
            "gold": [
                "the wicker man",
                "midsommar",
                "the witch",
                "apostle",
                "kill list",
                "a field in england",
                "the ritual",
                "hagazussa",
            ],
        },
        {
            "mood": "home invasion and paranoia",
            "gold": [
                "the strangers",
                "you're next",
                "funny games",
                "don't breathe",
                "hush",
                "inside",
                "the purge",
                "us",
            ],
        },
        {
            "mood": "sci-fi horror in space",
            "gold": [
                "alien",
                "aliens",
                "event horizon",
                "pandorum",
                "life",
                "sunshine",
                "the thing",
                "underwater",
            ],
        },
    ]

    mo.md(
        f"### Gold Test Set\n\n"
        f"**{len(TEST_SET)}** mood descriptions, each with curated gold titles."
    )
    return (TEST_SET,)


@app.cell
async def load_corpus_cell():
    """Load (or build) the horror movie corpus.

    The corpus is a broad, pre-built set of horror movies from OMDb.
    All matching is done at query time via sentence-transformer
    embeddings -- no hardcoded mood-to-movie mappings.
    """
    from app.services.corpus import (
        build_corpus,
        get_corpus_embeddings,
        load_corpus,
    )

    corpus = load_corpus()
    if not corpus:
        print("Corpus not found. Building from OMDb (first run)...")
        corpus = await build_corpus(pages=2)

    corpus_embeddings = get_corpus_embeddings(corpus)

    mo.md(
        f"### Horror Movie Corpus\n\n"
        f"**{len(corpus)}** horror movies loaded.  \n"
        f"Embeddings shape: `{corpus_embeddings.shape}`.  \n\n"
        f"All moods are ranked against this shared corpus via semantic search."
    )
    return corpus, corpus_embeddings


@app.cell
def define_metrics():
    """Scoring functions for ranked recommendation lists."""

    def normalize_title(t):
        t = t.lower().strip()
        for prefix in ("the ", "a ", "an "):
            if t.startswith(prefix):
                t = t[len(prefix) :]
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
def baseline_eval(TEST_SET, corpus, corpus_embeddings, evaluate_ranker):
    """Measure the current unified recommender as a baseline.

    Uses the full corpus as the candidate pool for every mood.
    """
    from app.services.corpus import semantic_search
    from app.services.unified_recommender import recommend_unified_semantic

    def baseline_ranker(mood, items):
        return recommend_unified_semantic(mood=mood, items=items, limit=6)

    # Build per-mood pools via semantic search over the corpus
    _pools = {}
    for _entry in TEST_SET:
        _mood = _entry["mood"]
        _candidates = semantic_search(
            _mood,
            corpus,
            corpus_embeddings,
            top_k=60,
        )
        _pools[_mood] = [{k: v for k, v in m.items() if not k.startswith("_")} for m in _candidates]

    baseline_scores = evaluate_ranker(baseline_ranker, _pools, TEST_SET, k=6)

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
def per_mood_breakdown(TEST_SET, corpus, corpus_embeddings, score_pipeline):
    def _per_mood_breakdown():
        """Show per-mood evaluation for the baseline."""
        from app.services.corpus import semantic_search
        from app.services.unified_recommender import recommend_unified_semantic

        rows = "| Mood | Hit@6 | P@6 | NDCG@6 | MRR | Top-3 Titles |\n"
        rows += "|------|-------|-----|--------|-----|---------------|\n"

        for _entry in TEST_SET:
            _mood, _gold = _entry["mood"], _entry["gold"]
            _candidates = semantic_search(
                _mood,
                corpus,
                corpus_embeddings,
                top_k=60,
            )
            _items = [{k: v for k, v in m.items() if not k.startswith("_")} for m in _candidates]
            if not _items:
                continue
            _ranked = recommend_unified_semantic(
                mood=_mood,
                items=_items,
                limit=6,
            )
            _titles = [it.get("title", "") for it in _ranked]
            _scores = score_pipeline(_titles, _gold, k=6)
            _top3 = ", ".join(_titles[:3])
            rows += (
                f"| {_mood[:40]} | {_scores['hit_rate@k']:.2f} | "
                f"{_scores['precision@k']:.2f} | {_scores['ndcg@k']:.2f} | "
                f"{_scores['mrr']:.2f} | {_top3} |\n"
            )

        return mo.md("### Per-Mood Breakdown (Baseline)\n\n" + rows)

    _per_mood_breakdown()
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
        "2. A **corpus-based pipeline** using sentence-transformer embeddings\n"
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
