"""Cross-Encoder Reranking for TerrorReco Recommendations.

Cross-encoder models tested:
- cross-encoder/ms-marco-MiniLM-L-6-v2 (default, balanced)
- cross-encoder/ms-marco-TinyBERT-L-2-v2 (faster, lighter)

Depends on: notebooks/1-evaluation.py (cached pools + evaluation harness).
"""

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")

with app.setup:
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

    from app.services.corpus import load_corpus

    mo.md("## 3 - Cross-Encoder Reranking")


@app.cell
def load_data():
    """Load the horror movie corpus and gold test set."""
    corpus = load_corpus()

    if not corpus:
        mo.md(
            "**Error:** Corpus not found.  \n"
            "Run `marimo run notebooks/1-evaluation.py` first to build the corpus."
        )

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

    # Every mood uses the full corpus as its candidate pool
    pools = {_entry["mood"]: list(corpus) for _entry in TEST_SET}

    mo.md(
        f"Loaded **{len(corpus)}** horror movies from corpus, "
        f"**{len(TEST_SET)}** test moods."
    )
    return TEST_SET, pools


@app.cell
def eval_helpers():
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

    def mrr_score(ranked_titles, gold):
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
            "mrr": mrr_score(ranked_titles, gold),
        }

    return (score_pipeline,)


@app.cell
def ce_models():
    CROSS_ENCODERS = {
        "ms-marco-MiniLM-L-6-v2": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "ms-marco-TinyBERT-L-2-v2": "cross-encoder/ms-marco-TinyBERT-L-2-v2",
    }

    mo.md(
        "### Cross-Encoder Models\n\n"
        "| Name | HuggingFace ID | Notes |\n"
        "|------|---------------|-------|\n"
        "| ms-marco-MiniLM-L-6-v2 | cross-encoder/ms-marco-MiniLM-L-6-v2 | Default, balanced |\n"
        "| ms-marco-TinyBERT-L-2-v2 | cross-encoder/ms-marco-TinyBERT-L-2-v2 | Faster, lighter |\n"
    )
    return (CROSS_ENCODERS,)


@app.cell
def run_experiments(CROSS_ENCODERS, TEST_SET, pools, score_pipeline):
    from sentence_transformers import CrossEncoder, SentenceTransformer

    from app.services.unified_recommender import (
        _cosine,
        _facet_proxy,
        _minmax,
        _mmr,
        _normalize_text,
        _popularity,
    )

    bi_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    def _embed(texts):
        return np.asarray(bi_model.encode(texts, normalize_embeddings=True), dtype=np.float32)

    def _bi_encoder_rank(mood, items, limit=30):
        if not items:
            return []
        plots = [_normalize_text(mood)] + [_normalize_text(m.get("overview") or "") for m in items]
        embs = _embed(plots)
        mood_vec, plot_vecs = embs[0:1], embs[1:]
        sem = _minmax(_cosine(mood_vec, plot_vecs).ravel())
        kw = _minmax(np.array([_facet_proxy(mood, it) for it in items], dtype=np.float32))
        pop = _minmax(np.array([_popularity(it) for it in items], dtype=np.float32))

        rec = np.zeros(len(items), dtype=np.float32)
        years = []
        for it in items:
            y = it.get("year") or it.get("release_date") or ""
            try:
                y_int = int(str(y)[:4])
            except Exception:
                y_int = None
            years.append(y_int)
        valid = [y for y in years if isinstance(y, int)]
        if valid:
            y_arr = np.array(
                [y if isinstance(y, int) else min(valid) for y in years], dtype=np.int32,
            )
            rec = _minmax(y_arr.astype(np.float32))

        blended = (0.45 * sem + 0.20 * kw + 0.20 * pop + 0.05 * rec).astype(np.float32)
        order = np.argsort(-blended)[:limit]
        return [(items[i], float(blended[i])) for i in order]

    def bi_only_ranker(mood, items, limit=6):
        top = _bi_encoder_rank(mood, items, limit=30)
        pool = [it for it, _ in top]
        scores = np.array([s for _, s in top], dtype=np.float32)
        return _mmr(pool, sims=scores, k=limit, lambda_=0.7)

    def make_ce_ranker(ce_model_name, top_n=20):
        ce = CrossEncoder(ce_model_name)

        def ranker(mood, items, limit=6):
            top = _bi_encoder_rank(mood, items, limit=top_n)
            pool = [it for it, _ in top]
            if not pool:
                return []
            pairs = [(mood, _normalize_text(it.get("overview") or "")) for it in pool]
            ce_scores = np.array(ce.predict(pairs), dtype=np.float32)
            ce_order = np.argsort(-ce_scores)
            reranked = [pool[i] for i in ce_order]
            reranked_scores = ce_scores[ce_order]
            return _mmr(reranked, sims=reranked_scores, k=limit, lambda_=0.7)

        return ranker

    # Evaluate all configurations
    experiment_results = {}
    total_configs = 1 + len(CROSS_ENCODERS) * 4  # baseline + CE models x N values
    config_idx = 0

    # (a) Bi-encoder only
    config_idx += 1
    print(f"[{config_idx}/{total_configs}] Evaluating bi-encoder baseline...")
    all_sc = []
    t0 = time.time()
    for entry in TEST_SET:
        mood, gold = entry["mood"], entry["gold"]
        items = pools.get(mood, [])
        if not items:
            continue
        ranked = bi_only_ranker(mood, items)
        titles = [it.get("title", "") for it in ranked]
        all_sc.append(score_pipeline(titles, gold, k=6))
    bi_time = time.time() - t0
    experiment_results["bi-encoder only"] = {
        k: float(np.mean([s[k] for s in all_sc])) for k in all_sc[0]
    }
    experiment_results["bi-encoder only"]["latency_s"] = bi_time
    print(f"  NDCG@6: {experiment_results['bi-encoder only']['ndcg@k']:.4f} ({bi_time:.1f}s)")

    # (b) Cross-encoder reranking with different N values
    for ce_name, ce_hf in CROSS_ENCODERS.items():
        print(f"Loading cross-encoder: {ce_name}...")
        for top_n in [10, 15, 20, 30]:
            config_idx += 1
            label = f"{ce_name} (N={top_n})"
            print(f"  [{config_idx}/{total_configs}] {label}...", end=" ")
            ranker = make_ce_ranker(ce_hf, top_n=top_n)
            all_sc = []
            t0 = time.time()
            for entry in TEST_SET:
                mood, gold = entry["mood"], entry["gold"]
                items = pools.get(mood, [])
                if not items:
                    continue
                ranked = ranker(mood, items)
                titles = [it.get("title", "") for it in ranked]
                all_sc.append(score_pipeline(titles, gold, k=6))
            elapsed = time.time() - t0
            experiment_results[label] = {
                k: float(np.mean([s[k] for s in all_sc])) for k in all_sc[0]
            }
            experiment_results[label]["latency_s"] = elapsed
            print(f"NDCG@6: {experiment_results[label]['ndcg@k']:.4f} ({elapsed:.1f}s)")

    print("Done.")
    mo.md(f"Evaluated **{len(experiment_results)}** configurations.")
    return (experiment_results,)


@app.cell
def results_table(experiment_results):
    _rows = "| Configuration | Hit@6 | P@6 | NDCG@6 | MRR | Time (s) |\n"
    _rows += "|--------------|-------|-----|--------|-----|----------|\n"

    _best_ndcg = max(_r["ndcg@k"] for _r in experiment_results.values())

    for _label, _scores in experiment_results.items():
        _marker = " **best**" if _scores["ndcg@k"] == _best_ndcg else ""
        _rows += (
            f"| {_label}{_marker} | {_scores['hit_rate@k']:.4f} | "
            f"{_scores['precision@k']:.4f} | {_scores['ndcg@k']:.4f} | "
            f"{_scores['mrr']:.4f} | {_scores['latency_s']:.1f} |\n"
        )

    mo.md("### Results: Bi-Encoder vs Cross-Encoder Reranking\n\n" + _rows)
    return


@app.cell
def ndcg_by_depth(experiment_results):
    """Show how NDCG changes with rerank depth."""
    _lines = ["### NDCG@6 by Rerank Depth\n", "```"]
    _max_bar = 40

    _bi_ndcg = experiment_results.get("bi-encoder only", {}).get("ndcg@k", 0)
    _lines.append(f"  {'bi-encoder only':40s} | {'#' * 20} {_bi_ndcg:.4f} (baseline)")

    for _label, _scores in sorted(experiment_results.items()):
        if _label == "bi-encoder only":
            continue
        _ndcg = _scores["ndcg@k"]
        _bar_len = int((_ndcg / max(_bi_ndcg, _ndcg, 0.001)) * _max_bar)
        _delta = _ndcg - _bi_ndcg
        _sign = "+" if _delta >= 0 else ""
        _lines.append(f"  {_label:40s} | {'#' * _bar_len} {_ndcg:.4f} ({_sign}{_delta:.4f})")

    _lines.append("```")
    mo.md("\n".join(_lines))
    return


@app.cell
def latency_chart(experiment_results):
    _lines = ["### Latency Comparison (full test set)\n", "```"]
    _max_bar = 40
    _max_lat = max(_r["latency_s"] for _r in experiment_results.values()) or 1

    for _label, _scores in sorted(experiment_results.items(), key=lambda x: x[1]["latency_s"]):
        _bar_len = int((_scores["latency_s"] / _max_lat) * _max_bar)
        _lines.append(f"  {_label:40s} | {'#' * _bar_len} {_scores['latency_s']:.1f}s")

    _lines.append("```")
    mo.md("\n".join(_lines))
    return


@app.cell
def recommendation(experiment_results):
    best = max(experiment_results.items(), key=lambda x: x[1]["ndcg@k"])
    bi_ndcg = experiment_results.get("bi-encoder only", {}).get("ndcg@k", 0)
    improvement = best[1]["ndcg@k"] - bi_ndcg

    if improvement > 0.01:
        advice = (
            f"Cross-encoder reranking **improves** NDCG@6 by {improvement:.4f}.  \n"
            f"Best config: `{best[0]}` (NDCG@6 = {best[1]['ndcg@k']:.4f}).  \n\n"
            "To integrate: wire up the cross-encoder path in "
            "`app/services/unified_recommender.py`, reranking the top-N "
            "candidates before applying MMR."
        )
    else:
        advice = (
            "Cross-encoder reranking shows **minimal improvement** over the bi-encoder alone.  \n"
            "The added latency may not be worth it for this dataset.  \n"
            "Consider revisiting if the candidate pool size or evaluation set grows."
        )

    mo.md("### Recommendation\n\n" + advice)
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
