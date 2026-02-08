"""Weight Optimization for TerrorReco Unified Recommender.

Grid search over blending weights (semantic, keyword, popularity, recency)
and MMR lambda parameter to find the combination that maximizes NDCG@6.

Depends on: notebooks/1-evaluation.py (cached pools + evaluation harness).
"""

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")

with app.setup:
    import itertools
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

    mo.md("## 4 - Weight Optimization")


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
        for _prefix in ("the ", "a ", "an "):
            if t.startswith(_prefix):
                t = t[len(_prefix):]
        return t

    def title_match(candidate, gold_set):
        _c = normalize_title(candidate)
        for _g in gold_set:
            _gn = normalize_title(_g)
            if _c == _gn or _c in _gn or _gn in _c:
                return True
        return False

    def ndcg_at_k(ranked_titles, gold, k=6):
        _gold_set = set(gold)
        _dcg = sum(
            1.0 / math.log2(_i + 2)
            for _i, _t in enumerate(ranked_titles[:k])
            if title_match(_t, _gold_set)
        )
        _idcg = sum(1.0 / math.log2(_i + 2) for _i in range(min(len(gold), k)))
        return _dcg / _idcg if _idcg > 0 else 0.0

    def hit_rate_at_k(ranked_titles, gold, k=6):
        _gold_set = set(gold)
        return 1.0 if any(title_match(_t, _gold_set) for _t in ranked_titles[:k]) else 0.0

    def precision_at_k(ranked_titles, gold, k=6):
        _gold_set = set(gold)
        return sum(1 for _t in ranked_titles[:k] if title_match(_t, _gold_set)) / k

    def mrr_score(ranked_titles, gold):
        _gold_set = set(gold)
        for _i, _t in enumerate(ranked_titles):
            if title_match(_t, _gold_set):
                return 1.0 / (_i + 1)
        return 0.0

    def score_pipeline(ranked_titles, gold, k=6):
        return {
            "hit_rate@k": hit_rate_at_k(ranked_titles, gold, k),
            "precision@k": precision_at_k(ranked_titles, gold, k),
            "ndcg@k": ndcg_at_k(ranked_titles, gold, k),
            "mrr": mrr_score(ranked_titles, gold),
        }

    return ndcg_at_k, score_pipeline


@app.cell
def precompute_signals(TEST_SET, pools):
    """Precompute all four signal arrays per mood so the grid search
    only varies the weight combination and MMR lambda."""
    from sentence_transformers import SentenceTransformer

    from app.services.unified_recommender import (
        _cosine,
        _facet_proxy,
        _minmax,
        _normalize_text,
        _popularity,
    )

    _model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    def _embed(texts):
        return np.asarray(_model.encode(texts, normalize_embeddings=True), dtype=np.float32)

    precomputed = {}

    print(f"Pre-computing signals for {len(TEST_SET)} moods...")
    for _idx, _entry in enumerate(TEST_SET, 1):
        _mood = _entry["mood"]
        _items = pools.get(_mood, [])
        if not _items:
            print(f"  [{_idx}/{len(TEST_SET)}] {_mood[:40]} -> SKIP (no items)")
            continue

        _plots = [_normalize_text(_mood)] + [
            _normalize_text(_m.get("overview") or "") for _m in _items
        ]
        _embs = _embed(_plots)
        _mood_vec, _plot_vecs = _embs[0:1], _embs[1:]
        _sem = _minmax(_cosine(_mood_vec, _plot_vecs).ravel())
        _kw = _minmax(np.array([_facet_proxy(_mood, _it) for _it in _items], dtype=np.float32))
        _pop = _minmax(np.array([_popularity(_it) for _it in _items], dtype=np.float32))

        _rec = np.zeros(len(_items), dtype=np.float32)
        _years = []
        for _it in _items:
            _y = _it.get("year") or _it.get("release_date") or ""
            try:
                _y_int = int(str(_y)[:4])
            except Exception:
                _y_int = None
            _years.append(_y_int)
        _valid = [_y for _y in _years if isinstance(_y, int)]
        if _valid:
            _y_arr = np.array(
                [_y if isinstance(_y, int) else min(_valid) for _y in _years], dtype=np.int32,
            )
            _rec = _minmax(_y_arr.astype(np.float32))

        precomputed[_mood] = {"items": _items, "sem": _sem, "kw": _kw, "pop": _pop, "rec": _rec}
        print(f"  [{_idx}/{len(TEST_SET)}] {_mood[:40]} -> {len(_items)} items")

    print("Done.")
    mo.md(f"Pre-computed signals for **{len(precomputed)}** moods.")
    return (precomputed,)


@app.cell
def grid_search(TEST_SET, ndcg_at_k, precomputed):
    from app.services.unified_recommender import _mmr

    _sem_vals = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    _kw_vals = [0.05, 0.10, 0.15, 0.20, 0.25]
    _pop_vals = [0.05, 0.10, 0.15, 0.20, 0.25]
    _rec_vals = [0.00, 0.05, 0.10]
    _lambda_vals = [0.5, 0.6, 0.7, 0.8, 0.9]

    _combos = []
    for _s, _k, _p, _r in itertools.product(_sem_vals, _kw_vals, _pop_vals, _rec_vals):
        _total = _s + _k + _p + _r
        if abs(_total - 1.0) < 0.001:
            _combos.append((_s, _k, _p, _r))

    _total_evals = len(_combos) * len(_lambda_vals)
    mo.md(
        f"**{len(_combos)}** valid weight combos x **{len(_lambda_vals)}** lambda values "
        f"= **{_total_evals}** configurations to evaluate."
    )

    print(f"Running grid search: {_total_evals} configurations...")
    _t0 = time.time()
    grid_results = []
    _done = 0

    for _w_sem, _w_kw, _w_pop, _w_rec in _combos:
        for _lam in _lambda_vals:
            _mood_ndcgs = []
            for _entry in TEST_SET:
                _mood = _entry["mood"]
                _gold = _entry["gold"]
                _data = precomputed.get(_mood)
                if _data is None:
                    continue
                _items = _data["items"]
                _blended = (
                    _w_sem * _data["sem"]
                    + _w_kw * _data["kw"]
                    + _w_pop * _data["pop"]
                    + _w_rec * _data["rec"]
                ).astype(np.float32)

                _order = np.argsort(-_blended)
                _pool_idx = _order[: max(10, 6 * 5)]
                _pool = [_items[_i] for _i in _pool_idx]
                _pool_scores = _blended[_pool_idx]
                _ranked = _mmr(_pool, sims=_pool_scores, k=6, lambda_=_lam)
                _titles = [_it.get("title", "") for _it in _ranked]
                _mood_ndcgs.append(ndcg_at_k(_titles, _gold, k=6))

            _avg_ndcg = float(np.mean(_mood_ndcgs)) if _mood_ndcgs else 0.0
            grid_results.append({
                "sem": _w_sem, "kw": _w_kw, "pop": _w_pop, "rec": _w_rec,
                "lambda": _lam, "ndcg@6": _avg_ndcg,
            })
            _done += 1
            if _done % 50 == 0 or _done == _total_evals:
                print(f"  {_done}/{_total_evals} ({_done * 100 // _total_evals}%)")

    _elapsed = time.time() - _t0
    grid_results.sort(key=lambda x: -x["ndcg@6"])
    print(f"Done in {_elapsed:.1f}s. Best NDCG@6: {grid_results[0]['ndcg@6']:.4f}")

    mo.md(
        f"Grid search completed in **{_elapsed:.1f}s**.  \n"
        f"Evaluated **{len(grid_results)}** configurations."
    )
    return (grid_results,)


@app.cell
def top_configs(grid_results):
    _rows = (
        "| Rank | Semantic | Keyword | Popularity | Recency | Lambda | NDCG@6 |\n"
        "|------|----------|---------|------------|---------|--------|--------|\n"
    )
    for _i, _r in enumerate(grid_results[:20]):
        _rows += (
            f"| {_i + 1} | {_r['sem']:.2f} | {_r['kw']:.2f} | "
            f"{_r['pop']:.2f} | {_r['rec']:.2f} | {_r['lambda']:.1f} | "
            f"{_r['ndcg@6']:.4f} |\n"
        )

    mo.md("### Top 20 Weight Configurations (by NDCG@6)\n\n" + _rows)
    return


@app.cell
def compare_best_vs_baseline(
    TEST_SET,
    grid_results,
    precomputed,
    score_pipeline,
):
    from app.services.unified_recommender import _mmr

    _best = grid_results[0]
    _baseline_w = {"sem": 0.45, "kw": 0.20, "pop": 0.20, "rec": 0.05, "lambda": 0.7}

    _configs = {"Baseline (0.45/0.20/0.20/0.05, lam=0.7)": _baseline_w, "Optimized": _best}
    final_results = {}

    for _label, _cfg in _configs.items():
        _all_scores = []
        for _entry in TEST_SET:
            _mood, _gold = _entry["mood"], _entry["gold"]
            _data = precomputed.get(_mood)
            if _data is None:
                continue
            _items = _data["items"]
            _blended = (
                _cfg["sem"] * _data["sem"]
                + _cfg["kw"] * _data["kw"]
                + _cfg["pop"] * _data["pop"]
                + _cfg["rec"] * _data["rec"]
            ).astype(np.float32)

            _order = np.argsort(-_blended)
            _pool_idx = _order[: max(10, 6 * 5)]
            _pool = [_items[_i] for _i in _pool_idx]
            _pool_scores = _blended[_pool_idx]
            _ranked = _mmr(_pool, sims=_pool_scores, k=6, lambda_=_cfg["lambda"])
            _titles = [_it.get("title", "") for _it in _ranked]
            _all_scores.append(score_pipeline(_titles, _gold, k=6))

        _avg = {_k: float(np.mean([_s[_k] for _s in _all_scores])) for _k in _all_scores[0]}
        final_results[_label] = _avg

    _rows = "| Config | Hit@6 | P@6 | NDCG@6 | MRR |\n"
    _rows += "|--------|-------|-----|--------|-----|\n"
    for _label, _scores in final_results.items():
        _rows += (
            f"| {_label} | {_scores['hit_rate@k']:.4f} | "
            f"{_scores['precision@k']:.4f} | {_scores['ndcg@k']:.4f} | "
            f"{_scores['mrr']:.4f} |\n"
        )

    _improvement = final_results["Optimized"]["ndcg@k"] - final_results.get(
        "Baseline (0.45/0.20/0.20/0.05, lam=0.7)", {}
    ).get("ndcg@k", 0)

    mo.md(
        "### Baseline vs Optimized\n\n" + _rows
        + f"\nNDCG@6 improvement: **{_improvement:+.4f}**"
    )
    return (final_results,)


@app.cell
def sensitivity(grid_results):
    """Show how NDCG varies as each weight dimension changes."""
    import collections

    _lines = ["### Weight Sensitivity (average NDCG@6 per weight value)\n"]

    for _dim_name, _dim_key in [
        ("Semantic", "sem"),
        ("Keyword", "kw"),
        ("Popularity", "pop"),
        ("Recency", "rec"),
        ("Lambda", "lambda"),
    ]:
        _buckets = collections.defaultdict(list)
        for _r in grid_results:
            _buckets[_r[_dim_key]].append(_r["ndcg@6"])

        _lines.append(f"\n**{_dim_name}:**\n")
        _lines.append("```")
        _max_bar = 30
        _max_val = max(float(np.mean(_v)) for _v in _buckets.values()) or 1

        for _val in sorted(_buckets):
            _avg = float(np.mean(_buckets[_val]))
            _bar_len = int((_avg / _max_val) * _max_bar)
            _lines.append(f"  {_val:5.2f} | {'#' * _bar_len} {_avg:.4f}")

        _lines.append("```")

    mo.md("\n".join(_lines))
    return


@app.cell
def recommendation(final_results, grid_results):
    _best = grid_results[0]
    _baseline_ndcg = final_results.get(
        "Baseline (0.45/0.20/0.20/0.05, lam=0.7)", {}
    ).get("ndcg@k", 0)
    _improvement = _best["ndcg@6"] - _baseline_ndcg

    mo.md(
        "### Recommended Configuration\n\n"
        f"```python\n"
        f"# Optimal weights (NDCG@6 = {_best['ndcg@6']:.4f}, improvement: {_improvement:+.4f})\n"
        f"weights = {{\n"
        f'    "semantic": {_best["sem"]:.2f},\n'
        f'    "keyword": {_best["kw"]:.2f},\n'
        f'    "popularity": {_best["pop"]:.2f},\n'
        f'    "recency": {_best["rec"]:.2f},\n'
        f"}}\n"
        f"diversity_lambda = {_best['lambda']:.1f}\n"
        f"```\n\n"
        "**To apply:** Update the default `w` dictionary in "
        "`recommend_unified_semantic()` in `app/services/unified_recommender.py`, "
        "and set `UNIFIED_DIVERSITY_LAMBDA` in `.env` or `app/settings.py`."
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
