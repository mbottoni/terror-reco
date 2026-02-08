"""Embedding Model Comparison for TerrorReco Recommendations.

Models compared:
- all-MiniLM-L6-v2 (22M params, fast)
- all-mpnet-base-v2 (109M params, current default)
- BAAI/bge-small-en-v1.5 (33M params)
- BAAI/bge-base-en-v1.5 (109M params)

Depends on: a built corpus (run notebooks/1-evaluation.py first).
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

    mo.md("## 2 - Embedding Model Comparison")


@app.cell
def load_data():
    """Load the horror movie corpus and gold test set."""
    from app.services.corpus import load_corpus

    corpus = load_corpus()

    if not corpus:
        mo.md(
            "**Error:** Corpus not found.  \n"
            "Run `marimo run notebooks/1-evaluation.py` first to build it."
        )

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

    # Every mood uses the full corpus as its candidate pool
    pools = {_entry["mood"]: list(corpus) for _entry in TEST_SET}

    mo.md(
        f"Loaded **{len(corpus)}** horror movies from corpus, "
        f"**{len(TEST_SET)}** test moods.  \n"
        f"Each mood uses the full corpus as candidates."
    )
    return TEST_SET, pools


@app.cell
def model_defs():
    MODELS = {
        "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
        "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
        "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
        "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
    }

    mo.md(
        "### Models to Compare\n\n"
        "| Short Name | HuggingFace ID | Notes |\n"
        "|------------|---------------|-------|\n"
        "| all-MiniLM-L6-v2 | sentence-transformers/all-MiniLM-L6-v2 | 22M, very fast |\n"
        "| all-mpnet-base-v2 | sentence-transformers/all-mpnet-base-v2 | 109M, current default |\n"
        "| bge-small-en-v1.5 | BAAI/bge-small-en-v1.5 | 33M, strong retrieval |\n"
        "| bge-base-en-v1.5 | BAAI/bge-base-en-v1.5 | 109M, top retrieval |\n"
    )
    return (MODELS,)


@app.cell
def eval_helpers():
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
def run_comparison(MODELS, TEST_SET, pools, score_pipeline):
    from sentence_transformers import SentenceTransformer

    from app.services.unified_recommender import (
        _cosine,
        _facet_proxy,
        _minmax,
        _mmr,
        _normalize_text,
        _popularity,
    )

    def _make_ranker(model_name):
        model = SentenceTransformer(model_name)

        def _embed(texts):
            return np.asarray(
                model.encode(texts, normalize_embeddings=True),
                dtype=np.float32,
            )

        def ranker(mood, items, limit=6):
            if not items:
                return []
            plots = [_normalize_text(mood)] + [
                _normalize_text(m.get("overview") or "") for m in items
            ]
            embs = _embed(plots)
            mood_vec, plot_vecs = embs[0:1], embs[1:]
            sem = _minmax(_cosine(mood_vec, plot_vecs).ravel())
            kw = _minmax(
                np.array(
                    [_facet_proxy(mood, it) for it in items],
                    dtype=np.float32,
                )
            )
            pop = _minmax(
                np.array(
                    [_popularity(it) for it in items],
                    dtype=np.float32,
                )
            )

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
                    [y if isinstance(y, int) else min(valid) for y in years],
                    dtype=np.int32,
                )
                rec = _minmax(y_arr.astype(np.float32))

            blended = (0.45 * sem + 0.20 * kw + 0.20 * pop + 0.05 * rec).astype(np.float32)
            order = np.argsort(-blended)
            pool_idx = order[: max(10, limit * 5)]
            pool = [items[i] for i in pool_idx]
            pool_scores = blended[pool_idx]
            return _mmr(pool, sims=pool_scores, k=limit, lambda_=0.7)

        return ranker, _embed

    results = {}
    latencies = {}

    print(f"Comparing {len(MODELS)} embedding models...")
    for _model_idx, (short_name, hf_id) in enumerate(MODELS.items(), 1):
        print(f"  [{_model_idx}/{len(MODELS)}] Loading {short_name}...")
        ranker, embed_fn = _make_ranker(hf_id)

        # Measure embedding latency on 50 sample texts
        sample_texts = []
        for _mk, _items in pools.items():
            for _it in _items[:5]:
                sample_texts.append(_it.get("overview") or "")
            if len(sample_texts) >= 50:
                break
        sample_texts = sample_texts[:50]

        _t0 = time.time()
        embed_fn(sample_texts)
        latencies[short_name] = (time.time() - _t0) * 1000
        print(f"           Latency: {latencies[short_name]:.0f}ms/50 texts")

        # Evaluate
        _all_scores = []
        for _entry in TEST_SET:
            _mood, _gold = _entry["mood"], _entry["gold"]
            _items = pools.get(_mood, [])
            if not _items:
                continue
            _ranked = ranker(_mood, _items)
            _titles = [_it.get("title", "") for _it in _ranked]
            _all_scores.append(score_pipeline(_titles, _gold, k=6))

        _avg = {key: float(np.mean([s[key] for s in _all_scores])) for key in _all_scores[0]}
        _avg["latency_ms"] = latencies[short_name]
        results[short_name] = _avg
        print(f"           NDCG@6: {_avg['ndcg@k']:.4f}, " f"Hit@6: {_avg['hit_rate@k']:.4f}")

    print("Done.")
    mo.md(f"Evaluated **{len(results)}** models.")
    return SentenceTransformer, results


@app.cell
def results_table(results):
    rows = (
        "| Model | Hit Rate@6 | Precision@6 | NDCG@6 | MRR "
        "| Latency (ms/50) |\n"
        "|-------|-----------|-------------|--------|-----"
        "|----------------|\n"
    )

    best_ndcg = max(r["ndcg@k"] for r in results.values())

    for name, scores in results.items():
        marker = " **best**" if scores["ndcg@k"] == best_ndcg else ""
        rows += (
            f"| {name}{marker} | {scores['hit_rate@k']:.4f} | "
            f"{scores['precision@k']:.4f} | {scores['ndcg@k']:.4f} | "
            f"{scores['mrr']:.4f} | {scores['latency_ms']:.0f} |\n"
        )

    mo.md("### Results\n\n" + rows)
    return


@app.cell
def bar_chart(results):
    """Render a text-based comparison."""
    viz_lines = ["### NDCG@6 by Model\n", "```"]
    _max_bar = 40
    _max_ndcg = max(r["ndcg@k"] for r in results.values()) or 1

    for _n, _s in sorted(results.items(), key=lambda x: -x[1]["ndcg@k"]):
        _bl = int((_s["ndcg@k"] / _max_ndcg) * _max_bar)
        viz_lines.append(f"  {_n:25s} | {'#' * _bl} {_s['ndcg@k']:.4f}")

    viz_lines.append("```\n")
    viz_lines.append("### Hit Rate@6 by Model\n")
    viz_lines.append("```")
    _max_hr = max(r["hit_rate@k"] for r in results.values()) or 1

    for _n, _s in sorted(results.items(), key=lambda x: -x[1]["hit_rate@k"]):
        _bl = int((_s["hit_rate@k"] / _max_hr) * _max_bar)
        viz_lines.append(f"  {_n:25s} | {'#' * _bl} {_s['hit_rate@k']:.4f}")

    viz_lines.append("```\n")
    viz_lines.append("### Latency (ms per 50 texts)\n")
    viz_lines.append("```")
    _max_lat = max(r["latency_ms"] for r in results.values()) or 1

    for _n, _s in sorted(results.items(), key=lambda x: x[1]["latency_ms"]):
        _bl = int((_s["latency_ms"] / _max_lat) * _max_bar)
        viz_lines.append(f"  {_n:25s} | {'#' * _bl} {_s['latency_ms']:.0f}ms")

    viz_lines.append("```")
    mo.md("\n".join(viz_lines))
    return


@app.cell
def heatmap(MODELS, SentenceTransformer, TEST_SET, pools):
    """Cosine similarity between a sample mood and top-10 movies."""
    _sample_mood = TEST_SET[0]["mood"]
    _items = pools.get(_sample_mood, [])[:10]

    if not _items:
        mo.md("No items to show heatmap.")
    else:
        _titles = [_it.get("title", "?")[:30] for _it in _items]
        _lines = [
            f"### Cosine Similarity: '{_sample_mood}' vs top-10\n",
        ]
        _lines.append("```")
        _lines.append(f"{'Movie':<32s} | " + " | ".join(f"{_n[:8]:>8s}" for _n in MODELS))
        _lines.append("-" * (34 + 11 * len(MODELS)))

        _sim_by_model = {}
        for _sn, _hf in MODELS.items():
            print(f"  Heatmap: encoding with {_sn}...")
            _m = SentenceTransformer(_hf)
            _texts = [_sample_mood] + [_it.get("overview") or "" for _it in _items]
            _embs = np.asarray(
                _m.encode(_texts, normalize_embeddings=True),
                dtype=np.float32,
            )
            _sims = (_embs[0:1] @ _embs[1:].T).ravel()
            _sim_by_model[_sn] = _sims

        for _i, _title in enumerate(_titles):
            _vals = " | ".join(f"{_sim_by_model[_n][_i]:8.4f}" for _n in MODELS)
            _lines.append(f"{_title:<32s} | {_vals}")

        _lines.append("```")
        mo.md("\n".join(_lines))
    return


@app.cell
def recommendation(results):
    best_model = max(results.items(), key=lambda x: x[1]["ndcg@k"])
    fastest = min(results.items(), key=lambda x: x[1]["latency_ms"])

    mo.md(
        "### Recommendation\n\n"
        f"**Best quality:** `{best_model[0]}` with "
        f"NDCG@6 = {best_model[1]['ndcg@k']:.4f}  \n"
        f"**Fastest:** `{fastest[0]}` at "
        f"{fastest[1]['latency_ms']:.0f}ms  \n\n"
        "To apply: update `_MODEL_NAME` in "
        "`app/services/unified_recommender.py`."
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
