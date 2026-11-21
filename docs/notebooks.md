# Research Notebooks

TerrorReco includes a set of interactive notebooks from the early evaluation and tuning of the recommendation engine. The Marimo notebooks (`.py` files) are the interactive tools; the Jupyter notebooks (`.ipynb`) are earlier explorations.

**The notebooks are no longer where ranking is measured.** That is done by scripts: `make eval` (`scripts/run_eval.py`) and `make eval-graded` (`scripts/graded_eval.py`), with results recorded in [evaluation-baseline.md](evaluation-baseline.md). The notebooks predate the film tags, the keyword-vector mix, the fitted blend and query understanding, and some of their conclusions were later overturned (see *Key Insights* below). One thing still depends on them: the 15-mood gold set (`TEST_SET` in `1-evaluation.py`) is read from that file by the corpus build and by the benchmarks, so do not move or rename it.

## Running the Notebooks

### Marimo Notebooks

```bash
# Install marimo if not already present
pip install marimo

# Run a notebook interactively
marimo edit notebooks/1-evaluation.py
```

Marimo notebooks are reactive -- changing a cell automatically re-runs dependent cells. Variables must be defined in exactly one cell; prefix temporary variables with `_` to make them cell-private.

### Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

## Notebook Overview

### `1-evaluation.py` -- Evaluation Framework

**Purpose:** Establish a gold test set and baseline metrics for the recommendation engine.

**What it does:**

1. **Gold Test Set** -- defines mood/title pairs where a human expert has identified the expected good matches (e.g., "slow-burn psychological horror" should return *Hereditary*, *The Witch*, etc.).
2. **Corpus Loading** -- loads the pre-built horror corpus from `data/horror_corpus.json`.
3. **Semantic Search Evaluation** -- runs `semantic_search()` against each mood and checks how many gold titles appear in the top-K results.
4. **Metrics Computation:**
   - **Hit Rate@K** -- fraction of moods where at least one gold title appears in the top K.
   - **Precision@K** -- average fraction of gold titles in the top K.
   - **NDCG@K** -- normalised discounted cumulative gain, considering rank positions.
   - **MRR** -- mean reciprocal rank of the first gold title found.
5. **Baseline Results** -- reports metrics for the default `all-mpnet-base-v2` model as a baseline.

### `2-embedding-models.py` -- Model Comparison

**Purpose:** Compare different sentence-transformer models on recommendation quality and latency.

**Models compared:**

| Model | Dimension | Characteristics |
|-------|-----------|----------------|
| `all-MiniLM-L6-v2` | 384 | Fast, lightweight |
| `all-mpnet-base-v2` | 768 | Best quality (default) |
| `BAAI/bge-small-en-v1.5` | 384 | Strong for retrieval |
| `BAAI/bge-base-en-v1.5` | 768 | Alternative to mpnet |

**What it does:**

1. Loads the corpus and gold test set.
2. For each model, embeds all corpus plots and each mood query.
3. Computes semantic search for each mood and evaluates Hit Rate@K, Precision@K, NDCG@K.
4. Measures encoding latency per model.
5. Produces comparison tables and heatmaps.

### `3-cross-encoder.py` -- Cross-Encoder Reranking

**Purpose:** Evaluate whether cross-encoder reranking improves precision over bi-encoder-only search.

**Cross-encoders tested:**

| Model | Size | Notes |
|-------|------|-------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M params | Good accuracy/speed balance |
| `cross-encoder/ms-marco-TinyBERT-L-2-v2` | 4.4M params | Fastest, lower accuracy |

**What it does:**

1. Runs bi-encoder semantic search to get a candidate pool.
2. Re-scores the top candidates using each cross-encoder (scoring each mood-plot pair individually).
3. Evaluates metrics at different rerank depths (top-20, top-30, top-50).
4. Compares metrics with and without cross-encoder reranking.

### `4-weight-tuning.py` -- Weight Optimisation

**Purpose:** Find optimal blending weights for the unified recommender through grid search.

**What it does:**

1. Pre-computes all scoring signals for the corpus (semantic similarity, keyword overlap, popularity, recency).
2. Defines a grid of weight combinations for the four signals.
3. For each weight combination, simulates the unified scoring pipeline and evaluates NDCG@6.
4. Also varies the MMR diversity lambda.
5. Reports the best weight configuration found.

**Weight grid:**

- Semantic: 0.3 to 0.6
- Keyword: 0.1 to 0.3
- Popularity: 0.1 to 0.3
- Recency: 0.0 to 0.1
- MMR lambda: 0.5 to 0.9

### Legacy Jupyter Notebooks

| Notebook | Description |
|----------|-------------|
| `0-test-recommendation.ipynb` | Initial OMDb recommendation testing |
| `0-1-improvements-baseline.ipynb` | Early baseline evaluation |
| `0-2-better-use-embeddings.ipynb` | Embedding-based improvements exploration |

These are kept for historical reference but are superseded by the Marimo notebooks.

## Key Insights

What the notebooks suggested, and what later measurement on the full corpus said (details in [evaluation-baseline.md](evaluation-baseline.md)):

1. **`all-mpnet-base-v2` is the best bi-encoder of those tried.** Still the model in use.
2. **Cross-encoder reranking: rejected.** The notebook's gain was measured on an old 21-film corpus. On 500 films it gained +0.0026 NDCG, lost precision and added about 1.1 seconds per query. It is wired to a setting and off.
3. **Blending beats pure semantic search, but only once a new signal existed.** Re-weighting the notebook's four signals gained nothing, twice. The shipped blend (tags .35, semantic .30, keyword .25, popularity .10, recency 0) was fitted by ridge regression on graded judgments after the tag signal was added; as served, unified beats semantic by +0.063 NDCG@6.
4. **MMR costs a little NDCG and is kept on purpose.** Turning it off scores higher, because the metric gives no credit for variety. Lambda stays at 0.7.
5. **What is embedded matters more than which model embeds it.** Not "full plots": plot summaries carry almost none of the tone and subgenre words people type. Adding TMDB keywords to the embedded text moved NDCG@6 from 0.31 to 0.50, and descriptive tags added more.
