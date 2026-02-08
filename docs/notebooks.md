# Research Notebooks

TerrorReco includes a set of interactive notebooks for evaluating and tuning the recommendation engine. The Marimo notebooks (`.py` files) are the primary evaluation tools; the Jupyter notebooks (`.ipynb`) are earlier explorations.

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

The notebook experiments produced these findings (your specific numbers will vary based on corpus size):

1. **`all-mpnet-base-v2` is the best bi-encoder** for this task, with consistently higher NDCG@K and Hit Rate@K than smaller models.
2. **Cross-encoder reranking provides a modest precision boost** at the cost of significantly higher latency. For real-time recommendations, the bi-encoder alone is a good trade-off.
3. **Blending multiple signals outperforms pure semantic search** -- adding popularity and keyword overlap as secondary signals catches cases where semantic similarity alone misses well-known genre staples.
4. **MMR diversity is essential** -- without it, results tend to cluster around the same subgenre. Lambda ~0.7 balances relevance and variety well.
5. **Corpus quality matters more than model choice** -- a comprehensive, well-detailed corpus (with full plots) makes a bigger difference than switching between similarly-sized embedding models.
