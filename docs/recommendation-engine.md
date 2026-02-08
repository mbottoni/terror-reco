# Recommendation Engine

This document explains how TerrorReco matches user text to horror movies. It covers the four available strategies, the ML pipeline, scoring signals, diversity mechanisms, and the non-deterministic behaviour that ensures fresh results on every request.

## Design Philosophy

- **No hardcoded keywords** -- matching is purely ML-based. The system understands meaning, not just words.
- **Corpus-first** -- a pre-built collection of horror movies is searched, not the live API (except for keyword/embedding fallback strategies).
- **Non-deterministic by design** -- controlled randomness ensures users see different relevant movies each time.
- **Multiple strategies** -- users can choose the approach that best fits their needs.

## Strategies

### 1. Semantic Search (default)

**Route:** `strategy=semantic`
**Module:** `app/services/recommender.py` -> `recommend_movies_advanced()`

How it works:

1. Load the pre-built horror corpus (`data/horror_corpus.json`).
2. Load or compute sentence-transformer embeddings for all corpus plots (`data/corpus_embeddings.npy`).
3. Embed the user's query text with the same model (`all-mpnet-base-v2`).
4. Compute cosine similarity between the query embedding and all corpus embeddings.
5. Apply temperature-controlled Gaussian noise to similarity scores (score perturbation).
6. Return top-K candidates, then apply filters (year, rating, language).
7. Perform weighted random sampling from the filtered pool for the final selection.

### 2. Unified (AI + Diversity)

**Route:** `strategy=unified`
**Module:** `app/services/unified_recommender.py` -> `recommend_unified_semantic()`

Builds on top of semantic search by blending four scoring signals:

| Signal | Weight | Description |
|--------|--------|-------------|
| Semantic | 0.45 | Sentence-transformer cosine similarity |
| Keyword | 0.20 | Token overlap between query and title+plot |
| Popularity | 0.20 | IMDb rating x log(1 + votes) + Metascore term |
| Recency | 0.05 | Weak prior favouring newer films |

After blending:

1. Add Gaussian noise to blended scores.
2. Select a pool of top candidates by perturbed score.
3. Apply **Maximal Marginal Relevance (MMR)** with a jittered lambda to balance relevance vs diversity.

### 3. TF-IDF Similarity

**Route:** `strategy=embedding`
**Module:** `app/services/strategies/embedding_omdb.py`

A lighter-weight approach:

1. Fetch horror movie candidates from OMDb.
2. Build a TF-IDF matrix from movie plot descriptions (scikit-learn).
3. Compute cosine similarity between the user query and each plot.
4. Rank by similarity score.

### 4. Keyword Match

**Route:** `strategy=keyword`
**Module:** `app/services/strategies/keyword_omdb.py`

The simplest approach:

1. Expand the user's mood into search queries (title fragments, subgenres).
2. Search OMDb for each query.
3. Fetch full details for each result.
4. Rank by IMDb rating weighted by vote count.

## ML Models

### Sentence-Transformers (Bi-Encoder)

- **Default model:** `sentence-transformers/all-mpnet-base-v2`
- **Embedding dimension:** 768
- **Used for:** Encoding both user queries and movie plot descriptions into the same vector space.
- **Similarity metric:** Cosine similarity (embeddings are L2-normalised).
- **Pre-download:** The model is pre-downloaded during Docker build and cached locally in `models/` for development.

The Marimo notebooks (`notebooks/2-embedding-models.py`) compare several alternatives:

| Model | Dim | Notes |
|-------|-----|-------|
| `all-mpnet-base-v2` | 768 | Best quality, default |
| `all-MiniLM-L6-v2` | 384 | Faster, slightly lower quality |
| `BAAI/bge-small-en-v1.5` | 384 | Competitive quality, small |
| `BAAI/bge-base-en-v1.5` | 768 | Alternative to mpnet |

### Cross-Encoders (evaluated but not in production)

The notebook `notebooks/3-cross-encoder.py` evaluates cross-encoder reranking:

- `cross-encoder/ms-marco-MiniLM-L-6-v2`
- `cross-encoder/ms-marco-TinyBERT-L-2-v2`

These provide higher precision but at the cost of latency (each query-document pair must be scored individually). Currently available as an optional enhancement.

## The Horror Movie Corpus

The corpus is built by `app/services/corpus.py`:

1. **Search phase** -- paginated OMDb search for horror-related terms.
2. **Detail phase** -- fetch full details (plot, director, actors, ratings, etc.) for each result.
3. **Filter** -- only keep titles with "Horror" in their genre.
4. **Deduplicate** -- by IMDb ID.
5. **Persist** -- save to `data/horror_corpus.json`.

Each corpus entry contains:

```
imdb_id, title, overview (plot), poster_url, release_date, year,
vote_average (IMDb rating), genre, director, actors, writer,
runtime, language, country, rated, awards, imdbVotes, Metascore
```

The corpus is built once on the first request and cached. It can be incrementally updated.

## Non-Deterministic Recommendations

To avoid showing the same results every time for the same query, randomness is injected at three levels:

### 1. Score Perturbation (Semantic Search)

In `semantic_search()`, a `temperature` parameter controls Gaussian noise added to cosine similarity scores. A wider pool is fetched first, noise is applied proportional to the score spread, and the pool is re-ranked. This shuffles similar-scoring movies.

### 2. Weighted Random Sampling (Advanced Recommender)

In `recommend_movies_advanced()`, instead of taking the top-K deterministically, a larger pool is created and movies are selected via weighted random sampling without replacement. Weights decay linearly from 1.0 (best) to 0.3 (end of pool), so top results are favoured but not guaranteed.

### 3. Blended Noise + Jittered MMR (Unified Strategy)

In `recommend_unified_semantic()`:
- Gaussian noise is added to blended scores before ranking.
- The MMR diversity lambda is jittered by +/-0.08 each request, subtly changing the relevance-diversity trade-off.

## Filters

Users can apply these filters from the Advanced Filters panel:

| Filter | Parameter | Effect |
|--------|-----------|--------|
| Min year | `min_year` | Exclude movies before this year |
| Max year | `max_year` | Exclude movies after this year |
| Min IMDb rating | `min_rating` | Exclude movies below this IMDb score (0-10) |
| Results count | `limit` | Number of recommendations (1-20) |
| Type | `kind` | Movie, series, or both |
| English only | `english` | Only show English-language titles |

Filters are applied after semantic scoring but before the final sampling step, so filtered-out movies don't take up slots.

## Evaluation

The Marimo notebooks provide a rigorous evaluation framework:

- **Gold test set** -- mood/title pairs with known good matches.
- **Metrics** -- Hit Rate@K, Precision@K, NDCG@K, Mean Reciprocal Rank (MRR).
- **Experiments** -- model comparison, cross-encoder evaluation, weight grid search.

See [notebooks.md](notebooks.md) for details.
