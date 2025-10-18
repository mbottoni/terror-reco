# Recommendation Strategies (No-Training)

This document summarizes the production-ready strategies used to generate movie recommendations without training custom models. It focuses on semantic retrieval with `sentence-transformers`, blended scoring with lightweight priors, optional reranking, and diversity.

## Goals
- High-quality results from user mood text, without training.
- Deterministic and explainable signals (popularity, recency, facets).
- Optional precision boost via cross-encoder.
- Diverse final list (avoid near-duplicates).

## Pipeline Overview
1) Query understanding (optional, low-cost)
   - Normalize mood; optionally translate to English for OMDb search.
   - (Optional) Small LLM prompt to expand mood into subgenres/keywords/eras.

2) Candidate generation
   - OMDb multi-page search using expanded queries; filter to horror; dedupe.
   - Fetch details (prefer plot=full); cache results.

3) Semantic scoring (sentence-transformers)
   - Model: `sentence-transformers/all-mpnet-base-v2` (default).
     - Alternatives: `multi-qa-mpnet-base-dot-v1` (search-focused),
       `paraphrase-multilingual-mpnet-base-v2` (multilingual).
   - Embed mood and movie plots with `.encode(..., normalize_embeddings=True)`.
   - Compute cosine similarity; min–max normalize across the candidate set.

4) Blend with lightweight priors
   - Popularity: IMDb rating × log(1 + votes) (+ small Metascore term).
   - Facet/keyword proxy: token overlap between mood and title+plot (cheap signal).
   - Recency: weak prior from year (tune toward classic vs modern).
   - Example weights (tune as needed):
     - semantic 0.45, keyword/facet 0.20, popularity 0.20, recency 0.05.

5) Optional reranking (precision boost)
   - Cross-encoder on top-50 (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`).
   - Normalize cross-encoder scores and blend back with semantic (e.g., 0.8 CE + 0.2 semantic) to reorder the short-list.

6) Diversity (MMR)
   - Apply Maximal Marginal Relevance to pick K items with balance of relevance vs novelty.
   - λ ≈ 0.7 works well (higher = more relevance; lower = more variety).

7) Personalization (no training)
   - Session boosts: if a user opens/clicks items, slightly boost semantically similar movies within the session.
   - Hard constraints: user opt-outs (e.g., “no gore”) via zero-shot facet tags or keywords.

8) Explainability & UX
   - Show a short “why it matches” line (key facets/terms) for each recommendation.
   - Expose tuning controls: diversity λ, semantic vs popularity weight, modern/classic bias, language filter, type (movie/series/both).

## Sentence-Transformers Strategy (Details)
1) Texts to embed
   - Query: normalized mood string.
   - Documents: normalized movie plots (`overview`). If missing/short, reduce semantic weight and rely more on popularity/keyword.

2) Embedding & similarity
   - Use `all-mpnet-base-v2` by default.
   - Generate L2-normalized embeddings for mood and plots; compute cosine similarity.
   - Min–max normalize similarities to [0, 1] over the pool for downstream blending.

3) Blended scoring
   - Final score example:
     - `score = 0.45*semantic + 0.20*keyword + 0.20*popularity + 0.05*recency − penalties`
   - Penalties: language mismatch; user negatives (e.g., “no found-footage”); missing plots.

4) Cross-encoder rerank (optional)
   - Rerank top-50 with `cross-encoder/ms-marco-MiniLM-L-6-v2` over (mood, plot) pairs.
   - Blend: `0.8 * CE + 0.2 * semantic` for the short-list.

5) MMR selection
   - From the reranked pool, select final K via MMR to ensure diversity while keeping relevance.

## Performance & Ops
- Batch/embed plots; cache embeddings per `imdbID` and OMDb details.
- Parallelize OMDb calls (httpx) and embedding batches.
- Timeouts and fallbacks: on failure or short plots, fall back to keyword+popularity.

## Quickstart (Notebook)
The notebook `0-2-improvements-baseline copy.ipynb` contains:
- Utilities: sentence-transformers/TF‑IDF embeddings, normalization, MMR.
- `recommend_unified(...)`: blended scoring + optional cross-encoder + MMR.
- Example cell: builds an OMDb pool and runs the unified recommender.

Tip: set `use_cross_encoder=True` to enable heavier reranking when latency allows.


