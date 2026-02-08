# Architecture

This document describes the high-level system design of TerrorReco, its components, and how data flows from a user query to rendered recommendations.

## System Overview

TerrorReco is a server-rendered FastAPI application. The browser sends traditional form submissions; the server processes them, calls the recommendation pipeline, and returns fully rendered HTML. A small amount of client-side JavaScript powers the movie detail modal and the feedback (like/dislike) AJAX calls.

```
Browser (Jinja2 HTML)
   |  POST /search  or  GET /recommend
   v
FastAPI (app/main.py)
   |
   +-- Auth layer (app/auth.py, app/security.py)
   +-- Session & DB (app/db.py, app/models.py)
   |
   +-- Strategy Router
   |      |
   |      +-- "semantic"   --> recommend_movies_advanced()
   |      +-- "unified"    --> recommend_movies_advanced() --> recommend_unified_semantic()
   |      +-- "embedding"  --> recommend_movies(strategy="embedding")
   |      +-- "keyword"    --> recommend_movies(strategy="keyword")
   |
   +-- Filters (year range, min IMDb rating, language, type)
   +-- Stochastic Sampling (score perturbation, weighted selection, jittered MMR)
   |
   v
Jinja2 Template (results.html)
   |
   +-- Movie cards grid
   +-- Movie detail modal (JS)
   +-- Feedback buttons (JS --> POST /api/feedback)
```

## Component Responsibilities

### `app/main.py` -- Application Core

The entry point. Defines all routes, the startup event (database init, model pre-loading), the strategy router, the feedback API, and template rendering. Includes the `STRATEGY_LABELS` mapping and the `/recommend` endpoint that dispatches to the correct strategy.

### `app/auth.py` -- Authentication Router

Handles `/login`, `/register`, `/logout`. Uses Argon2 password hashing via `app/security.py`. Sessions are stored in signed cookies (itsdangerous). CSRF tokens protect all POST forms.

### `app/models.py` -- Data Models

Three SQLAlchemy ORM models:

- **`User`** -- id, email, password_hash, created_at, relationships to history and feedback.
- **`SearchHistory`** -- records each search (mood, strategy, results JSON, timestamp).
- **`MovieFeedback`** -- per-user per-movie like/dislike with unique constraint on (user_id, imdb_id).

### `app/db.py` -- Database Layer

Connection management with support for SQLite (development) and PostgreSQL (production with SSL). Provides `get_db()` as a FastAPI dependency and `init_db()` for table creation with retry logic.

### `app/settings.py` -- Configuration

Pydantic Settings class that reads from environment variables. Centralises all configuration: OMDb key, Stripe keys, database URL, debug flag, unified recommender weights, diversity lambda, and more.

### `app/services/corpus.py` -- Corpus & Semantic Search

The heart of the ML pipeline:

1. **`build_corpus()`** -- fetches horror movies from OMDb via paginated search, deduplicates by IMDb ID, extracts full details (plot, director, actors, etc.), and persists to `data/horror_corpus.json`.
2. **`load_corpus()`** -- loads the cached corpus from disk.
3. **`get_corpus_embeddings()`** -- computes or loads sentence-transformer embeddings for all corpus plots, cached to `data/corpus_embeddings.npy`.
4. **`semantic_search()`** -- embeds the user query, computes cosine similarity against the corpus, optionally adds temperature-controlled noise, and returns the top-K results.

### `app/services/recommender.py` -- Recommendation Orchestrator

Provides `recommend_movies()` (delegates to keyword or embedding strategies) and `recommend_movies_advanced()` (uses corpus semantic search with filters and weighted random sampling for non-determinism).

### `app/services/unified_recommender.py` -- Unified Blended Scorer

Implements `recommend_unified_semantic()` which blends four signals:

- Semantic similarity (sentence-transformer cosine)
- Keyword overlap proxy
- Popularity (IMDb rating x log-votes)
- Recency (year-based)

Applies Maximal Marginal Relevance (MMR) for diversity. Adds controlled noise and jittered lambda for non-determinism.

### `app/services/strategies/` -- Strategy Implementations

- **`base.py`** -- `RecommenderStrategy` protocol defining the `recommend()` interface.
- **`keyword_omdb.py`** -- `KeywordOMDbStrategy` that expands mood into search queries, fetches from OMDb, and ranks by IMDb rating.
- **`embedding_omdb.py`** -- `EmbeddingOMDbStrategy` that fetches candidates from OMDb and ranks by TF-IDF cosine similarity on plot text.

### `app/services/omdb_client.py` -- OMDb API Client

Async HTTP client (httpx) for OMDb. Supports search (`s=`) and detail (`i=`) lookups. Used by both the corpus builder and the keyword/embedding strategies.

### `app/stripe_payments.py` -- Payments

Stripe Checkout integration for "Buy me a coffee" donations. Handles session creation, success/cancel redirects, and webhook verification.

### Templates (`app/templates/`)

Server-rendered Jinja2 templates with a shared `_base.html` that contains the dark theme CSS, navigation, and flash message rendering.

| Template | Route | Purpose |
|----------|-------|---------|
| `index.html` | `GET /` | Search form with mood input, strategy selector, advanced filters |
| `loading.html` | `GET /loading` | Animated loading screen, auto-redirects to `/recommend` |
| `results.html` | `GET /recommend` | Movie grid, detail modal, feedback buttons |
| `login.html` | `GET /login` | Login form |
| `register.html` | `GET /register` | Registration form with password strength meter |
| `history.html` | `GET /history` | User's past searches |
| `coffee.html` | `GET /stripe/coffee` | Coffee purchase page |
| `coffee_success.html` | `GET /stripe/success` | Payment success |
| `coffee_cancel.html` | `GET /stripe/cancel` | Payment cancelled |

## Data Flow: From Query to Results

1. User types a mood on the index page and submits the form.
2. The browser navigates to `/loading?mood=...&strategy=...&filters...`.
3. `loading.html` shows a spooky loading animation, then JS redirects to `/recommend?...`.
4. `ui_recommendations()` in `main.py` dispatches to the selected strategy.
5. The strategy fetches/searches movies, applies filters (year, rating, language).
6. Stochastic sampling selects the final set from a larger pool.
7. Results are passed to `results.html` which renders the movie card grid.
8. Clicking a card opens the detail modal (pure JS, no server round-trip).
9. Like/dislike buttons send `POST /api/feedback` via fetch (AJAX).

## Persistence

| Store | Format | Purpose |
|-------|--------|---------|
| SQLite / PostgreSQL | Relational DB | Users, search history, movie feedback |
| `data/horror_corpus.json` | JSON file | Cached horror movie corpus from OMDb |
| `data/corpus_embeddings.npy` | NumPy binary | Pre-computed sentence-transformer embeddings |
| `models/` | HuggingFace cache | Locally cached sentence-transformer model weights |
