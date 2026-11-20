# Development Guide

This guide covers local setup, code organisation, testing, linting, and contributing to TerrorReco.

## Prerequisites

- Python 3.11+
- Git
- (Optional) Docker & Docker Compose
- A TMDB API Read Access Token (corpus discovery) and an OMDb API key (rating enrichment)

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/maruanottoni/terror_reco.git
cd terror_reco
```

### 2. Create a virtual environment and install dependencies

```bash
make setup
# or manually:
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[dev]'
```

The `[dev]` extra installs testing and linting tools: pytest, mypy, ruff, black, respx.

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your OMDB_API_KEY
```

### 4. Build the horror corpus

```bash
make corpus              # ~500 films; resumable, safe to re-run after any interruption
make corpus-validate     # re-check an existing corpus against the quality gates
```

Requires `TMDB_BEARER_TOKEN` (discovery) and `OMDB_API_KEY` (rating enrichment). Corpus-based
strategies raise `CorpusNotBuiltError` until this has run -- the build deliberately never
happens inside a user request.

### 5. (Optional) Pre-download the ML model

```bash
python scripts/download_model.py
```

This caches the `all-mpnet-base-v2` sentence-transformer model locally in `models/`, avoiding a download on first request.

### 6. Start the development server

```bash
make run
# or:
source .venv/bin/activate
set -a; source .env; set +a
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The `--reload` flag enables auto-restart on code changes.

## Code Organisation

```
app/
├── main.py              # App, middleware, recommendation routes and JSON APIs
├── auth.py              # Register, sign in, sign out, session revocation
├── account.py           # Change password/email, export, delete account
├── password_reset.py    # Reset by emailed single-use link
├── onboarding.py        # Pick films you love after sign-up
├── history.py, watchlist.py
├── stripe_payments.py   # Checkout and webhook (records payments)
├── models.py            # SQLAlchemy models
├── schemas.py           # Request/response models for the JSON APIs
├── db.py                # Engine, sessions, startup migrations
├── security.py          # Argon2, CSRF token and check, URL scheme checks
├── middleware.py        # Security headers, rate limits
├── observability.py     # Logging, optional Sentry
├── static_assets.py     # Fingerprinted static URLs
├── settings.py          # Pydantic settings
├── services/
│   ├── corpus.py             # Corpus, embeddings, semantic search
│   ├── recommender.py        # Semantic strategy, filters, shared links
│   ├── unified_recommender.py  # Blended scoring, BM25, MMR
│   ├── personalization.py    # Taste vector, "because you liked"
│   ├── providers.py          # Where-to-watch lookups
│   ├── email.py              # Transactional email
│   ├── tmdb_client.py        # Async TMDB client (offline scripts)
│   └── omdb_client.py        # Async OMDb client (offline scripts)
├── templates/           # Jinja2 templates
└── static/              # Stylesheet, fonts, preview image
```

## Testing

### Run all tests

```bash
make test
# or:
pytest -q
```

### Test structure

| File | Tests |
|------|-------|
| `tests/conftest.py` | Shared fixtures: in-memory SQLite DB, TestClient, session override |
| `tests/test_auth.py` | Registration, sign-in, email normalisation, session revocation, log hygiene |
| `tests/test_account.py`, `test_password_reset.py`, `test_onboarding.py` | Account settings, reset flow, taste picks |
| `tests/test_csrf.py`, `test_middleware.py`, `test_settings.py` | CSRF on every write route, headers and rate limits, startup guards |
| `tests/test_ranking.py`, `test_personalization.py`, `test_query_cache.py` | BM25, MMR, franchise rule, taste signal, query cache |
| `tests/test_eval_gate.py` | Frozen slice of the eval, replayed without the model |
| `tests/test_refine.py`, `test_share.py`, `test_templates.py`, `test_link_previews.py` | Results page behaviour, shared links, escaping, accessibility, previews |
| `tests/test_set_feedback.py`, `test_interactions.py`, `test_json_bodies.py` | Feedback and usage APIs, body validation |
| `tests/test_stripe.py` | Checkout, success page, webhook and payment recording, with the SDK mocked |
| `tests/test_providers.py`, `test_static_assets.py`, `test_observability.py`, `test_startup.py` | Where to watch, static caching, logging, lifespan |
| `tests/test_corpus_build.py` | TMDB to corpus mapping, OMDb overlay, embedding cache key, shipped corpus checks |

### Key testing patterns

- **No test touches the network.** The request path makes no outbound calls; Stripe and the email sender are replaced with fakes.
- **The model is not needed.** Ranking tests supply vectors directly, and the eval gate replays stored ones.
- **The client behaves like a page.** The `client` fixture sends the session's CSRF token on `POST`, as real pages do. Use a plain `TestClient`, or pass an explicit `X-CSRF-Token`, to test the check itself.
- **Never assert a specific film order.** Ranking is deliberately random; assert on shape, count and membership, or pass `temperature=0`.
- **Database is in-memory** -- `conftest.py` creates a fresh SQLite database per test.
- **`DEBUG=true` is forced** -- ensures session cookies work over HTTP in `TestClient`.

### Running a single test

```bash
pytest tests/test_auth.py -v
pytest tests/test_auth.py::test_register_success -v
```

## Linting & Type Checking

### Ruff (linting)

```bash
make lint
# or:
ruff check .
```

Configuration in `pyproject.toml`:
- Line length: 100
- Rules: E, F, I (imports), B (bugbear), UP (pyupgrade)
- Excludes: `.venv`, `build`, `dist`, `notebooks`

### Black (formatting)

```bash
black .           # format
black --check .   # check only
```

### Mypy (type checking)

```bash
make typecheck
# or:
mypy app
```

Configuration: strict mode with `ignore_missing_imports = true`.

### Auto-fix

```bash
make format   # runs ruff --fix and black
```

## Running the Full CI Pipeline Locally

```bash
make ci
```

This runs `lint`, `typecheck`, and `test` in sequence -- the same checks as GitHub Actions.

To also test the Docker build:

```bash
make ci && docker build -t terror-reco:ci .
```

## Adding a New Recommendation Strategy

A strategy must search the corpus. Nothing on the request path may call TMDB or OMDb.

1. **Write the ranking function** in `app/services/` (see `recommend_movies_advanced()` and `recommend_unified_semantic()`). It is synchronous and CPU-bound, so the route calls it through `run_in_threadpool`.
2. **Add a branch** for it in `ui_recommendations()` in `app/main.py`, and an entry in `STRATEGY_LABELS`.
3. **Add the `<option>`** in `app/templates/index.html`. The results page's selector is built from `STRATEGY_LABELS` automatically.
4. **Measure it**: `make eval`, and read `docs/evaluation-baseline.md` first. A single run is not a measurement.
5. **Write tests** that assert on shape and membership, not order.

## Adding a New Filter

1. Add the parameter and its logic to `_passes_filters()` and `recommend_movies_advanced()` in `app/services/recommender.py`. Filters run before ranking.
2. Add the `Query()` parameter to `ui_recommendations()` in `app/main.py` and pass it through.
3. Add it to `refine_context()` in `app/main.py`, so the results page shows a removable chip for it and keeps it when the search is refined.
4. Add it to `films_without_each_filter()`, so the empty state can name it.
5. Add the input in `app/templates/index.html`.

## Changing What Gets Embedded

Editing `embedding_text()`, or rebuilding the corpus, changes ranking and will turn the CI eval gate red on purpose. Run `make eval` to see the effect, then `make eval-gate` to rebuild the fixture, and commit both.

## Database Changes

Schema changes go through Alembic: `make migration m="what changed"`, review the generated file, `make migrate`. The app applies migrations at startup, so a deploy migrates itself. Migrations only run forward; a rollback of the app does not roll the schema back.

## Commits and Pull Requests

Work on an issue goes through a pull request that references it ("Closes #N"). Titles are short, lowercase and unprefixed, describing what changed. Merging to `main` deploys to production, so the pull request's CI run is the gate: it lints, type-checks, runs the tests, builds the image and boots it offline.

## Common Make Targets

```bash
make setup       # Create venv, install deps
make run         # Start dev server with hot reload
make test        # Run pytest
make lint        # Check ruff + black
make typecheck   # Run mypy
make ci          # Full CI: lint + typecheck + test
make format      # Auto-fix linting issues
make corpus      # Build/refresh the horror corpus (resumable)
make corpus-validate  # Validate the existing corpus
make eval        # Score the recommender against the gold set
make eval-gate   # Rebuild the frozen eval slice CI replays
make providers   # Refresh the where-to-watch snapshot
make migration m="..."  # Generate a migration; make migrate applies it
make docker      # Build and run Docker
make clean       # Remove .venv and caches
```
