# Development Guide

This guide covers local setup, code organisation, testing, linting, and contributing to TerrorReco.

## Prerequisites

- Python 3.11+
- Git
- (Optional) Docker & Docker Compose
- Only to rebuild or grow the corpus: a TMDB API Read Access Token (discovery) and an OMDb API key (rating enrichment)

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/mbottoni/terror-reco.git
cd terror-reco
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

Create a `.env` file in the project root (there is no `.env.example`):

```bash
DEBUG=1
SECRET_KEY=any-value-is-fine-with-debug-on
# Only to rebuild or grow the corpus, or refresh the where-to-watch snapshot:
TMDB_BEARER_TOKEN=...
OMDB_API_KEY=...
```

Email, Google sign-in and Stripe are off until their variables are set; see
[deployment.md](deployment.md).

### 4. The horror corpus

The corpus (`data/horror_corpus.json`) and its tags (`data/film_tags.json`) are committed, so
there is nothing to build. Embeddings are computed on the first search (a minute or so) and
cached in `data/`.

```bash
make grow                # add the next 40 horror films the corpus lacks
make corpus-validate     # re-check the existing corpus against the quality gates
```

`make grow` requires `TMDB_BEARER_TOKEN` and `OMDB_API_KEY`; see
[corpus-growth.md](corpus-growth.md). `make corpus` rebuilds the original 500-film base and
writes out only the films in its own checkpoint, so on the grown corpus it would drop the films
added since. If the corpus file is missing, a search raises `CorpusNotBuiltError`: a build never
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
├── email_verification.py  # Confirm an address by emailed link
├── google_auth.py       # Sign in with Google
├── account.py           # Change password/email, export, delete account
├── password_reset.py    # Reset by emailed single-use link
├── onboarding.py        # Pick films you love after sign-up
├── history.py           # Past searches
├── watchlist.py         # Saved films; publishing them at a public link
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
│   ├── corpus.py             # Corpus, tags, embeddings, semantic search
│   ├── recommender.py        # Semantic strategy, filters, shared links
│   ├── unified_recommender.py  # Blended scoring, tag signal, BM25, MMR
│   ├── query_understanding.py  # Qualities, exclusions, decades, people in a query
│   ├── title_reference.py    # Queries that name a film
│   ├── personalization.py    # Taste vector, "because you liked"
│   ├── providers.py          # Where-to-watch lookups
│   ├── email.py              # Transactional email
│   ├── tmdb_client.py        # Async TMDB client (offline scripts)
│   └── omdb_client.py        # Async OMDb client (offline scripts)
├── templates/           # Jinja2 templates
└── static/              # Stylesheet, fonts, icons, preview image
```

`corpus.py`, `unified_recommender.py`, `query_understanding.py` and `title_reference.py` must
not import from the rest of the app: the Docker build computes embeddings with only the first
two copied in, and fails if they need anything else at import time.

Outside `app/`: `data/` (corpus, tags, where-to-watch snapshot, growth log), `eval/` (benchmark
queries and graded judgments), `scripts/` (corpus, tags, evaluation), `migrations/`, `ops/`.

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
| `tests/test_email_verification.py`, `test_google_auth.py` | Address confirmation and neutral sign-up answers; Google sign-in with the profile fetch stubbed |
| `tests/test_account.py`, `test_password_reset.py`, `test_onboarding.py` | Account settings, reset flow, taste picks |
| `tests/test_csrf.py`, `test_middleware.py`, `test_settings.py`, `test_security_hardening.py` | CSRF on every write route, headers and rate limits, startup guards |
| `tests/test_ranking.py`, `test_personalization.py`, `test_query_cache.py` | BM25, MMR, franchise rule, taste signal, query cache |
| `tests/test_tag_signal.py`, `test_film_tags.py` | The tag signal in the blend; the tag file's vocabulary, coverage and where tags are used |
| `tests/test_query_understanding.py`, `test_title_reference.py` | Reading qualities, exclusions, decades and people; detecting a named film, and never in a benchmark mood |
| `tests/test_strategy_routing.py`, `test_served_ranking.py`, `test_recently_shown.py`, `test_blank_filters.py` | Strategy fallback and default, what a request actually serves, demotion of films just shown, blank filter fields |
| `tests/test_eval_gate.py`, `test_graded_eval.py` | Frozen slice of the eval, replayed without the model; the graded benchmark's scoring |
| `tests/test_refine.py`, `test_share.py`, `test_templates.py`, `test_link_previews.py` | Results page behaviour, shared links, escaping, accessibility, previews |
| `tests/test_ui_shell.py`, `test_error_pages.py`, `test_privacy.py`, `test_favicon.py` | Nav and account menu on every page, error pages, privacy page, icon |
| `tests/test_set_feedback.py`, `test_interactions.py`, `test_json_bodies.py` | Feedback and usage APIs, body validation |
| `tests/test_watchlist.py`, `test_public_lists.py` | Saving films; publishing and withdrawing a public list |
| `tests/test_stripe.py` | Checkout, chosen amounts, success page, webhook and payment recording, with the SDK mocked |
| `tests/test_providers.py`, `test_static_assets.py`, `test_observability.py`, `test_startup.py` | Where to watch, static caching, logging, lifespan |
| `tests/test_corpus_build.py`, `test_grow_corpus.py` | TMDB to corpus mapping, OMDb overlay, embedding cache key, shipped corpus checks; growth rules, API budget and predicted tags |
| `tests/test_discovery.py` | Match explanations on each card and "more like this" |

### Key testing patterns

- **No test touches the network.** The request path makes no outbound calls; Stripe, the email sender and Google's profile fetch are replaced with fakes.
- **Accounts have two modes.** With an email provider configured, sign-up requires a confirmed address; without one it signs in at once. Tests cover both; use the `outbox` fixture for the first.
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
4. **Measure it**: `make eval-graded` (55 moods, the tighter interval) and `make eval`, and read `docs/evaluation-baseline.md` first. A single run is not a measurement, and a ranker whose results were never graded shows holes that score zero: run `make eval-pool` and grade them first.
5. **Write tests** that assert on shape and membership, not order.

## Adding a New Filter

1. Add the parameter and its logic to `_passes_filters()` and `recommend_movies_advanced()` in `app/services/recommender.py`. Filters run before ranking.
2. Add the `Query()` parameter to `ui_recommendations()` in `app/main.py` and pass it through.
3. Add it to `refine_context()` in `app/main.py`, so the results page shows a removable chip for it and keeps it when the search is refined.
4. Add it to `films_without_each_filter()`, so the empty state can name it.
5. Add the input in `app/templates/index.html`.

## Changing What Gets Embedded

Editing `embedding_text()`, the keyword-vector share (`KEYWORD_VECTOR_SHARE`), the text of a film the eval gate holds, or the tag vocabulary in `data/film_tags.json` changes ranking and will turn the CI eval gate red on purpose. Run `make eval-graded` to see the effect, then `make eval-gate` to rebuild the fixture, and commit both.

Tags must come from the vocabulary in `data/film_tags.json`. They go into the keyword vector, BM25's text and the match chips, and deliberately not into `embedding_text()`.

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
make grow        # Add the next 40 horror films the corpus lacks
make corpus      # Rebuild the 500-film base corpus (resumable; drops films added by make grow)
make corpus-validate  # Validate the existing corpus
make eval        # Score the recommender against the hand-written gold set
make eval-graded # Score against the graded judgments (55 moods)
make eval-pool   # List films the rankers return that have no grade yet
make eval-gate   # Rebuild the frozen eval slice CI replays
make eval-export # Export an eval set from real feedback and usage
make tune        # Grid-search blend weights
make providers   # Refresh the where-to-watch snapshot
make migration m="..."  # Generate a migration; make migrate applies it
make migrate-check      # Fail if models and migrations have drifted
make requirements       # Regenerate requirements.txt from uv.lock
make docker      # Build and run Docker
make clean       # Remove .venv and caches
```
