# TerrorReco

> Describe a mood, get horror films that fit it. Semantic search with sentence-transformers over a curated corpus built from TMDB and OMDb.

[![CI](https://github.com/mbottoni/terror-reco/actions/workflows/ci.yml/badge.svg)](https://github.com/mbottoni/terror-reco/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## What is TerrorReco?

TerrorReco lets users describe a mood, scene, or vibe in free text and instantly get matched with horror movies that fit. Instead of browsing categories or tags, you just write something like:

- *"a slow-burn psychological thriller set in a remote cabin"*
- *"found footage alien abduction at night"*
- *"gothic vampire romance in Eastern Europe"*

The system uses **ML-based semantic search** (sentence-transformers) against a pre-built corpus of horror movies to find the best matches. It also reads the query for what a single vector cannot carry: "no gore", "from the 80s", "with Vincent Price", "like Hereditary but less bleak".

## Key Features

**Finding films**
- **Semantic search** -- sentence-transformer embeddings (`all-mpnet-base-v2`) match your text against each film's title, genre, keywords and plot by meaning
- **Two strategies** -- Unified, the default (descriptive tags + semantic + BM25 keywords + popularity, with MMR for variety), and plain Semantic Search
- **Descriptive tags** -- every film carries tags from a fixed vocabulary of 140 ("slow burn", "very gory", "folk horror"), which supply the tone words plot summaries lack
- **Queries are read, not just embedded** -- wanted and excluded qualities, decades and named people are picked out by rule; films that break one are moved down the ranking
- **"Like this film"** -- a query that names a film searches with that film's own vector and leaves the film and its franchise out
- **Fresh results every time** -- controlled randomness, and never two films from the same franchise on one page
- **Filters** -- year range, minimum IMDb rating, English only, result count, and a "more mainstream / more obscure" tilt
- **Refine in place** -- edit the mood, drop a filter or shuffle from the results page; an empty page says which filter caused it
- **Film details** -- cast, crew, runtime, awards, an IMDb link, "more like this" and where to watch in your region
- **Shareable results** -- a link that reproduces exactly the films on screen, with a proper preview when pasted

**Accounts**
- **Sign in** -- email and password, or Sign in with Google; addresses are confirmed by emailed link once an email provider is configured
- **Personalisation** -- pick films you love at sign-up; results lean towards your taste and say "because you liked ..."
- **Watchlist and history** -- save films for later and publish the list at an unguessable link; past searches are kept
- **Account controls** -- change password or email, download your data, delete your account, sign out everywhere, reset a forgotten password by email
- **Security** -- Argon2 hashing, revocable sessions, CSRF tokens on every write, rate limits, security headers
- **Privacy page** -- what is stored, who else receives data, and how to export or delete it

**Running it**
- **No network calls per request** -- the corpus, tags, embeddings and provider data ship with the app
- **A catalogue that grows by itself** -- a daily GitHub Actions job adds the next horror films it lacks, within the free API limits, tests the result and deploys it
- **Measured ranking** -- a hand-written gold set, a 55-mood graded benchmark, and a frozen slice replayed in CI
- **One-command deploys** -- push to `main`: tests, image build, an offline smoke test of the image, deploy, health check; one-click rollback; nightly database backups; an uptime check every five minutes
- **Stripe** -- "Buy me a coffee" checkout with a choice of amount, payments recorded from the webhook

## Architecture at a Glance

```
User query
    |
    v
[FastAPI]  -->  does the query name a film?  -->  filters (year, rating, language)  -->  strategy
                                                      |
                             +------------------------+------------------------+
                             |                                                 |
                             v                                                 v
                         Semantic                                    Unified (default)
                  (cosine similarity on                        (query read for qualities, exclusions,
                   cached corpus embeddings)                    decades and people; a 60-film pool
                             |                                  re-ranked by a blend: tags + semantic
                             |                                  + BM25 + popularity + your taste,
                             |                                  then MMR for variety)
                             +------------------------+------------------------+
                                                      |
                                                      v
                              one film per franchise, top of the ranking with score noise
                                                      |
                                                      v
                                   [Results page: cards, details, feedback]
```

See [docs/architecture.md](docs/architecture.md) for the full system design.

## Quick Start

### Prerequisites

- Python 3.11+

Running the app needs no API keys: the corpus is committed. Only rebuilding or growing it does:

- A TMDB API Read Access Token (free at [themoviedb.org](https://www.themoviedb.org/settings/api)) — corpus discovery
- An OMDb API key (free at [omdbapi.com/apikey.aspx](https://www.omdbapi.com/apikey.aspx)) — IMDb rating enrichment

### 1. Clone and install

```bash
git clone https://github.com/mbottoni/terror-reco.git
cd terror-reco
make setup
# or manually:
python3.11 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e '.[dev]'
```

### 2. Configure environment

Create a `.env` file in the project root:

```bash
DEBUG=1
SECRET_KEY=any-value-is-fine-with-debug-on
# Only needed to rebuild the corpus or refresh the where-to-watch snapshot:
TMDB_BEARER_TOKEN=...
OMDB_API_KEY=...
```

Variables:

| Variable | Description |
|----------|-------------|
| `TMDB_BEARER_TOKEN` | TMDB API Read Access Token — used to discover films for the corpus |
| `OMDB_API_KEY` | Your OMDb API key — used to enrich films with IMDb rating/votes |
| `DEBUG` | Set to `1` for local development |
| `SECRET_KEY` | Signs sessions and CSRF tokens. Any value works with `DEBUG=1`; production requires a random one of at least 32 characters |

See [docs/deployment.md](docs/deployment.md) for all environment variables.

### 3. The horror corpus

The corpus (`data/horror_corpus.json`, 650 films and growing) is committed, along with each
film's descriptive tags (`data/film_tags.json`), so there is nothing to build before the first
run. Embeddings are computed from it on first use.

To add films to it:

```bash
make grow        # the next 40 horror films the corpus lacks
```

The same script runs daily in GitHub Actions; see [docs/corpus-growth.md](docs/corpus-growth.md).

`make corpus` rebuilds the original 500-film base from TMDB and OMDb. It is resumable, but it
writes out only the films in its own checkpoint, so run on the grown corpus it would drop the
films `make grow` added. After changing corpus text, run `make eval-gate` as well: CI replays a
frozen slice of the evaluation and will fail until that fixture is rebuilt.

### 4. Run the server

```bash
make run
# or:
source .venv/bin/activate
set -a; source .env; set +a
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000) and start describing your horror mood.

### 5. Run with Docker

```bash
docker build -t terror-reco:latest .
docker run --env-file .env -p 8000:10000 terror-reco:latest
```

The image listens on `$PORT`, 10000 by default.

## Project Structure

```
terror-reco/
├── app/
│   ├── main.py                 # App, middleware, recommendation routes and APIs
│   ├── auth.py                 # Register, sign in, sign out, session revocation
│   ├── email_verification.py   # Confirm an address by emailed link
│   ├── google_auth.py          # Sign in with Google
│   ├── account.py              # Change password/email, export, delete account
│   ├── password_reset.py       # Reset by emailed single-use link
│   ├── onboarding.py           # Pick films you love after sign-up
│   ├── history.py              # Past searches
│   ├── watchlist.py            # Saved films, and publishing them at a public link
│   ├── stripe_payments.py      # Checkout and webhook (records payments)
│   ├── models.py, schemas.py   # SQLAlchemy models; request/response models
│   ├── db.py                   # Engine, sessions, startup migrations
│   ├── security.py             # Argon2, CSRF, URL checks
│   ├── middleware.py           # Security headers, rate limits
│   ├── observability.py        # Logging, optional Sentry
│   ├── static_assets.py        # Fingerprinted static URLs
│   ├── settings.py             # Pydantic settings
│   ├── services/
│   │   ├── corpus.py           # Corpus, tags, embeddings, semantic search
│   │   ├── recommender.py      # Semantic strategy, filters, shared links
│   │   ├── unified_recommender.py  # Blended scoring, tag signal, BM25, MMR
│   │   ├── query_understanding.py  # Qualities, exclusions, decades, people in a query
│   │   ├── title_reference.py  # Queries that name a film
│   │   ├── personalization.py  # Taste vector, "because you liked"
│   │   ├── providers.py        # Where-to-watch lookups
│   │   ├── email.py            # Transactional email
│   │   └── tmdb_client.py, omdb_client.py   # Used by offline scripts only
│   ├── templates/              # Jinja2 templates
│   └── static/                 # Stylesheet, fonts, icons, preview image
├── data/                       # Corpus, film tags, where-to-watch snapshot, growth log (committed)
├── eval/                       # Benchmark queries and graded judgments
├── migrations/                 # Alembic migrations
├── scripts/                    # Corpus build and growth, tag prediction, eval, snapshots
├── ops/                        # Backup script run on the server
├── tests/                      # Pytest suite
├── notebooks/                  # Marimo evaluation notebooks
├── docs/
├── .github/                    # CI, deploy/rollback, daily corpus growth, uptime check
├── Dockerfile, docker-compose.prod.yml, Caddyfile
├── Makefile
└── pyproject.toml, uv.lock
```

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | System design, data flow, component responsibilities |
| [Recommendation Engine](docs/recommendation-engine.md) | Strategies, ML pipeline, scoring, diversity, non-determinism |
| [API Reference](docs/api-reference.md) | All HTTP endpoints with parameters and responses |
| [Database](docs/database.md) | Schema, models, relationships, migrations |
| [Deployment](docs/deployment.md) | Production setup, CI/CD, rollback, backups, environment variables |
| [Corpus Growth](docs/corpus-growth.md) | How the catalogue grows daily, API limits, predicted tags |
| [Evaluation Baseline](docs/evaluation-baseline.md) | The measurement log: every ranking change and what it scored |
| [Research Notebooks](docs/notebooks.md) | Evaluation framework, model comparison, weight tuning |
| [Development Guide](docs/development.md) | Local setup, testing, linting, contributing |
| [Stripe Setup](docs/STRIPE_SETUP.md) | Payment integration configuration |
| [Stripe Testing](docs/STRIPE_TESTING.md) | Local payment testing guide |

## Make Targets

```bash
make setup       # Create venv and install dependencies
make run         # Start development server
make docker      # Build and run Docker container
make lint        # Run ruff + black checks
make typecheck   # Run mypy strict type checking
make test        # Run pytest suite
make grow        # Add the next 40 horror films the corpus lacks
make corpus      # Rebuild the 500-film base corpus (resumable; see the warning above)
make corpus-validate  # Check the existing corpus against quality gates
make eval        # Score the recommender against the hand-written gold set
make eval-graded # Score against the graded judgments (55 moods)
make eval-pool   # List films the rankers return that have no grade yet
make eval-gate   # Rebuild the frozen eval slice CI replays
make eval-export # Export an eval set from real feedback and usage
make tune        # Grid-search blend weights
make providers   # Refresh the where-to-watch snapshot
make migrate     # Apply database migrations
make migration m="..."  # Generate a migration from the models
make migrate-check      # Fail if models and migrations have drifted
make requirements       # Regenerate requirements.txt from uv.lock
make ci          # Run full CI pipeline (lint + typecheck + test)
make format      # Auto-fix linting issues
make clean       # Remove venv and caches
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Web framework | FastAPI + Uvicorn |
| Templating | Jinja2 |
| ML embeddings | sentence-transformers (`all-mpnet-base-v2`) |
| Keyword scoring | BM25 (in-house) |
| Film data | TMDB (discovery, keywords, franchises, providers) + OMDb (IMDb ratings) |
| Database | SQLAlchemy (SQLite dev / PostgreSQL prod) |
| Auth | Argon2 + revocable session cookies + CSRF; optional Google sign-in (OpenID Connect) |
| Email | Resend (optional: address confirmation, password reset) |
| Payments | Stripe Checkout |
| Hosting | Docker Compose on one server, Caddy for HTTPS |
| CI/CD | GitHub Actions (test, build, smoke-test, deploy, rollback, daily corpus growth, uptime check) |
| Linting | Ruff + Black |
| Type checking | mypy (strict) |
| Testing | pytest |
| Notebooks | Marimo (interactive) + Jupyter |

## License

MIT
