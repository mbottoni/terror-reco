# TerrorReco

> A horror movie recommendation engine powered by semantic search, sentence-transformers, and the OMDb API.

[![CI](https://github.com/maruanottoni/terror_reco/actions/workflows/ci.yml/badge.svg)](https://github.com/maruanottoni/terror_reco/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## What is TerrorReco?

TerrorReco lets users describe a mood, scene, or vibe in free text and instantly get matched with horror movies that fit. Instead of browsing categories or tags, you just write something like:

- *"a slow-burn psychological thriller set in a remote cabin"*
- *"found footage alien abduction at night"*
- *"gothic vampire romance in Eastern Europe"*

The system uses **ML-based semantic search** (sentence-transformers) against a pre-built corpus of horror movies to find the best matches -- no hardcoded keyword lists.

## Key Features

- **Semantic Search** -- sentence-transformer embeddings (`all-mpnet-base-v2`) match your text against movie plot descriptions by meaning, not keywords
- **4 Recommendation Strategies** -- Semantic Search, Unified (AI + Diversity via MMR), TF-IDF Similarity, and Keyword Match, selectable from the UI
- **Non-Deterministic Results** -- controlled randomness (score perturbation, weighted sampling, jittered MMR) ensures fresh recommendations every time
- **Advanced Filters** -- minimum year, maximum year, minimum IMDb rating, result count, type (movie/series), English-only toggle
- **Movie Detail Modal** -- click any card to see full details (director, cast, runtime, genre, awards) with a direct IMDb link
- **Like / Dislike Feedback** -- logged-in users can rate recommendations; feedback is stored for future personalisation
- **User Accounts** -- registration, login, session management with Argon2 password hashing and CSRF protection
- **Search History** -- past searches are saved and browsable per user
- **Stripe Integration** -- "Buy me a coffee" payments with checkout, webhooks, success/cancel flows
- **Dark Horror UI** -- fully responsive dark theme designed for the genre
- **Docker & CI/CD** -- Dockerfile with pre-downloaded ML model, GitHub Actions pipeline (lint, type-check, test, build)

## Architecture at a Glance

```
User Query
    |
    v
[FastAPI App]  -->  Strategy Router
    |                   |
    |    +--------------+--------------+--------------+
    |    |              |              |              |
    v    v              v              v              v
 Semantic          Unified         TF-IDF        Keyword
 (corpus +      (semantic +      (sklearn       (OMDb title
  SBERT          blended +       cosine on       search +
  embeddings)    MMR diversity)  OMDb plots)     IMDb rank)
    |              |              |              |
    +--------------+--------------+--------------+
                   |
                   v
            [Filters & Sampling]
            (year, rating, language,
             stochastic selection)
                   |
                   v
            [Results Page]
            (cards, modal, feedback)
```

See [docs/architecture.md](docs/architecture.md) for the full system design.

## Quick Start

### Prerequisites

- Python 3.11+
- An OMDb API key (free at [omdbapi.com/apikey.aspx](https://www.omdbapi.com/apikey.aspx))

### 1. Clone and install

```bash
git clone https://github.com/maruanottoni/terror_reco.git
cd terror_reco
make setup
# or manually:
python3.11 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e '.[dev]'
```

### 2. Configure environment

```bash
cp .env.example .env   # then edit with your keys
```

Required variables:

| Variable | Description |
|----------|-------------|
| `OMDB_API_KEY` | Your OMDb API key |
| `DEBUG` | Set to `1` for local development |
| `SECRET_KEY` | Session encryption key (auto-generated if missing) |

See [docs/deployment.md](docs/deployment.md) for all environment variables.

### 3. Run the server

```bash
make run
# or:
source .venv/bin/activate
set -a; source .env; set +a
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000) and start describing your horror mood.

### 4. Run with Docker

```bash
docker compose up --build
# or:
docker build -t terror-reco:latest .
docker run --env-file .env -p 8000:8000 terror-reco:latest
```

## Project Structure

```
terror_reco/
├── app/                        # FastAPI application
│   ├── main.py                 # App entry point, routes, startup
│   ├── auth.py                 # Authentication (login, register, logout)
│   ├── history.py              # Search history management
│   ├── models.py               # SQLAlchemy models (User, SearchHistory, MovieFeedback)
│   ├── db.py                   # Database connection & session management
│   ├── security.py             # Argon2 hashing, CSRF tokens
│   ├── settings.py             # Pydantic Settings configuration
│   ├── stripe_payments.py      # Stripe "Buy me a coffee" integration
│   ├── services/
│   │   ├── corpus.py           # Horror movie corpus builder & semantic search
│   │   ├── recommender.py      # Recommendation orchestrator (strategy routing)
│   │   ├── omdb_client.py      # Async OMDb API client
│   │   ├── unified_recommender.py  # Unified blended scorer with MMR
│   │   └── strategies/
│   │       ├── base.py         # Strategy protocol/interface
│   │       ├── keyword_omdb.py # Keyword search + IMDb ranking
│   │       └── embedding_omdb.py   # TF-IDF cosine similarity
│   ├── templates/              # Jinja2 HTML templates
│   └── static/                 # CSS, images, assets
├── notebooks/                  # Marimo & Jupyter evaluation notebooks
├── tests/                      # Pytest test suite
├── docs/                       # Project documentation
├── scripts/                    # Utility scripts (model download)
├── Dockerfile                  # Production container
├── docker-compose.yml          # Local Docker setup
├── Makefile                    # Common commands
└── pyproject.toml              # Python project config & tool settings
```

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | System design, data flow, component responsibilities |
| [Recommendation Engine](docs/recommendation-engine.md) | Strategies, ML pipeline, scoring, diversity, non-determinism |
| [API Reference](docs/api-reference.md) | All HTTP endpoints with parameters and responses |
| [Database](docs/database.md) | Schema, models, relationships, migrations |
| [Deployment](docs/deployment.md) | Docker, Render, CI/CD, environment variables |
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
| TF-IDF | scikit-learn |
| Movie data | OMDb API |
| Database | SQLAlchemy (SQLite dev / PostgreSQL prod) |
| Auth | Argon2 + session cookies + CSRF |
| Payments | Stripe Checkout |
| Containers | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Linting | Ruff + Black |
| Type checking | mypy (strict) |
| Testing | pytest |
| Notebooks | Marimo (interactive) + Jupyter |

## License

MIT
