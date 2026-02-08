# Development Guide

This guide covers local setup, code organisation, testing, linting, and contributing to TerrorReco.

## Prerequisites

- Python 3.11+
- Git
- (Optional) Docker & Docker Compose
- An OMDb API key for integration testing

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

### 4. (Optional) Pre-download the ML model

```bash
python scripts/download_model.py
```

This caches the `all-mpnet-base-v2` sentence-transformer model locally in `models/`, avoiding a download on first request.

### 5. Start the development server

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
├── main.py              # App entry point: routes, startup, feedback API
├── auth.py              # Auth router: login, register, logout, CSRF
├── history.py           # History router: search history per user
├── models.py            # SQLAlchemy models: User, SearchHistory, MovieFeedback
├── db.py                # DB engine, session management, init
├── security.py          # Argon2 hashing, CSRF tokens
├── settings.py          # Pydantic Settings (env var config)
├── stripe_payments.py   # Stripe Checkout integration
├── services/
│   ├── corpus.py             # Corpus builder + semantic search
│   ├── recommender.py        # Strategy orchestrator
│   ├── omdb_client.py        # Async OMDb HTTP client
│   ├── unified_recommender.py  # Blended scorer + MMR
│   └── strategies/
│       ├── base.py           # Strategy protocol
│       ├── keyword_omdb.py   # Keyword search strategy
│       └── embedding_omdb.py # TF-IDF strategy
├── templates/           # Jinja2 HTML templates
└── static/              # CSS, images
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
| `tests/test_auth.py` | Registration, login, logout, validation, CSRF, flash messages |
| `tests/test_recommender_omdb.py` | Recommendation endpoint with mocked OMDb API |
| `tests/test_strategies.py` | Strategy implementations (keyword expansion, TF-IDF ranking) |

### Key testing patterns

- **External APIs are mocked** -- OMDb calls use `respx` to intercept HTTP requests.
- **Database is in-memory** -- `conftest.py` creates a fresh SQLite database per test session.
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

1. **Create the strategy class** in `app/services/strategies/`:

```python
# app/services/strategies/my_strategy.py
from typing import Any
from ..omdb_client import OMDbClient

class MyStrategy:
    def __init__(self, client: OMDbClient) -> None:
        self.client = client

    async def recommend(self, mood: str, limit: int = 6) -> list[dict[str, Any]]:
        # Your logic here
        ...
```

2. **Register it** in `app/services/recommender.py`:

```python
def get_strategy(name: str, client: OMDbClient) -> RecommenderStrategy:
    if name == "my_strategy":
        from .strategies.my_strategy import MyStrategy
        return MyStrategy(client)
    ...
```

3. **Add it to the strategy router** in `app/main.py` inside `ui_recommendations()`.

4. **Add it to the UI** in `app/templates/index.html` (strategy `<select>`) and update `STRATEGY_LABELS`.

5. **Write tests** in `tests/test_strategies.py`.

## Adding a New Filter

1. Add the parameter to `recommend_movies_advanced()` in `app/services/recommender.py`.
2. Add the filtering logic in the candidate loop.
3. Add the `Query()` parameter to `ui_recommendations()` in `app/main.py`.
4. Pass it through to the strategy calls.
5. Add the HTML input in `app/templates/index.html` (inside the filters panel).
6. Add the parameter name to the forwarding array in `app/templates/loading.html`.

## Commit Conventions

This project uses descriptive commit messages:

- `add: ...` for new features
- `fix: ...` for bug fixes
- `update: ...` for enhancements
- `refactor: ...` for restructuring
- `docs: ...` for documentation

## Common Make Targets

```bash
make setup       # Create venv, install deps
make run         # Start dev server with hot reload
make test        # Run pytest
make lint        # Check ruff + black
make typecheck   # Run mypy
make ci          # Full CI: lint + typecheck + test
make format      # Auto-fix linting issues
make docker      # Build and run Docker
make clean       # Remove .venv and caches
```
