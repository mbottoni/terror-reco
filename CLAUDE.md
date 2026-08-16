# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
make setup       # venv + pip install -e '.[dev]'
make run         # sources .env, uvicorn --reload on :8000  (note: depends on setup, so it reinstalls first)
make ci          # lint + typecheck + test — same as GitHub Actions
make lint        # ruff check . && black --check .
make typecheck   # mypy app  (strict mode)
make test        # pytest -q
make format      # ruff --fix + black, never fails
make docker      # build + run container
```

Single test: `pytest tests/test_auth.py::test_register_success -v`

`make run` reinstalls the package every time. For a fast iteration loop, skip it:
`set -a; source .env; set +a; .venv/bin/uvicorn app.main:app --reload --port 8000`

Notebooks are Marimo `.py` files (reactive; a variable may be defined in only one cell, `_`-prefix cell-private ones): `marimo edit notebooks/1-evaluation.py`

## Architecture

FastAPI app, fully server-rendered with Jinja2. Movie data comes from **OMDb**; ranking is done with **sentence-transformers** (`all-mpnet-base-v2`). No SPA — a little inline JS drives the detail modal and the feedback AJAX call.

**Request path:** `index.html` form → `GET /loading` (animation, JS forwards query params) → `GET /recommend` (`app/main.py`) → strategy router → `results.html`.

Because `/loading` sits between the form and the results, **any new query parameter must be added in three places**: the `Query()` signature in `ui_recommendations()`, the form input in `index.html`, and the param-forwarding array in `loading.html`. Miss the last one and the filter silently vanishes.

### The four strategies

`ui_recommendations()` in `app/main.py` dispatches on `strategy`, validated against `STRATEGY_LABELS`. Two families with very different data sources:

**Corpus-based (offline, fast)** — `semantic` and `unified` search a local cached corpus:
- `services/corpus.py` — `build_corpus()` sweeps ~65 broad OMDb title terms, keeps horror-genre hits, caches to `data/horror_corpus.json`; `get_corpus_embeddings()` caches SBERT vectors to `data/corpus_embeddings.npy`; `semantic_search()` does the query-time cosine ranking.
- `services/recommender.py` → `recommend_movies_advanced()` — semantic search, then year/rating/language filters, then weighted random sampling.
- `services/unified_recommender.py` → `recommend_unified_semantic()` — takes a 60-item pool from the above and re-ranks by a blend (semantic .45 / keyword overlap .20 / popularity .20 / recency .05), then applies MMR for diversity.

**Live-OMDb (slow, network-bound)** — `keyword` and `embedding` in `services/strategies/`, both fetching from OMDb per request. `embedding_omdb.py` uses sklearn TF-IDF, *not* the neural model, despite the name.

The embedding cache is keyed only on `len(corpus)` (`corpus.py:287`). Changing plot text without changing the movie count silently reuses stale vectors — delete `data/corpus_embeddings.npy` when in doubt. `_save_corpus()` already unlinks it on write.

### Everything is deliberately non-deterministic

Three independent randomness sources, all intentional: Gaussian score perturbation scaled to the pool's score spread (`corpus.py`, `unified_recommender.py`), weighted sampling without replacement over the top pool (`recommender.py`), and a ±0.08 jitter on the MMR lambda. **Never write a test asserting a specific movie ordering** — assert on shape, count, and membership instead. `temperature=0` in `semantic_search()` is the deterministic escape hatch.

### Config and DB

`settings.py` is a Pydantic `BaseSettings` behind `@lru_cache`. Tests that change env vars must call `get_settings.cache_clear()` — see `tests/test_recommender_omdb.py`.

`db.py` normalises `postgres://` URLs to the `postgresql+psycopg` dialect and passes SSL/keepalive via `connect_args` rather than the URL query string (putting `sslmode` in the query is unreliable with this dialect). `init_db()` retries with backoff but never raises — the app boots even with the DB down.

## Repo state worth knowing

- **The committed corpus is only 21 movies**, mostly titles containing the literal word "horror". `build_corpus()` bails after 5 consecutive OMDb errors, and a rate-limited run stopped it early. Both corpus-based strategies are ranking against those 21 items, so semantic quality is far below what `docs/recommendation-engine.md` describes. Rebuilding it (with backoff/resume) is the highest-impact fix available.
- `data/` and `models/` are gitignored and **not** copied into the Docker image, so a fresh container rebuilds the corpus from OMDb on first request.
- `PyYAML` is imported by `strategies/embedding_omdb.py` but not declared in `pyproject.toml` — it only resolves transitively via transformers.
- CSRF tokens are HMAC-verified but not bound to the session: `validate_csrf_token()` never compares against `request.session["csrf"]`, so a token from any session validates in any other.
- `tests/` mixes pytest files (`test_*.py`) with manual scripts (`manual_*.py`, `debug_stripe.py`, `deployment_checklist.py`) that hit real services and are not collected by pytest.
- OMDb is always mocked in tests via `respx`; `conftest.py` forces `DEBUG=true` so session cookies aren't `Secure` (TestClient speaks plain HTTP) and swaps the DB for in-memory SQLite.

## Adding a strategy

Four coordinated edits: the class in `services/strategies/` (must satisfy the `recommend(mood, limit)` protocol in `base.py`), a branch in `get_strategy()` in `services/recommender.py`, a branch plus a `STRATEGY_LABELS` entry in `app/main.py`, and the `<option>` in `index.html`.
