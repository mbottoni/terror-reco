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
make corpus      # build/refresh the corpus (resumable)
make eval        # score against the gold set
make migrate     # alembic upgrade head
make tune        # grid-search blend weights
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

**Corpus-based (offline, fast)** — `semantic` and `unified` search a local cached corpus and do **no network I/O at request time**:
- `services/corpus.py` — load/save, the TMDB→corpus field mapping, embedding cache, and `semantic_search()` (query-time cosine ranking).
- **`embedding_text()` defines what gets embedded**: `title + genre + keywords + overview`, *not* the plot alone. Plot summaries are narrative prose and contain almost none of the tone/subgenre vocabulary users actually type ("campy", "folk horror"); TMDB keywords supply it. This one change moved NDCG@6 from 0.3111 to 0.4997 — see `docs/evaluation-baseline.md`. **Anything computing a semantic score must use `embedding_text()`**; `recommend_unified_semantic` once embedded `overview` alone and silently scored worse than the search feeding it.
- `services/recommender.py` → `recommend_movies_advanced()` — semantic search, then year/rating/language filters, then weighted random sampling.
- `services/unified_recommender.py` → `recommend_unified_semantic()` — takes a 60-item pool from the above and re-ranks by a blend (semantic .45 / keyword overlap .20 / popularity .20 / recency .05), then applies MMR for diversity.

**Live-OMDb (slow, network-bound)** — `keyword` and `embedding` in `services/strategies/`, both fetching from OMDb per request. `embedding_omdb.py` uses sklearn TF-IDF, *not* the neural model, despite the name.

### Corpus building

Built **offline** by `scripts/build_corpus.py` (`make corpus`), never during a request. Requesting a recommendation with no corpus raises `CorpusNotBuiltError` rather than silently crawling — an in-request crawl that got rate-limited is what previously froze the corpus at 21 films.

Discovery uses **TMDB** (`with_genres=27`), because OMDb's `s=` searches *titles only* and cannot browse a genre. OMDb is still used to enrich each film with `imdbRating`/`imdbVotes`/`Metascore`/`Awards`.

Three things about this pipeline are load-bearing:
- **`vote_average` must stay on the IMDb scale.** The `min_rating` filter and `_popularity()` (`rating × log(1+votes)`) assume IMDb semantics and the votes come from OMDb. `apply_omdb_enrichment()` overwrites the TMDB rating for this reason; `rating_source` records which scale a record ended up on.
- **Sort by `vote_count.desc`, not `popularity.desc`.** TMDB popularity is a live trending metric — sorting by it makes the corpus non-reproducible and skews it modern.
- **The eval gold set is force-seeded** (stage A), parsed out of `notebooks/1-evaluation.py` by AST so it can't drift. Without those ~112 titles in the corpus, every metric the notebook computes is measuring an empty intersection.

The build checkpoints to `data/.corpus_build_state.json` after every stage and every 50 records; `--resume` skips completed work. Embeddings are keyed on a **content hash** of the embedded text (`corpus_fingerprint()`), so editing plot text invalidates the cache even when the film count is unchanged.

### Everything is deliberately non-deterministic

Three independent randomness sources, all intentional: Gaussian score perturbation scaled to the pool's score spread (`corpus.py`, `unified_recommender.py`), weighted sampling without replacement over the top pool (`recommender.py`), and a ±0.08 jitter on the MMR lambda. **Never write a test asserting a specific movie ordering** — assert on shape, count, and membership instead. `temperature=0` is the deterministic escape hatch and is now threaded through `semantic_search()`, `recommend_movies_advanced()` and `recommend_unified_semantic()` — all three take `seed`/`temperature`, and `temperature=0` disables sampling and MMR jitter as well as score noise. `make eval` relies on this.

### Config and DB

`settings.py` is a Pydantic `BaseSettings` behind `@lru_cache`. Tests that change env vars must call `get_settings.cache_clear()` — see `tests/test_recommender_omdb.py`.

`db.py` normalises `postgres://` URLs to the `postgresql+psycopg` dialect and passes SSL/keepalive via `connect_args` rather than the URL query string (putting `sslmode` in the query is unreliable with this dialect). `init_db()` retries with backoff but never raises — the app boots even with the DB down.

## Repo state worth knowing

- `data/horror_corpus.json` **is committed** (626 KB) and copied into the Docker image, so deploys need no TMDB/OMDb keys at build time. Embeddings (`.npy`) and the build checkpoint stay gitignored and are regenerated from the corpus on first use.
- Filters are applied to the corpus **before** ranking (`_passes_filters` in `recommender.py`), keeping corpus rows and embedding rows aligned. Filtering the top-K afterwards used to starve restrictive queries.
- `PyYAML` is imported by `strategies/embedding_omdb.py` but not declared in `pyproject.toml` — it only resolves transitively via transformers.
- CSRF tokens are HMAC-verified but not bound to the session: `validate_csrf_token()` never compares against `request.session["csrf"]`, so a token from any session validates in any other.
- `tests/` mixes pytest files (`test_*.py`) with manual scripts (`manual_*.py`, `debug_stripe.py`, `deployment_checklist.py`) that hit real services and are not collected by pytest.
- OMDb is always mocked in tests via `respx`; `conftest.py` forces `DEBUG=true` so session cookies aren't `Secure` (TestClient speaks plain HTTP) and swaps the DB for in-memory SQLite.

## Database migrations

Schema changes go through **Alembic**, not `create_all`. `init_db()` runs `alembic upgrade head`
at startup (falling back to `create_all` only when `alembic.ini` is absent, e.g. in tests), so a
deploy migrates itself.

```bash
make migration m="what changed"   # autogenerate from the models
make migrate                      # upgrade to head
make migrate-check                # fail if models and migrations have drifted
```

`migrations/env.py` takes the URL from app settings, not `alembic.ini` — the ini's
`sqlalchemy.url` is deliberately blank. Generated migrations are auto-formatted by
alembic post-write hooks so they pass `make lint` without a manual pass.

## Personalisation and tuning

`MovieFeedback` now feeds ranking via `services/personalization.py`. Two mechanisms with
deliberately different strength: a **taste vector** (mean corpus embedding of liked films,
blended in as a weighted signal, requires >= `MIN_LIKES_FOR_TASTE` likes) and **demotion**
(disliked `imdb_id`s pushed below everything else). Disliked films are demoted, *not*
filtered, so a heavily-rated user still gets a full page. `_load_taste()` in `main.py`
swallows every error — personalisation must never break search.

**Two roadmap features were measured and deliberately left off** (`docs/evaluation-baseline.md`):
tuned blend weights beat the default by +0.0127 NDCG against a ~0.02 run-to-run stddev with
no holdout, and cross-encoder reranking gained +0.0026 NDCG while *losing* precision and
adding ~1.1s per query. Both are wired to settings (`UNIFIED_WEIGHTS`,
`UNIFIED_USE_CROSS_ENCODER`) so they stay testable. **Do not enable either by citing the
notebooks** — `notebooks/3-cross-encoder.py` measured its gain on the old 21-film corpus.

Before claiming any ranking change helps, run `make eval` — and remember a single run is
not a measurement, since the pipeline's stddev is comparable to most effects worth finding.

## Adding a strategy

Four coordinated edits: the class in `services/strategies/` (must satisfy the `recommend(mood, limit)` protocol in `base.py`), a branch in `get_strategy()` in `services/recommender.py`, a branch plus a `STRATEGY_LABELS` entry in `app/main.py`, and the `<option>` in `index.html`.
