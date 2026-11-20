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
make eval-gate   # rebuild the frozen eval slice CI replays
make providers   # refresh the where-to-watch snapshot (then commit it)
make migrate     # alembic upgrade head
make tune        # grid-search blend weights
```

Single test: `pytest tests/test_auth.py::test_register_success -v`

`make run` reinstalls the package every time. For a fast iteration loop, skip it:
`set -a; source .env; set +a; .venv/bin/uvicorn app.main:app --reload --port 8000`

Notebooks are Marimo `.py` files (reactive; a variable may be defined in only one cell, `_`-prefix cell-private ones): `marimo edit notebooks/1-evaluation.py`

## Architecture

FastAPI app, fully server-rendered with Jinja2. Movie data comes from **TMDB** (discovery, keywords, franchises, providers) and **OMDb** (IMDb ratings); ranking is done with **sentence-transformers** (`all-mpnet-base-v2`). No SPA — a little inline JS drives the detail modal and the feedback AJAX call.

**Request path:** `index.html` form → `GET /recommend` (`app/main.py`) → strategy router → `results.html`. There is no interstitial: the `/loading` page and its param-forwarding array were removed, so a new query parameter needs only the `Query()` signature in `ui_recommendations()` and the form input in `index.html`.

### The two strategies

`ui_recommendations()` in `app/main.py` dispatches on `strategy`, validated against `STRATEGY_LABELS`; an unknown value (including the retired `keyword`) falls back to the default. **The default is `unified`** (`USE_UNIFIED_RECOMMENDER`, on): as served it beats semantic by +0.063 NDCG@6. Both strategies search a local cached corpus and do **no network I/O at request time**:
- `services/corpus.py` — load/save, the TMDB→corpus field mapping, embedding cache, and `semantic_search()` (query-time cosine ranking).
- **`embedding_text()` defines what gets embedded**: `title + genre + keywords + overview`, *not* the plot alone. Plot summaries are narrative prose and contain almost none of the tone/subgenre vocabulary users actually type ("campy", "folk horror"); TMDB keywords supply it. This one change moved NDCG@6 from 0.3111 to 0.4997 — see `docs/evaluation-baseline.md`. **Anything computing a semantic score must use `embedding_text()`**; `recommend_unified_semantic` once embedded `overview` alone and silently scored worse than the search feeding it.
- **A film's stored vector is a mix**: 70% the vector of `embedding_text()`, 30% the vector of its keywords encoded alone, re-normalised (`mix_keyword_vectors()`, `KEYWORD_VECTOR_SHARE`). The mix is linear, so one stored matrix gives the blended cosine and nothing downstream knows there were two encodes. Measured at +0.04 NDCG@6 on the graded benchmark; scoring fields separately (max/mean) was far worse. Changing the share changes `EMBEDDING_RECIPE`, which invalidates the embedding cache and turns the eval gate red until `make eval-gate` is rerun.
- **Descriptive tags** live in `data/film_tags.json` (committed, copied into the image before the embeddings step): 140 fixed tags, 1-10 per film, **written by an AI model, not by TMDB or a person**. `load_corpus()` attaches them as a comma-separated `tags` field; `save_corpus()` strips it again. They go into `keyword_text()` (the second vector), into BM25's item text and into the "why this matched" chips, and **deliberately not into `embedding_text()`** (measured: that loses most of the gain). A film added by a corpus rebuild has no tags until someone adds them to the file; a test fails below 95% coverage. New tags must come from the file's vocabulary.
- `semantic` → `services/recommender.py` → `recommend_movies_advanced()` — semantic search, with year/rating/language filters applied first; the page is the top of that ranking.
- `unified` → `services/unified_recommender.py` → `recommend_unified_semantic()` — takes a 60-item pool from the above and re-ranks by a blend (tags .35 / semantic .30 / keyword .25 / popularity .10 / recency 0), then applies MMR for diversity. **The weights were fitted** (non-negative ridge on the graded judgments, checked on held-out moods: +0.07 NDCG@6); re-fitting the old four signals alone gained nothing, and bigger models (gradient boosting, 18 features) did worse held-out. The `tags` signal (`tag_scores()`) is the cosine between the query and the best-matching of a film's tag *definitions* ("tag: meaning", from the vocabulary in `data/film_tags.json`, embedded once per process); for a query that names a film it is the Jaccard overlap with that film's tags. It returns zeros rather than raising. The eval-gate fixture stores the tag-definition vectors, so **changing the tag vocabulary also needs `make eval-gate`**.

**Queries that name a film** ("like Hereditary but less bleak") are resolved by `services/title_reference.py`: the search vector becomes the film's stored vector (70/30 with the rest of the query), BM25 matches against the film's tags and keywords, and the film plus its franchise are excluded. `corpus.resolve_query()` is the single entry point; `recommend_movies_advanced()` calls it on the **whole** corpus before filtering and passes the result into `semantic_search()`. Detection is conservative by design (whole query, after a cue like "like", or leading before "but"/"meets"); a false positive hijacks a mood, so a test asserts that no benchmark mood triggers it. `COMMON_WORD_TITLES` lists single-word titles read as moods when typed alone; add to it when a corpus rebuild brings in a title that is an ordinary word. Measured with `graded_eval.py titles` (0.20 -> 0.64 NDCG@6 on 24 title queries). The module must stay free of app imports, like `corpus.py`.

**Queries are read before they are ranked** (`services/query_understanding.py`, used by the unified strategy through `query_intent()`): wanted tags (a tag's name, a short alias list, or a phrase close to a tag name), excluded tags inside a negation ("no gore", "not a comedy", "sad more than scary") with the opposite tag wanted, a decade, and people named in full (matched against the corpus's own credits). Three effects: `widen_pool()` adds keyword matches and fully-matching films to the 60 candidates (called from `recommend_movies_advanced()` when `keep_internal=True`, on the already-filtered rows); the tag signal becomes half aspect coverage; a film that breaks a stated fact loses `VIOLATION_PENALTY` (0.35) from its blended score. BM25's item text now includes director, writer, cast, country and language. It is rule-based and must stay free of app imports; anything unrecognised is ignored and ranked as before. `QUERY_UNDERSTANDING = False` turns all of it off for benchmarking. Measured on 23 constraint queries (`graded_eval.py score --group constraint`): 0.41 to 0.80 NDCG@6, moods unharmed. **The eval gate replays without it** (the fixture has no vectors for query phrases, so the intent comes back empty there).

The live-OMDb strategies (`keyword`, and the TF-IDF `embedding` before it) are gone. `services/omdb_client.py` remains only for the offline corpus build.

### Corpus building

Built **offline** by `scripts/build_corpus.py` (`make corpus`), never during a request. Requesting a recommendation with no corpus raises `CorpusNotBuiltError` rather than silently crawling — an in-request crawl that got rate-limited is what previously froze the corpus at 21 films.

Discovery uses **TMDB** (`with_genres=27`), because OMDb's `s=` searches *titles only* and cannot browse a genre. OMDb is still used to enrich each film with `imdbRating`/`imdbVotes`/`Metascore`/`Awards`.

Three things about this pipeline are load-bearing:
- **`vote_average` must stay on the IMDb scale.** The `min_rating` filter and `_popularity()` (`rating × log(1+votes)`) assume IMDb semantics and the votes come from OMDb. `apply_omdb_enrichment()` overwrites the TMDB rating for this reason; `rating_source` records which scale a record ended up on.
- **Sort by `vote_count.desc`, not `popularity.desc`.** TMDB popularity is a live trending metric — sorting by it makes the corpus non-reproducible and skews it modern.
- **The eval gold set is force-seeded** (stage A), parsed out of `notebooks/1-evaluation.py` by AST so it can't drift. Without those ~112 titles in the corpus, every metric the notebook computes is measuring an empty intersection.

**The catalogue grows by itself.** `scripts/grow_corpus.py` (`make grow`, and daily in `.github/workflows/grow-corpus.yml`) adds the next horror films the corpus lacks, alternating TMDB's most-voted and best-rated lists, within OMDb's free 1,000 requests a day (default budget 200 a run). Rules for inclusion and the 90-day memory of declined films are in `rejection_reason()` and `data/corpus_growth.json`. New films get **predicted** tags (`scripts/tag_predictor.py`, about 55% right on new films, listed under `predicted` in the tag file) until a proper pass replaces them. The workflow commits to `main` with the workflow token, which does not trigger CI, so it starts `ci.yml` itself (`workflow_dispatch`); `ci.yml` deploys on any event that is not a pull request. **A grown corpus invalidates benchmark comparisons until the new films are judged** (`make eval-pool`). See `docs/corpus-growth.md`.

The build checkpoints to `data/.corpus_build_state.json` after every stage and every 50 records; `--resume` skips completed work. Embeddings are keyed on a **content hash** of the embedded text (`corpus_fingerprint()`), so editing plot text invalidates the cache even when the film count is unchanged.

**One film per franchise.** Each corpus record carries `collection_id`/`collection` from TMDB's
`belongs_to_collection` (253 of 500 films are in one). Both strategies drop all but the best-ranked
entry of a collection (`first_of_each_collection()`): semantic on the retrieval order, unified on
the blended score, before MMR. MMR alone does not do this -- sequels are near-identical in
embedding space and a relevant one still wins a slot beside its sibling. Measured before the rule:
20% of semantic pages and 12% of unified pages repeated a franchise. The hand-written gold set
lists several entries of some franchises, so this costs a little NDCG (0.4909 -> 0.4770 unified,
inside the noise). `scripts/add_collections.py` backfills the field without a corpus rebuild.

**Where to watch** is a second committed data file, `data/watch_providers.json` (TMDB watch
providers, data by JustWatch; eight regions). It is a **snapshot**: `make providers` refreshes it
and the file must be re-committed, otherwise it silently goes stale. The modal shows the snapshot
date and the JustWatch credit, which TMDB's terms require. It is served by `/api/providers/{id}`
from `services/providers.py` -- a dictionary lookup, no network call -- and is deliberately kept
out of the corpus so neither the page payload nor the embedded text grows.

### Everything is deliberately non-deterministic

Two randomness sources, both intentional: Gaussian score perturbation scaled to the pool's score spread (`corpus.py`, `unified_recommender.py`) and a ±0.08 jitter on the MMR lambda. A third, drawing the page at random from the top 18 films (`recommender.py`), was **removed**: measured as served it cost a third of the page's quality, and it also corrupted the unified strategy's candidate pool. **Measure through the request path** (`graded_eval.py score` has "as served" arms); a number taken at `temperature=0` skips whatever only runs with randomness on. **Never write a test asserting a specific movie ordering** — assert on shape, count, and membership instead. `temperature=0` is the deterministic escape hatch and is now threaded through `semantic_search()`, `recommend_movies_advanced()` and `recommend_unified_semantic()` — all three take `seed`/`temperature`, and `temperature=0` disables sampling and MMR jitter as well as score noise. `make eval` relies on this.

### Config and DB

`settings.py` is a Pydantic `BaseSettings` behind `@lru_cache`. Tests that change env vars must call `get_settings.cache_clear()` — see `tests/test_recommender_omdb.py`.

`db.py` normalises `postgres://` URLs to the `postgresql+psycopg` dialect and passes SSL/keepalive via `connect_args` rather than the URL query string (putting `sslmode` in the query is unreliable with this dialect). `init_db()` retries with backoff but never raises — the app boots even with the DB down.

## Repo state worth knowing

- `data/film_tags.json` is committed too (see "Descriptive tags" above).
- `data/horror_corpus.json` **is committed** (and grows; 650 films at the time of writing) and copied into the Docker image, so deploys need no TMDB/OMDb keys at build time. Embeddings (`.npy`) and the build checkpoint stay gitignored and are regenerated from the corpus on first use.
- Filters are applied to the corpus **before** ranking (`_passes_filters` in `recommender.py`), keeping corpus rows and embedding rows aligned. Filtering the top-K afterwards used to starve restrictive queries.
- **Accounts behave differently once an email provider is configured** (`RESEND_API_KEY` + `EMAIL_FROM`; `email_verification.verification_enabled()`). With one: registration always redirects to `/auth/check-email` and never signs in, an existing address is told by email, and a password sign-in is refused until the address is confirmed. Without one: registration signs in at once and still says "Email already registered". The second mode is a known leak, kept because no neutral answer is honest without email. Tests cover both; use the `outbox` fixture for the first.
- **Sign in with Google** (`google_auth.py`) is off unless `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` are set. Accounts are matched on Google's `sub`, then on a Google-verified address. An account created this way has `password_hash == "!"` (`security.NO_PASSWORD`): check `has_password()` before asking for a current password. Linking to an account whose address was never confirmed discards that account's password first. Tests stub `google_auth.fetch_profile`; nothing calls Google.
- Every state-changing route requires the session's CSRF token (`require_csrf` in `security.py`), as a hidden `csrf` field or an `X-CSRF-Token` header. One token per session, read by page scripts from `<meta name="csrf-token">`. The test `client` fixture sends it automatically; use a plain `TestClient` to test the check itself.
- `scripts/manual/` holds hand-run scripts (`manual_*.py`, `debug_stripe.py`, `deployment_checklist.py`) that hit real services. They predate the Hetzner deploy and are not maintained; `tests/` contains only pytest files.
- No test touches the network (the request path has no outbound calls left); `conftest.py` forces `DEBUG=true` so session cookies aren't `Secure` (TestClient speaks plain HTTP) and swaps the DB for in-memory SQLite.

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

**Measured and deliberately left off** (`docs/evaluation-baseline.md`): cross-encoder reranking
gained +0.0026 NDCG while *losing* precision and adding ~1.1s per query. (Tuning the weights of
the original four signals also gained nothing, twice; the current weights were fitted only once a
new signal existed.) Both are wired to settings (`UNIFIED_WEIGHTS`,
`UNIFIED_USE_CROSS_ENCODER`) so they stay testable. **Do not enable either by citing the
notebooks** — `notebooks/3-cross-encoder.py` measured its gain on the old 21-film corpus.

CI replays a frozen slice of the eval without the model (`tests/test_eval_gate.py`, fixture in
`tests/fixtures/`): 236 films and the 8 dev-split moods, with precomputed vectors. It fails if
NDCG@6 drops more than 0.10 below the recorded baseline, **and** if the text that gets embedded
changes for any fixture film -- so editing `embedding_text()` or rebuilding the corpus will turn
it red on purpose. Rebuild with `make eval-gate` (needs the model) and commit the fixture with the
change. It guards against a collapse; it cannot show an improvement.

There is a second, wider benchmark: `make eval-graded` scores 55 moods against pooled, graded
judgments (`eval/judgments.json`, graded 0-2 **by an AI judge, not a person**). Its paired interval
is about ±0.034 NDCG@6, against ±0.06 for the 15 hand-written moods, so prefer it for comparing
rankers. A ranker whose results were never judged shows **holes**, which score 0: run
`make eval-pool`, judge the listed films, add them to the file, then score. Do not read a score
with holes as a measurement.

Before claiming any ranking change helps, run `make eval` — and remember a single run is
not a measurement, since the pipeline's stddev is comparable to most effects worth finding.

## Interface

One stylesheet, `app/static/styles.css`, built on the custom properties at its top (`--bg`, `--surface`, `--text`, `--red`, `--serif`, ...). **Use the tokens, never a literal colour**, and reuse the existing classes before adding one. Headlines are Instrument Serif, everything else Inter; both are self-hosted in `app/static/fonts/` and nothing loads from another origin. The nav, the account menu and the footer live in `_base.html` and read `user` from the template context: **a page that does not pass `user` shows "Log in" to a signed-in visitor** (`tests/test_ui_shell.py`). The film grid is `.grid` > `.card` > `.card-media` + `.card-body`, six columns down to two. Under `prefers-reduced-motion` transitions are set to `none`, not to a tiny duration: a near-zero `transition: all` on every element also transitions inherited `visibility` and left the film dialog blank. The home page's shelf links each poster to a `like <title>` search (`_catalogue()` in `main.py`), so it only offers films the title detector recognises.

## Adding a strategy

Three coordinated edits: the ranking function in `services/`, a branch plus a `STRATEGY_LABELS` entry in `ui_recommendations()` in `app/main.py`, and the `<option>` in `index.html`. It must search the corpus — nothing on the request path may call OMDb or TMDB.

## Deployment

Production is one Hetzner box running `docker-compose.prod.yml`: the app, Postgres and Caddy
(automatic HTTPS). **Pushing to `main` deploys it** — `.github/workflows/ci.yml` runs
`build-test` → `image` (build, push to `ghcr.io/mbottoni/terror-reco:<sha>`) → `deploy`
(scp the compose file and Caddyfile, `docker compose pull && up -d`, then poll `/healthz`).
A red `build-test` never ships; a deploy that leaves `/healthz` unhealthy fails the run.

The server holds exactly three files in `/opt/terror-reco`: the compose file, the Caddyfile
and `.env.prod`. **`.env.prod` is the only thing not in git** — secrets, `SITE_ADDRESS`
(the domain Caddy serves) and `DATABASE_SSLMODE=disable` (the Postgres container has no TLS;
the default `require` is for managed Postgres). A new setting that production needs must be
added there by hand; the pipeline does not manage it. Two are enforced at startup when `DEBUG` is off:
a real `SECRET_KEY` and `PUBLIC_BASE_URL` (the site's origin; every absolute URL the app emits
is built from it, never from the request's `Host` header). **When the domain changes, update
`SITE_ADDRESS`, `PUBLIC_BASE_URL` and the `DEPLOY_URL` Actions variable together.**

On Linux, torch is locked to the CPU-only build: `pyproject.toml` declares it as a direct
dependency sourced from the PyTorch CPU index (`[tool.uv.sources]`), held to one minor version so
Linux and macOS run the same torch. The default PyPI wheel pulls in about 9 GB of CUDA libraries
that a server with no GPU never loads. The Dockerfile's `TORCH_VERSION` must match the lockfile.
Embeddings are computed at image build, so a fresh container never re-encodes the corpus on
a user's first request. That build step copies in only `data/horror_corpus.json`,
`services/corpus.py` and `services/unified_recommender.py` first, so its layer is reused unless
one of those three changes. **Keep those two modules free of imports from the rest of the app**,
or the image build fails.
