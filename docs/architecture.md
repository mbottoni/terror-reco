# Architecture

How TerrorReco is put together, and how a query becomes a page of films.

## System overview

TerrorReco is a server-rendered FastAPI application. The browser submits plain
forms and follows plain links; the server ranks films from a local corpus and
returns finished HTML. A small amount of JavaScript drives the film detail
view, feedback, the watchlist and usage logging. Nothing on the request path
calls an external API.

```
Browser
   |  GET /recommend?mood=...&strategy=...&filters
   v
Caddy (TLS)  -->  FastAPI (app/main.py)
   |
   +-- Middleware: security headers, per-client rate limits, sessions
   +-- Auth and accounts (auth.py, email_verification.py, google_auth.py,
   |                      account.py, password_reset.py, onboarding.py)
   +-- CSRF check on every state-changing route (security.py)
   |
   +-- Strategy router (ui_recommendations)
   |      +-- "unified"  --> recommend_movies_advanced() --> recommend_unified_semantic()   (default)
   |      +-- "semantic" --> recommend_movies_advanced()
   |      +-- ids=...    --> films_by_ids()            (a shared link: no ranking)
   |
   +-- Does the query name a film? (title_reference.py, on the whole corpus)
   +-- Filters before ranking (year range, minimum IMDb rating, English only)
   +-- Unified only: read the query for qualities, exclusions, decades, people
   |   (query_understanding.py), widen the pool, blend the signals, MMR
   +-- One film per franchise; the page is the top of the ranking, with score noise
   |
   v
results.html
   +-- Refine bar (search, removable filter chips, shuffle)
   +-- Film cards, detail dialog, "where to watch", "more like this"
   +-- Feedback: per film (signed in) and per set (anyone)
```

## Components

### `app/main.py`: application core

Creates the app, registers middleware and routers, and holds the recommendation
routes: `/` (search form and a shelf of well-known films), `/recommend`
(results), `/privacy`, `/api/recommendations`,
`/api/similar/{imdb_id}`, `/api/providers/{imdb_id}`, `/api/feedback`,
`/api/set-feedback`, `/api/interaction` and `/healthz`. Startup runs in a
`lifespan` handler: migrate the database, load the model, warm the corpus.

`ui_recommendations()` validates `strategy` against `STRATEGY_LABELS`; an
unknown value falls back to the default. Ranking is CPU-bound, so it runs in
the threadpool, never on the event loop.

### Accounts and security

| Module | What it does |
|--------|--------------|
| `auth.py` | Register, sign in, sign out, sign out everywhere. Emails are normalised. Sessions carry a per-user version, so they can be revoked |
| `account.py` | Change password or email, export data, delete the account |
| `password_reset.py` | Reset by emailed, single-use, hashed token. Off until an email provider is configured |
| `email_verification.py` | Confirm an address by emailed link; registration that answers the same for new and existing addresses. Off until an email provider is configured |
| `google_auth.py` | Sign in with Google (OpenID Connect code flow with PKCE). Off until a client id and secret are configured |
| `onboarding.py` | Pick films you love after signing up, which seeds personalisation |
| `security.py` | Argon2 hashing, the session CSRF token and its check, URL scheme checks |
| `middleware.py` | Security response headers and in-process, per-client rate limits |

Sessions are signed cookies (Starlette `SessionMiddleware`). Every route that
changes state requires the session's CSRF token, sent as a hidden form field
or an `X-CSRF-Token` header. The Stripe webhook is the exception: it is
authenticated by signature.

### `app/models.py`: data

`User`, `SearchHistory` and `SearchResult` (what was shown, in rank order),
`MovieFeedback` (per-film like or dislike), `WatchlistItem`, `PublicList` (a
published watchlist), `Payment`, `PasswordResetToken`,
`EmailVerificationToken`, and three tables of anonymous usage data: `SetFeedback`
(a verdict on a whole result set), `Impression` (a set that was shown) and
`Interaction` (a film opened, clicked through to, or saved). See
[database.md](database.md).

### `app/db.py` and migrations

Connection handling for SQLite (development) and PostgreSQL (production).
`init_db()` runs `alembic upgrade head` at startup and never raises, so the
app boots even if the database is down. Schema changes go through Alembic.

### `app/settings.py`: configuration

One Pydantic settings class, read from the environment. With `DEBUG` off it
refuses to start without a real `SECRET_KEY` and a `PUBLIC_BASE_URL`.

### `app/services/corpus.py`: corpus and semantic search

No network I/O. The corpus is built offline.

- `load_corpus()` / `save_corpus()` read and write `data/horror_corpus.json`.
  Loading attaches each film's descriptive tags from `data/film_tags.json`;
  saving strips them again.
- `embedding_text()` defines the text that is embedded: title, genre, keywords
  and overview. Anything computing a semantic score must use it.
- `mix_keyword_vectors()` makes the stored vector: 70% the vector of that
  text, 30% the vector of the film's tags and keywords encoded alone.
- `get_corpus_embeddings()` computes or loads the embeddings, cached and keyed
  on a hash of the embedded text and the recipe.
- `resolve_query()` turns a query into the vector to search with: the query's
  own embedding, or a named film's stored vector.
- `semantic_search()` takes cosine similarity against the corpus, optionally
  adds noise, and returns the top K.

### `app/services/recommender.py`: the semantic strategy

`recommend_movies_advanced()` resolves the query against the whole corpus,
filters the corpus, runs semantic search, keeps one film per franchise and
returns the top of that ranking. Repeat searches vary through score noise and
by demoting films just shown. It also provides `similar_movies()`,
`films_by_ids()` (shared links), `explain_match()` (the tag and keyword chips
on each card) and `films_without_each_filter()` (the empty-state explanation).

### `app/services/unified_recommender.py`: the unified strategy

`recommend_unified_semantic()` takes a 60-film pool from the above and re-ranks
it by a blend of signals whose weights were fitted on the graded benchmark:

- tags (0.35): how well the film's descriptive tags answer the query,
- semantic similarity (0.30),
- BM25 keyword match (0.25), with document frequencies taken from the whole corpus,
- popularity (0.10): IMDb rating × log of votes,
- recency (0, kept so a setting can turn it on).

A signed-in user with enough likes adds their taste vector. A film that breaks
something the query states (an excluded quality, a decade, a named person)
loses a fixed penalty. It then keeps one film per franchise and applies
Maximal Marginal Relevance for variety. See
[recommendation-engine.md](recommendation-engine.md).

### `app/services/title_reference.py`: queries that name a film

Detects a corpus title in a query ("like Hereditary but less bleak"), so the
search can use that film's vector and leave the film and its franchise out.
Detection is conservative: a wrongly detected title would replace a mood
search with "films like X".

### `app/services/query_understanding.py`: reading the query

Rule-based. Picks out wanted and excluded tags, a decade, and people named in
full, which the unified strategy uses to widen its candidate pool, to score
tag coverage and to penalise violations. Anything it does not recognise is
ignored.

These two modules, `corpus.py` and `unified_recommender.py` import only one
another, and only inside functions, never the rest of the app: the image build
computes embeddings with just `corpus.py` and `unified_recommender.py` copied
in.

### Other services

| Module | What it does |
|--------|--------------|
| `services/personalization.py` | Taste vector from liked films, demotion of disliked ones, "because you liked" |
| `services/providers.py` | Where-to-watch lookups from the `data/watch_providers.json` snapshot |
| `services/email.py` | Transactional email through Resend; a logged no-op when unconfigured |
| `services/tmdb_client.py`, `services/omdb_client.py` | API clients used only by the offline scripts |
| `watchlist.py` | Saved films, and publishing the list at a random public address (`/l/{slug}`) |
| `stripe_payments.py` | "Buy me a coffee" checkout with a choice of amount; the webhook records payments and refunds |
| `static_assets.py` | Fingerprinted static URLs with long cache headers |
| `observability.py` | Application logging and optional Sentry error tracking |

### Offline scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `build_corpus.py` | Build the 500-film base corpus from TMDB (discovery) and OMDb (IMDb ratings). Resumable |
| `grow_corpus.py` | Add the next horror films the corpus lacks, without a rebuild. Runs daily in GitHub Actions |
| `tag_predictor.py` | Predict descriptive tags for newly added films from their tagged neighbours |
| `add_collections.py` | Backfill each film's TMDB franchise without a rebuild |
| `fetch_watch_providers.py` | Refresh the where-to-watch snapshot |
| `run_eval.py`, `tune_weights.py` | Score the ranker against the hand-written gold set; search blend weights |
| `graded_eval.py` | The graded benchmark: pool films to judge, merge grades, score moods, title queries and constraint queries |
| `eval_gate.py` | Rebuild the frozen eval slice that CI replays |
| `export_eval_set.py` | Turn feedback and usage data into graded eval cases |
| `download_model.py` | Cache the sentence-transformer model in `models/` for local development |

The base build discovers films on TMDB (`/discover/movie?with_genres=27`,
sorted by vote count) because OMDb cannot browse a genre. Stages: seed the
eval gold set, most-voted films, a decade-stratified fill, TMDB detail, OMDb
enrichment, validation. Progress is checkpointed, so an interrupted build
resumes. Since then the corpus grows through `grow_corpus.py`; see
[corpus-growth.md](corpus-growth.md).

### Templates and static files

Jinja2 templates extend `_base.html`, which holds the nav, the account menu
and the footer. One stylesheet, built on design tokens, is served from a
fingerprinted URL; the Inter and Instrument Serif fonts are self-hosted.

| Template | Route | Purpose |
|----------|-------|---------|
| `index.html` | `GET /` | Search form, strategy selector, filters, a shelf of films to start from |
| `results.html` | `GET /recommend` | Refine bar, film grid, detail dialog, feedback |
| `login.html`, `register.html` | `/auth/login`, `/auth/register` | Sign in and sign up |
| `check_email.html` | `/auth/check-email` | "Check your inbox" after sign-up |
| `forgot.html`, `reset.html` | `/auth/forgot`, `/auth/reset/{token}` | Password reset |
| `onboarding.html` | `GET /onboarding/` | Pick films you love |
| `account.html` | `GET /account/` | Password, email, data export, deletion |
| `history.html`, `watchlist.html` | `/history/`, `/watchlist/` | Past searches; saved films |
| `public_list.html` | `GET /l/{slug}` | A published watchlist, read-only |
| `privacy.html` | `GET /privacy` | What is stored and how to export or delete it |
| `coffee*.html` | `/stripe/...` | Donation page and its outcomes |
| `error.html` | (errors) | 404, 429 and failure pages |

## From query to results

1. The search form submits `GET /recommend` with the mood, strategy and filters.
2. Rate limiting and the session middleware run; the handler loads the user.
3. The query is checked for a named film, filters are applied to the corpus,
   then the chosen strategy ranks what is left. The unified strategy also
   reads the query for qualities, exclusions, a decade and people.
4. One film per franchise is kept and the page is the top of the ranking.
   With `seed` set the request runs at `temperature=0` and is reproducible,
   which the eval relies on.
5. For a signed-in user the search is saved to history; for any real visitor
   an anonymous impression is recorded (preview crawlers are skipped).
6. `results.html` renders the cards, the refine bar and the feedback control.
7. Opening a card shows the detail dialog, which fetches "where to watch" and
   "more like this" from the API.

If filters leave nothing, the page says which filter is responsible and how
many films removing it would allow.

## Persistence

| Store | Format | Purpose |
|-------|--------|---------|
| SQLite / PostgreSQL | Relational | Accounts, history, feedback, watchlist, payments, usage data |
| `data/horror_corpus.json` | JSON, committed | The film corpus (650 films at the time of writing, and growing) |
| `data/film_tags.json` | JSON, committed | The 140-tag vocabulary with definitions, and each film's tags |
| `data/watch_providers.json` | JSON, committed | Where-to-watch snapshot (refresh with `make providers`) |
| `data/corpus_growth.json` | JSON, committed | Films the growth script declined, and a log of its runs |
| `data/corpus_embeddings.npy` and `.meta.json` | NumPy, gitignored | Embeddings and the hash they were built from |
| `data/.corpus_build_state.json` | JSON, gitignored | Corpus build checkpoint |
| `eval/moods.json`, `eval/judgments.json` | JSON, committed | Benchmark queries and their graded films |
| `tests/fixtures/eval_gate.*` | JSON and NumPy, committed | Frozen eval slice replayed in CI |

The corpus, the tags and the provider snapshot ship in the Docker image. Embeddings are
computed when the image is built, so a new container never encodes the corpus
while a user waits. A recommendation requested with no corpus raises
`CorpusNotBuiltError` instead of crawling: an in-request crawl that was
rate-limited is what once froze the corpus at 21 films.

## Deployment

One server runs the app, PostgreSQL, Caddy and a nightly backup job with
Docker Compose. Pushing to `main` builds, tests, smoke-tests and deploys the
image. Two scheduled workflows run beside that: one grows the corpus daily and
starts a deploy, one checks `/healthz` every five minutes. See
[deployment.md](deployment.md).
