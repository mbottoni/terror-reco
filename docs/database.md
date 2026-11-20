# Database

TerrorReco uses SQLAlchemy 2. Development defaults to a SQLite file; production
runs PostgreSQL in a container beside the app. The models are in
`app/models.py`; this page summarises them.

## Connection management

`app/db.py` builds the engine from `DATABASE_URL`.

- `postgres://` and `postgresql://` URLs are normalised to the
  `postgresql+psycopg` dialect.
- SSL is passed as a connect argument, taken from `DATABASE_SSLMODE` (default
  `require`, for managed PostgreSQL). The production container has no TLS, so
  it sets `DATABASE_SSLMODE=disable`.
- `init_db()` runs the Alembic migrations at startup, retrying with backoff.
  It never raises: the app boots even with the database down, and `/healthz`
  still answers.
- `get_db()` is the FastAPI dependency that yields a session per request.

## Tables

### Accounts

**`users`**

| Column | Notes |
|--------|-------|
| `id` | Primary key |
| `email` | Unique. Stored trimmed and lowercased |
| `password_hash` | Argon2 |
| `session_version` | Copied into the session at sign-in and checked on every request. Bumping it signs the user out everywhere |
| `email_verified_at` | When the owner confirmed they can read mail at `email`. Null until then, and again after the address changes. Accounts older than this column were marked confirmed |
| `google_sub` | Google's stable id for the account this user signs in with, if any. Unique |
| `created_at` | |

`password_hash` holds `!` for an account that has no password (created by signing in with Google); nothing verifies against it.

**`email_verification_tokens`** — one row per emailed confirmation link: `user_id`, `token_hash` (SHA-256), `email` (the address it was sent to, so it cannot confirm a later one), `expires_at` (24 hours), `used_at`.

**`password_reset_tokens`** — one row per emailed reset link.

| Column | Notes |
|--------|-------|
| `user_id` | Deleted with the user |
| `token_hash` | SHA-256 of the token. The token itself is never stored |
| `expires_at`, `used_at` | Links last 30 minutes and work once |

### What a signed-in user did

**`search_history`** — one row per search: `user_id`, `mood`, `strategy`,
`created_at`. (`results_json` is a legacy column, no longer written.)

**`search_results`** — the films that search returned, in rank order:
`search_id`, `position`, `imdb_id`, `title`. Deleted with the search.

**`movie_feedback`** — a like (`rating = 1`) or dislike (`-1`) per user and
film, unique on `(user_id, imdb_id)`, with the `mood` and `strategy` it was
given under. Feeds personalisation. Picks made at onboarding are rows here
with `strategy = "onboarding"` and no mood.

**`watchlist`** — films saved to watch later: `user_id`, `imdb_id`, `title`,
`poster_url`, unique on `(user_id, imdb_id)`. Kept apart from feedback on
purpose: saving a film says nothing about liking it.

**`public_lists`** — a watchlist published at a random address: `user_id` (one per user), `slug`, `title`, `description`. A row is the publication: unpublishing deletes it, and publishing again creates a new slug, so a withdrawn link stays dead.

### Payments

**`payments`** — one row per completed Stripe checkout, written by the webhook.

| Column | Notes |
|--------|-------|
| `checkout_session_id` | Unique, which makes repeated webhook deliveries harmless |
| `stripe_event_id`, `payment_intent_id` | |
| `amount_total`, `currency` | Amount in minor units |
| `status` | `paid` or `refunded` |
| `user_id` | Optional. Set to null, not deleted, when the account is deleted |

### Anonymous usage data

These exist to build an evaluation set from real use. They are keyed by a
random per-session `visitor_id`, with no IP address and no user agent.

**`set_feedback`** — a "good picks" (`verdict = 1`) or not (`-1`) on a whole
result set: `mood`, `strategy`, `imdb_ids` (in the order shown), `result_key`.
Unique on `(visitor_id, result_key)`. Carries an optional `user_id`, set to
null when that account is deleted.

**`impressions`** — a result set that was rendered: `result_key`, `mood`,
`strategy`, `imdb_ids`. Unique on `(visitor_id, result_key)`. No user link.
Preview crawlers and shared-link views are not recorded.

**`interactions`** — something a visitor did with one film in a set they were
shown: `result_key`, `imdb_id`, `kind` (`open`, `imdb_click`, `save`,
`similar_click`) and `position` (the rank it was shown at). One row per
visitor, set, film and kind.

`result_key` is the SHA-256 of the mood and the ordered film ids, so it
identifies "this set for this mood" across the three tables.

## Relationships

```
users ─┬─< search_history ──< search_results
       ├─< movie_feedback
       ├─< watchlist
       ├─< password_reset_tokens
       ├─< email_verification_tokens
       ├─< payments          (user_id nullable, kept on account deletion)
       └─< set_feedback      (user_id nullable, kept on account deletion)

impressions ··· interactions   (joined by result_key and visitor_id; no user)
```

Deleting an account removes the user, their history and result rows,
feedback, watchlist, reset and confirmation tokens, and unlinks their payments and set
verdicts.

## Migrations

Schema changes go through Alembic (`migrations/`).

```bash
make migration m="what changed"   # autogenerate from the models
make migrate                      # upgrade to head
make migrate-check                # fail if models and migrations have drifted
```

The app runs `alembic upgrade head` at startup, so a deploy migrates itself.
Migrations only run forward: rolling the app back to an older version does not
roll the schema back.

## Environments

| Environment | Database | Configuration |
|-------------|----------|---------------|
| Development | SQLite file | Default when `DATABASE_URL` is unset (`sqlite:///./app.db`) |
| Tests | SQLite in memory | Set up in `tests/conftest.py`; tables are created per test |
| Production | PostgreSQL 17 | `DATABASE_URL=postgresql://terror:<password>@db:5432/terror`, `DATABASE_SSLMODE=disable` |

SQLite does not enforce foreign keys by default, so behaviour that depends on
`ON DELETE` rules is also done explicitly in code (see `delete_account()`),
and was verified once against the production PostgreSQL.

## Backups

A nightly `pg_dump` with rotation and a restore check. See
[deployment.md](deployment.md#backups).
