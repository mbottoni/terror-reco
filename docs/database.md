# Database

TerrorReco uses SQLAlchemy ORM for database access. In development it defaults to SQLite; in production it connects to PostgreSQL (e.g. on Render).

## Connection Management

Defined in `app/db.py`:

- **URL normalisation** -- converts `postgres://` URLs (common in PaaS) to `postgresql+psycopg://` for SQLAlchemy.
- **SSL handling** -- adds `sslmode=require` for PostgreSQL connections.
- **Retry logic** -- `init_db()` retries table creation up to 3 times with exponential backoff.
- **Session dependency** -- `get_db()` is a FastAPI dependency that yields a session and auto-closes it.

## Models

All models are defined in `app/models.py` and inherit from `Base` (declared in `app/db.py`).

### User

Stores registered user accounts.

| Column | Type | Notes |
|--------|------|-------|
| `id` | Integer | Primary key, auto-increment |
| `email` | String(256) | Unique, indexed |
| `password_hash` | String(512) | Argon2 hash |
| `created_at` | DateTime | Default: `utcnow()` |

**Relationships:**
- `searches` -> `SearchHistory` (one-to-many)
- `feedback` -> `MovieFeedback` (one-to-many)

### SearchHistory

Records each recommendation search a user performs.

| Column | Type | Notes |
|--------|------|-------|
| `id` | Integer | Primary key, auto-increment |
| `user_id` | Integer | Foreign key -> `users.id`, indexed |
| `mood` | String(512) | The user's query text |
| `strategy` | String(64) | Which strategy was used |
| `results_json` | JSON (`dict[str, Any]`) | The full results payload |
| `created_at` | DateTime | Default: `utcnow()` |

**Relationships:**
- `user` -> `User` (many-to-one)

### MovieFeedback

Stores per-user per-movie like/dislike ratings.

| Column | Type | Notes |
|--------|------|-------|
| `id` | Integer | Primary key, auto-increment |
| `user_id` | Integer | Foreign key -> `users.id`, indexed |
| `imdb_id` | String(20) | IMDb ID, indexed |
| `title` | String(512) | Movie title (denormalised for convenience) |
| `rating` | Integer | `+1` (like) or `-1` (dislike) |
| `mood` | String(512) | Nullable -- the query that prompted this |
| `strategy` | String(64) | Nullable -- which strategy was active |
| `created_at` | DateTime | Default: `utcnow()` |
| `updated_at` | DateTime | Default: `utcnow()`, auto-updated on change |

**Constraints:**
- Unique constraint on `(user_id, imdb_id)` -- each user can rate each movie once.

**Relationships:**
- `user` -> `User` (many-to-one)

## Schema Diagram

```
users
  id (PK)
  email (UNIQUE)
  password_hash
  created_at
      |
      |--- 1:N ---> search_history
      |                id (PK)
      |                user_id (FK)
      |                mood
      |                strategy
      |                results_json
      |                created_at
      |
      |--- 1:N ---> movie_feedback
                       id (PK)
                       user_id (FK)
                       imdb_id
                       title
                       rating
                       mood
                       strategy
                       created_at
                       updated_at
                       UNIQUE(user_id, imdb_id)
```

## Table Creation

Tables are created automatically on application startup via `init_db()` called in the `@app.on_event("startup")` handler. SQLAlchemy's `Base.metadata.create_all()` is used, which is safe to call multiple times (it only creates missing tables).

## Switching Databases

| Environment | Database | Configuration |
|-------------|----------|---------------|
| Development | SQLite | Default when `DATABASE_URL` is not set (`sqlite:///./terror_reco.db`) |
| Testing | SQLite (in-memory) | Set up in `tests/conftest.py` |
| Production | PostgreSQL | Set `DATABASE_URL=postgresql+psycopg://...` |

## Future: Migrations

Currently, schema changes require dropping and recreating tables. For production evolution, consider adding Alembic:

```bash
pip install alembic
alembic init alembic
alembic revision --autogenerate -m "initial"
alembic upgrade head
```
