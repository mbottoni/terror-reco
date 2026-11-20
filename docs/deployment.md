# Deployment

This guide covers deploying TerrorReco with Docker, on Render, and the CI/CD pipeline.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OMDB_API_KEY` | Yes | -- | OMDb API key ([get one free](https://www.omdbapi.com/apikey.aspx)) |
| `SECRET_KEY` | Production | -- | Signs sessions and CSRF tokens. The app refuses to start in production without a random value of at least 32 characters |
| `PUBLIC_BASE_URL` | Production | -- | The site's origin, e.g. `https://example.com`. Absolute URLs (Stripe return URLs) are built from it. The app refuses to start in production without it |
| `DATABASE_URL` | Production | SQLite file | PostgreSQL connection URL |
| `DEBUG` | No | `false` | Set to `1` or `true` for development mode |
| `PORT` | No | `10000` | Port the server listens on |
| `LOG_LEVEL` | No | `INFO` | Level for the app's own log lines |
| `SENTRY_DSN` | Optional | -- | Turns on error tracking. Unset means off. No request bodies, cookies or user details are sent |
| `STRIPE_PUBLISHABLE_KEY` | Optional | -- | Stripe public key (for payments) |
| `STRIPE_SECRET_KEY` | Optional | -- | Stripe secret key |
| `STRIPE_WEBHOOK_SECRET` | Optional | -- | Stripe webhook signing secret |
| `COFFEE_PRICE_ID` | Optional | -- | Stripe Price ID for the coffee product |

### Creating a `.env` file

```bash
OMDB_API_KEY=your_key_here
SECRET_KEY=a-long-random-string
DEBUG=1
```

## Production (Hetzner, deployed from GitHub)

Production is a single Hetzner Cloud server running `docker-compose.prod.yml`:
the app, Postgres 17 and Caddy, which obtains and renews the HTTPS certificate.

**Deploying is `git push origin main`.** The CI workflow then:

1. `build-test` — lint, type-check, tests.
2. `image` — builds the Docker image and pushes it to
   `ghcr.io/mbottoni/terror-reco:<commit sha>` (and `:latest`).
3. `deploy` — copies `docker-compose.prod.yml` and `Caddyfile` to the server,
   pulls the new image, restarts the stack, and fails the run unless
   `/healthz` reports `ok`.

**Rolling back, or deploying without a commit:** Actions → *Deploy a chosen
version* → *Run workflow*, and enter the full commit sha of a green run on
`main` (or `gh workflow run deploy.yml -f image_tag=<sha>`). It runs the image
CI already built for that commit, with that commit's compose file, and applies
the same health check. Images from the last three days stay on the server, so
a recent rollback does not need to download anything.

It does not roll the database back: migrations only run forward. An older
image on a newer schema usually works, but check the migrations between the
two commits before rolling back across one that removed or renamed something.

### What lives where

| Thing | Location |
|-------|----------|
| Code, compose file, Caddyfile | this repository |
| `DEPLOY_HOST`, `DEPLOY_SSH_KEY`, `DEPLOY_KNOWN_HOSTS` | GitHub Actions secrets |
| `DEPLOY_URL` (health-check base URL) | GitHub Actions variable |
| App secrets, `SITE_ADDRESS`, DB password | `/opt/terror-reco/.env.prod` on the server |
| Database | `terror-reco_pgdata` Docker volume on the server |

### Backups

The `backup` service (`ops/backup.sh`) runs `pg_dump` every night at 03:00 UTC
into `/opt/terror-reco/backups/` on the server, keeps the newest 14, and takes
one at start-up if there is none from today. Each dump is checked with
`pg_restore --list` before it is kept.

**These dumps are on the same server as the database.** They cover a bad
migration or a mistaken delete, not the loss of the server. For an off-box
copy, create a bucket on S3-compatible storage (Cloudflare R2's free tier is
enough) and add to `.env.prod`:

```bash
BACKUP_S3_BUCKET=terror-reco-backups
BACKUP_S3_ENDPOINT=https://<account id>.r2.cloudflarestorage.com
BACKUP_S3_ACCESS_KEY_ID=...
BACKUP_S3_SECRET_ACCESS_KEY=...
# BACKUP_S3_PROVIDER=Cloudflare   # or AWS, Backblaze, Other
```

then `docker compose -f docker-compose.prod.yml --env-file .env.prod up -d backup`.
Every later dump is uploaded too, and copies older than 60 days are removed.

Commands, run on the server from `/opt/terror-reco`:

```bash
C="docker compose -f docker-compose.prod.yml --env-file .env.prod"
$C exec backup sh /backup.sh once      # take a dump now
$C exec backup sh /backup.sh verify    # restore the newest dump into a scratch DB and compare
$C logs backup                         # what ran, and any UPLOAD FAILED / BACKUP FAILED
```

**To restore for real** (replaces the live data; stop the app first):

```bash
$C stop app
$C exec backup sh -c 'dropdb terror && createdb terror && pg_restore --no-owner -d terror /backups/<file>.dump'
$C start app
```

### Setting up a new server

1. Ubuntu 24.04 with Docker; firewall open on 22, 80, 443 only.
2. A `deploy` user in the `docker` group whose `authorized_keys` holds the
   public half of `DEPLOY_SSH_KEY`, owning `/opt/terror-reco`.
3. `/opt/terror-reco/.env.prod` (mode 600) with at least:

   ```bash
   SITE_ADDRESS=example.com
   DEBUG=false
   SECRET_KEY=<long random string>
   PUBLIC_BASE_URL=https://example.com
   POSTGRES_PASSWORD=<random>
   DATABASE_URL=postgresql://terror:<same password>@db:5432/terror
   DATABASE_SSLMODE=disable
   OMDB_API_KEY=...
   # plus the Stripe and TMDB keys from the table above
   ```

4. Point the domain's `A` record at the server, update the three `DEPLOY_*`
   secrets and `DEPLOY_URL`, and push to `main`.

## Docker

### Build

```bash
docker build -t terror-reco:latest .
```

The Dockerfile:

1. Starts from `python:3.11-slim`.
2. Installs system deps (build-essential, ca-certificates, libpq-dev for PostgreSQL SSL).
3. Installs Python dependencies from `pyproject.toml`.
4. **Pre-downloads the sentence-transformer model** (`all-mpnet-base-v2`) so there's no cold-start download at runtime.
5. Copies the application source.
6. Exposes port 10000 (Render convention).
7. Runs with `uvicorn`.

### Run

```bash
# With env file
docker run --env-file .env -p 8000:10000 terror-reco:latest

# Or with docker-compose (auto-loads .env)
docker compose up --build
```

### Docker Compose

The `docker-compose.yml` mounts `./app` as a read-only volume for live development, maps port 8000, and loads environment from `.env`.

## Render Deployment

TerrorReco is designed for [Render](https://render.com):

1. **Create a Web Service** pointing to your GitHub repo.
2. **Runtime:** Docker.
3. **Environment variables:** Set `OMDB_API_KEY`, `SECRET_KEY`, `DATABASE_URL` (Render provides a PostgreSQL instance).
4. **Port:** The Dockerfile defaults to 10000 which is Render's expected port.

### Render PostgreSQL

1. Create a PostgreSQL instance on Render.
2. Copy the Internal Database URL.
3. Set it as `DATABASE_URL` in your web service's environment.

The app automatically handles:
- Converting `postgres://` to `postgresql+psycopg://`.
- Adding SSL parameters for secure connections.

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push and PR to `main`:

| Step | Command | What it checks |
|------|---------|---------------|
| Lint | `ruff check .` | Code style, imports, common errors |
| Type-check | `mypy app` | Static type checking (strict mode) |
| Tests | `pytest` | Unit tests with mocked external services |
| Docker build | `docker build -t terror-reco:ci .` | Ensures the image builds successfully |

### Running CI Locally

```bash
make ci
# or individually:
make lint
make typecheck
make test
```

For the full pipeline including Docker:

```bash
make ci && docker build -t terror-reco:ci .
```

## First-Run Behaviour

On the first request using the Semantic or Unified strategy, the app will:

1. **Build the horror corpus** by querying OMDb (takes a few minutes, requires API quota).
2. **Compute embeddings** for all corpus plots using the sentence-transformer model.
3. **Cache both** to `data/horror_corpus.json` and `data/corpus_embeddings.npy`.

Subsequent requests use the cached data and are fast (numpy dot-product).

### Pre-building the Corpus

To avoid the first-request delay, you can build the corpus ahead of time:

```python
import asyncio
from app.services.corpus import build_corpus

asyncio.run(build_corpus(pages=5))
```

### Pre-downloading the ML Model

For local development (without Docker):

```bash
python scripts/download_model.py
```

This downloads `all-mpnet-base-v2` to the `models/` directory.

## Production Checklist

- [ ] Set `DEBUG=false` (enables secure cookies, disables debug pages)
- [ ] Set a strong `SECRET_KEY` (don't rely on auto-generation)
- [ ] Configure `DATABASE_URL` for PostgreSQL
- [ ] Set `OMDB_API_KEY` with sufficient quota
- [ ] Build the horror corpus before first user request
- [ ] (Optional) Configure Stripe keys for payment support
- [ ] (Optional) Set up Stripe webhooks pointing to `https://yourdomain.com/stripe/webhook`
