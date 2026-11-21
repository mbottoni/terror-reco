# Deployment

How TerrorReco is deployed: one server running Docker Compose, shipped by GitHub Actions on every push to `main`.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SECRET_KEY` | Production | -- | Signs sessions and CSRF tokens. The app refuses to start in production without a random value of at least 32 characters |
| `PUBLIC_BASE_URL` | Production | -- | The site's origin, e.g. `https://example.com`. Absolute URLs (Stripe return URLs, emailed links, the Google redirect) are built from it. The app refuses to start in production without it |
| `DATABASE_URL` | Production | SQLite file | PostgreSQL connection URL |
| `DATABASE_SSLMODE` | No | `require` | `disable` for the Postgres container in the production stack, which has no TLS |
| `OMDB_API_KEY` | Corpus scripts only | -- | OMDb API key ([get one free](https://www.omdbapi.com/apikey.aspx)). IMDb ratings when building or growing the corpus. The running app never calls OMDb |
| `TMDB_BEARER_TOKEN` (or `TMDB_API_KEY`) | Corpus scripts only | -- | TMDB credentials for discovery, film detail and the where-to-watch snapshot. The running app never calls TMDB |
| `DEBUG` | No | `false` | Set to `1` or `true` for development mode |
| `LOG_LEVEL` | No | `INFO` | Level for the app's own log lines |
| `SENTRY_DSN` | Optional | -- | Turns on error tracking. Unset means off. No request bodies, cookies or user details are sent |
| `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | Optional | -- | Turn on "Sign in with Google". Create an OAuth client (type: web application) in the Google Cloud console and add `<PUBLIC_BASE_URL>/auth/google/callback` as an authorised redirect URI |
| `RESEND_API_KEY`, `EMAIL_FROM` | Optional | -- | Turn on email: password reset, address confirmation at sign-up, and a sign-up form that does not reveal which addresses have accounts. The sender's domain must be verified with Resend. **Setting these changes sign-up:** new accounts must open an emailed link before they can sign in |
| `STRIPE_PUBLISHABLE_KEY` | Optional | -- | Stripe public key (for payments) |
| `STRIPE_SECRET_KEY` | Optional | -- | Stripe secret key |
| `STRIPE_WEBHOOK_SECRET` | Optional | -- | Stripe webhook signing secret |
| `COFFEE_PRICE_ID` | Optional | -- | Stripe Price used for the default $3 donation. Other amounts are priced by the app; without this, the default is too |

Ranking switches, all optional and normally left alone (`app/settings.py`):
`USE_UNIFIED_RECOMMENDER` (default on: unified is the default strategy),
`UNIFIED_WEIGHTS` (a JSON object overriding the fitted blend),
`UNIFIED_DIVERSITY_LAMBDA` (0.7), `UNIFIED_USE_CROSS_ENCODER` (off; measured
and rejected), `PERSONALIZATION_ENABLED` (on), `PERSONALIZATION_TASTE_WEIGHT`
(0.15) and `RATE_LIMIT_ENABLED` (on).

### Creating a `.env` file

```bash
DEBUG=1
SECRET_KEY=any-value-is-fine-with-debug-on
# Only to rebuild or grow the corpus, or refresh the where-to-watch snapshot:
TMDB_BEARER_TOKEN=...
OMDB_API_KEY=...
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
| `DEPLOY_URL` (health-check base URL, also used by the uptime check) | GitHub Actions variable |
| `TMDB_BEARER_TOKEN`, `TMDB_API_KEY`, `OMDB_API_KEY` (for the daily corpus growth job) | GitHub Actions secrets |
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

### Uptime check

`.github/workflows/uptime.yml` asks production's `/healthz` for a healthy
answer every five minutes, from GitHub's network. Three failed attempts over
about a minute open an issue labelled `outage`, assigned to the repository
owner, so GitHub sends an email. While the site stays down it adds one comment
an hour. The first healthy check afterwards records the recovery time and
closes the issue, so the issue list doubles as an outage log.

It is a floor, not a full monitor: GitHub often runs scheduled workflows
several minutes late, and stops scheduling them after 60 days without a
commit. To test the alert, run the workflow by hand with a URL that fails:

```bash
gh workflow run uptime.yml -f url=https://example.invalid/healthz
```

### Daily corpus growth

`.github/workflows/grow-corpus.yml` runs `scripts/grow_corpus.py` every day at
04:17 UTC (and on demand). When it adds films it validates the corpus, runs
the whole test suite, commits the data files to `main` and starts `ci.yml`
itself, because a push made with the workflow token does not trigger other
workflows. So **a deploy can happen without anyone pushing**. A run that adds
nothing commits only its log, with no deploy. See
[corpus-growth.md](corpus-growth.md).

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
   # plus, as wanted, the Stripe, email, Google and Sentry settings from the
   # table above. The server needs no TMDB or OMDb key.
   ```

4. Point the domain's `A` record at the server, update the three `DEPLOY_*`
   secrets and `DEPLOY_URL`, and push to `main`.

## Docker

### Build

```bash
docker build -t terror-reco:latest .
```

The Dockerfile:

1. Starts from `python:3.11-slim` and installs system dependencies.
2. Installs the CPU build of PyTorch, then the locked Python dependencies from `requirements.txt` (generated from `uv.lock`), leaving out the CUDA wheels a CPU-only server never uses. The image is about 3.5 GB; with them it was 11.9 GB.
3. Switches to an unprivileged user.
4. **Downloads the sentence-transformer model** and then sets the hub to offline mode, so the container never contacts huggingface.co.
5. Copies in only the corpus, the film tags and the two modules that turn a film into a vector, and **computes the corpus embeddings**, so a new container never encodes the corpus while a user waits. Copying so little first means the layer is reused unless the corpus, the tags or those modules change.
6. Copies the rest of the app and data, and the migrations.
7. Runs `uvicorn` on `$PORT` (10000 by default; the production compose file runs it on 8000 behind Caddy).

CI boots every image it builds with no network and production settings, and requires `/healthz` to report `ok`, before the image can be pushed.

### Run

```bash
# With env file
docker run --env-file .env -p 8000:10000 terror-reco:latest

# Or with docker-compose (auto-loads .env)
docker compose up --build
```

### Docker Compose

The `docker-compose.yml` mounts `./app` as a read-only volume for live development, maps port 8000, and loads environment from `.env`.

## CI/CD Pipeline

`.github/workflows/ci.yml` runs on every push and pull request to `main`, and when the corpus growth workflow starts it:

| Job | What it does |
|-----|--------------|
| `build-test` | Installs from the lockfile and checks it is current, then `ruff`, `black --check`, `mypy app` and `pytest` (which includes the ranking regression gate) |
| `image` | Builds the Docker image, boots it offline with production settings and checks `/healthz`. On a push to `main`, pushes it to GHCR |
| `deploy` | Any run on `main` that is not a pull request: ships that image to the server and fails unless the site reports healthy |

The other workflows:

| Workflow | What it does |
|----------|--------------|
| `deploy.yml` | The manual deploy and rollback described above. Uses the same deploy action as `ci.yml`, `.github/actions/deploy` |
| `grow-corpus.yml` | Daily: adds films to the corpus, commits them and starts `ci.yml` |
| `uptime.yml` | Every five minutes: checks `/healthz` and opens an issue when the site is down |

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

## First Run

Nothing is built on first request. The corpus is committed and the image carries its embeddings. In local development, embeddings are computed from the corpus the first time a search runs (a minute or so) and cached in `data/`.

To add films to the corpus, run `make grow` (needs TMDB and OMDb keys). A search with no corpus present returns a clear error page and `/healthz` reports `degraded`; the app never crawls for data inside a request.

For local development without Docker, `python scripts/download_model.py` caches the model in `models/`.

## Production Checklist

Enforced at startup when `DEBUG` is off (the app refuses to boot otherwise):

- [ ] `SECRET_KEY`: a random value of at least 32 characters
- [ ] `PUBLIC_BASE_URL`: the site's `https://` origin

Needed for the stack to work:

- [ ] `SITE_ADDRESS`, `POSTGRES_PASSWORD`, `DATABASE_URL`, `DATABASE_SSLMODE=disable` (see *Setting up a new server*)
- [ ] The three `DEPLOY_*` secrets and the `DEPLOY_URL` variable in GitHub Actions

Recommended, and off until configured:

- [ ] Off-box backups: the four `BACKUP_S3_*` variables (see *Backups*)
- [ ] Error tracking: `SENTRY_DSN`
- [ ] Uptime: `uptime.yml` checks `/healthz` at `DEPLOY_URL` already; an external monitor is still worth having, since GitHub runs schedules late
- [ ] Corpus growth: the `TMDB_BEARER_TOKEN`, `TMDB_API_KEY` and `OMDB_API_KEY` Actions secrets
- [ ] Email (password reset, address confirmation): `RESEND_API_KEY` and `EMAIL_FROM`, which need a verified sending domain
- [ ] Sign in with Google: `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET`
- [ ] Stripe: the four Stripe variables, and a webhook endpoint in the Stripe dashboard pointing at `https://<domain>/stripe/webhook`

When the domain changes, update `SITE_ADDRESS`, `PUBLIC_BASE_URL` and `DEPLOY_URL` together.
