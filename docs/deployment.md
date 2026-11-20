# Deployment

How TerrorReco is deployed: one server running Docker Compose, shipped by GitHub Actions on every push to `main`.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OMDB_API_KEY` | Yes | -- | OMDb API key ([get one free](https://www.omdbapi.com/apikey.aspx)) |
| `SECRET_KEY` | Production | -- | Signs sessions and CSRF tokens. The app refuses to start in production without a random value of at least 32 characters |
| `PUBLIC_BASE_URL` | Production | -- | The site's origin, e.g. `https://example.com`. Absolute URLs (Stripe return URLs) are built from it. The app refuses to start in production without it |
| `DATABASE_URL` | Production | SQLite file | PostgreSQL connection URL |
| `DEBUG` | No | `false` | Set to `1` or `true` for development mode |
| `LOG_LEVEL` | No | `INFO` | Level for the app's own log lines |
| `SENTRY_DSN` | Optional | -- | Turns on error tracking. Unset means off. No request bodies, cookies or user details are sent |
| `RESEND_API_KEY`, `EMAIL_FROM` | Optional | -- | Turn on email: password reset, address confirmation at sign-up, and a sign-up form that does not reveal which addresses have accounts. The sender's domain must be verified with Resend. **Setting these changes sign-up:** new accounts must open an emailed link before they can sign in |
| `STRIPE_PUBLISHABLE_KEY` | Optional | -- | Stripe public key (for payments) |
| `STRIPE_SECRET_KEY` | Optional | -- | Stripe secret key |
| `STRIPE_WEBHOOK_SECRET` | Optional | -- | Stripe webhook signing secret |
| `COFFEE_PRICE_ID` | Optional | -- | Stripe Price used for the default $3 donation. Other amounts are priced by the app; without this, the default is too |

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

1. Starts from `python:3.11-slim` and installs system dependencies.
2. Installs the CPU build of PyTorch, then the locked Python dependencies from `requirements.txt` (generated from `uv.lock`), leaving out the CUDA wheels a CPU-only server never uses. The image is about 3.5 GB; with them it was 11.9 GB.
3. Switches to an unprivileged user.
4. **Downloads the sentence-transformer model** and then sets the hub to offline mode, so the container never contacts huggingface.co.
5. Copies the app, the corpus and the where-to-watch snapshot.
6. **Computes the corpus embeddings**, so a new container never encodes 500 films while a user waits.
7. Runs `uvicorn`.

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

`.github/workflows/ci.yml` runs on every push and pull request to `main`:

| Job | What it does |
|-----|--------------|
| `build-test` | Installs from the lockfile and checks it is current, then `ruff`, `black --check`, `mypy app` and `pytest` (which includes the ranking regression gate) |
| `image` | Builds the Docker image, boots it offline with production settings and checks `/healthz`. On a push to `main`, pushes it to GHCR |
| `deploy` | Push to `main` only: ships that image to the server and fails unless the site reports healthy |

`.github/workflows/deploy.yml` is the manual deploy and rollback described above. Both use the same deploy action, `.github/actions/deploy`.

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

To rebuild the corpus, run `make corpus` (needs TMDB and OMDb keys). A search with no corpus present returns a clear error page and `/healthz` reports `degraded`; the app never crawls for data inside a request.

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
- [ ] An external uptime check on `/healthz`
- [ ] Email (password reset, address confirmation): `RESEND_API_KEY` and `EMAIL_FROM`, which need a verified sending domain
- [ ] Stripe: the four Stripe variables, and a webhook endpoint in the Stripe dashboard pointing at `https://<domain>/stripe/webhook`

When the domain changes, update `SITE_ADDRESS`, `PUBLIC_BASE_URL` and `DEPLOY_URL` together.
