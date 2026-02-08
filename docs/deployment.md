# Deployment

This guide covers deploying TerrorReco with Docker, on Render, and the CI/CD pipeline.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OMDB_API_KEY` | Yes | -- | OMDb API key ([get one free](https://www.omdbapi.com/apikey.aspx)) |
| `SECRET_KEY` | Recommended | Auto-generated | Session encryption key (must be stable in production) |
| `DATABASE_URL` | Production | SQLite file | PostgreSQL connection URL |
| `DEBUG` | No | `false` | Set to `1` or `true` for development mode |
| `PORT` | No | `10000` | Port the server listens on |
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
