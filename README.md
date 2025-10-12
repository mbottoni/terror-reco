
# Recommendation engine for horror movies

The objective of this project is to create a fully functional
recommendation engine for horror movies. This project should
search public api's of movie databases and given the "mood" of 
the user (The user will type) the site should search the 
best matches on the database and offer 5 movie options.

This should contain the whole project, the frontend and the backend. 
The project should use Docker files, GH Actions, Only Python code.

Future work include a monetization strategy using stripe

---

## Setup

- This project now uses OMDb only. Get a free API key from `https://www.omdbapi.com/apikey.aspx`.
- Create a `.env` file with:
  - `OMDB_API_KEY=...`
  - `DEBUG=1`

## Run locally

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e '.[dev]'
set -a; source .env; set +a
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker build -t terror-reco:latest .
docker run --env-file .env -p 8000:8000 terror-reco:latest
```

Or with docker-compose (auto-loads `.env`):

```bash
docker compose up --build
```

## CI

- GitHub Actions workflow runs lint, type-check, tests, and builds the image.

## Notes

- Gunicorn is included in the image and used as the production server.