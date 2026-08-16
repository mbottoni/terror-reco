FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps — ca-certificates & libpq-dev are needed for reliable
# SSL connections from psycopg v3 to cloud-hosted PostgreSQL (e.g. Render).
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only FIRST to avoid the massive CUDA packages (~6 GB).
# sentence-transformers depends on torch; pre-installing the CPU wheel means
# the resolver below finds it already satisfied and skips the CUDA build.
# The version is pinned to the lockfile so this cannot silently drift.
ARG TORCH_VERSION=2.10.0
RUN pip install --upgrade pip && \
    pip install "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu

# Install app deps from the LOCKFILE rather than re-resolving. Re-resolving
# meant the image shipped different versions than CI tested against; the
# requirements file is generated from uv.lock by `make requirements`.
COPY pyproject.toml README.md requirements.txt /app/
RUN pip install --no-deps -r requirements.txt

# Pre-download the sentence-transformer model so the container starts fast
# (no HuggingFace download at runtime)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"

# Copy source
COPY app /app/app

# Ship the pre-built horror corpus so the container serves recommendations
# immediately, without needing TMDB/OMDb keys at build time. Embeddings are
# not committed; they are computed from this corpus on first use.
COPY data /app/data

# Migrations run at startup via init_db(), so they must ship with the image
COPY alembic.ini /app/alembic.ini
COPY migrations /app/migrations

# Render expects Docker services to listen on port 10000
EXPOSE 10000

# Run with uvicorn and respect $PORT (default 10000)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
