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
# The lockfile pins the PyPI (CUDA) build of torch, whose nvidia-*/triton
# wheels are listed as packages in their own right -- so `--no-deps` installed
# all ~9 GB of them next to the CPU torch above. Nothing imports them on a
# CPU-only box; dropping them takes the image from 11.9 GB to a few GB.
RUN grep -v -E '^(nvidia-|triton==)' requirements.txt > /tmp/requirements.cpu.txt && \
    pip install --no-deps -r /tmp/requirements.cpu.txt && \
    rm /tmp/requirements.cpu.txt

# Everything from here on runs as an unprivileged user, and so does the app.
# Created before the model download so the cache it writes is owned by that
# user from the start; a later `chown -R` would copy ~450 MB into a new layer.
RUN useradd --create-home --uid 10001 app && chown app:app /app
ENV HF_HOME=/app/.cache/huggingface
USER app

# Pre-download the sentence-transformer model so the container starts fast
# (no HuggingFace download at runtime)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"

# The model is in the image now. Offline mode stops the hub client from
# phoning home on every start, so boot does not depend on huggingface.co
# being reachable. Set before the embeddings step below, which proves at
# build time that the cached model loads without the network.
ENV HF_HUB_OFFLINE=1

# Copy source
COPY --chown=app:app app /app/app

# Ship the pre-built horror corpus so the container serves recommendations
# immediately, without needing TMDB/OMDb keys at build time.
COPY --chown=app:app data /app/data

# Embeddings are not committed, so compute them here rather than on the first
# request: a fresh container would otherwise re-encode the whole corpus while a
# user waits. A valid cache copied in with data/ is reused as-is.
RUN python -c "from app.services.corpus import get_corpus_embeddings, load_corpus; get_corpus_embeddings(load_corpus())"

# Migrations run at startup via init_db(), so they must ship with the image
COPY --chown=app:app alembic.ini /app/alembic.ini
COPY --chown=app:app migrations /app/migrations

# Render expects Docker services to listen on port 10000
EXPOSE 10000

# Run with uvicorn and respect $PORT (default 10000)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
