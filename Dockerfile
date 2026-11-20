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
# The lockfile resolves torch from the CPU-only index on Linux (see
# [tool.uv.sources] in pyproject.toml), so requirements.txt asks for
# torch==<version>+cpu and lists no CUDA wheels. That exact build is already
# installed by the step above, which is why no extra index is needed here.
RUN pip install --no-deps -r requirements.txt

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

# Embeddings are not committed, so they are computed here, not on a user's
# first request. The step is expensive (about three minutes on two cores), so
# only what it actually reads is copied in before it: the corpus and the two
# modules that turn a film into a vector. A change anywhere else in the app
# then reuses this layer instead of re-encoding 500 films on every build.
#
# If embedding_text() or the model wrapper ever imports another app module,
# this step fails loudly at build time, which is the right moment to notice.
# Created as the app user, so the embeddings step can write into data/.
RUN mkdir -p /app/app/services /app/data
COPY --chown=app:app app/__init__.py /app/app/__init__.py
COPY --chown=app:app app/services/__init__.py app/services/corpus.py app/services/unified_recommender.py /app/app/services/
COPY --chown=app:app data/horror_corpus.json /app/data/horror_corpus.json
COPY --chown=app:app data/film_tags.json /app/data/film_tags.json
RUN python -c "from app.services.corpus import get_corpus_embeddings, load_corpus; get_corpus_embeddings(load_corpus())"

# The rest of the application, and the rest of the data (the where-to-watch
# snapshot). Copying over the directories above adds files; it does not
# remove the embeddings just written.
COPY --chown=app:app app /app/app
COPY --chown=app:app data /app/data

# Migrations run at startup via init_db(), so they must ship with the image
COPY --chown=app:app alembic.ini /app/alembic.ini
COPY --chown=app:app migrations /app/migrations

# Render expects Docker services to listen on port 10000
EXPOSE 10000

# Run with uvicorn and respect $PORT (default 10000)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
