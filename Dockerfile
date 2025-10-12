FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install app deps
COPY pyproject.toml README.md /app/
RUN pip install --upgrade pip && pip install ".[dev]"

# Copy source
COPY app /app/app

# Render expects Docker services to listen on port 10000
EXPOSE 10000

# Run with uvicorn and respect $PORT (default 10000)
CMD sh -c "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"
