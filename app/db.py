from __future__ import annotations

import logging
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from tenacity import retry, stop_after_attempt, wait_exponential

from .settings import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


def _is_postgres(url: str) -> bool:
    return url.startswith(("postgres", "postgresql"))


def _normalize_database_url(raw: str) -> str:
    """Convert postgres:// URLs to the SQLAlchemy psycopg v3 dialect.

    SSL parameters are intentionally kept OUT of the URL; they are passed
    via ``connect_args`` so that psycopg v3 receives them as proper libpq
    keyword arguments (required for reliable SSL on Render / cloud Postgres).
    """
    if not raw:
        return raw
    p = urlparse(raw)

    # Only normalise PostgreSQL URLs, leave SQLite and others untouched
    if p.scheme not in ("postgres", "postgresql"):
        return raw

    scheme = "postgresql+psycopg"

    # Strip sslmode from URL query — we'll pass it via connect_args instead
    query_pairs = {k: v for k, v in parse_qsl(p.query) if k != "sslmode"}
    new_query = urlencode(query_pairs)
    new_p = p._replace(scheme=scheme, query=new_query)
    return urlunparse(new_p)


_settings = get_settings()
_DB_URL = _normalize_database_url(_settings.DATABASE_URL)

# --- engine kwargs ----------------------------------------------------------
_engine_kwargs: dict[str, Any] = {"future": True}

if _is_postgres(_DB_URL):
    # Pass SSL and keepalive settings as libpq keyword args so psycopg v3
    # negotiates SSL correctly (putting sslmode in the URL query string is
    # unreliable with the SQLAlchemy psycopg dialect).
    _engine_kwargs["connect_args"] = {
        "sslmode": "require",
        "connect_timeout": 10,
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5,
    }
    # Recycle connections before the server's idle-timeout kills them
    _engine_kwargs["pool_recycle"] = 300
    # Emit a lightweight SELECT 1 before handing out a connection so stale /
    # server-closed connections are silently replaced.
    _engine_kwargs["pool_pre_ping"] = True

_engine = create_engine(_DB_URL, **_engine_kwargs)
_SessionLocal = sessionmaker(
    bind=_engine, autoflush=False, autocommit=False, expire_on_commit=False
)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def _create_tables() -> None:
    """Try to run CREATE TABLE with exponential-backoff retries."""
    Base.metadata.create_all(bind=_engine)


def init_db() -> None:
    """Create tables, retrying transient connection failures.

    If the database is still unreachable after all retries the error is
    logged but the application is **not** killed — this lets the web
    process start and serve non-DB routes (e.g. health checks) while the
    database comes online.
    """
    try:
        _create_tables()
        logger.info("Database tables verified / created.")
    except Exception:
        logger.warning(
            "Could not connect to the database after retries. "
            "The app will start, but DB-dependent routes may fail until "
            "the database becomes reachable.",
            exc_info=True,
        )


@contextmanager
def get_db_session() -> Iterator[Session]:
    session = _SessionLocal()
    try:
        yield session
    finally:
        session.close()


def get_db() -> Generator[Session, None, None]:
    # FastAPI dependency: yields a session and ensures cleanup
    with get_db_session() as session:
        yield session
