from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, Iterator
<<<<<<< HEAD
=======
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
>>>>>>> 8adc085 (fix db for render)

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .settings import get_settings


class Base(DeclarativeBase):
	pass


def _normalize_database_url(raw: str) -> str:
	"""Convert postgres URLs to SQLAlchemy's psycopg v3 driver and ensure SSL on Render."""
	if not raw:
		return raw
	p = urlparse(raw)
	scheme = p.scheme
	# Map postgres/postgresql to psycopg driver name
	if scheme in ("postgres", "postgresql"):
		scheme = "postgresql+psycopg"
	# Ensure sslmode=require if not present (and not sqlite)
	query_pairs = dict(parse_qsl(p.query))
	if scheme.startswith("postgresql") and "sslmode" not in query_pairs:
		query_pairs["sslmode"] = "require"
	new_query = urlencode(query_pairs)
	new_p = p._replace(scheme=scheme, query=new_query)
	return urlunparse(new_p)


_settings = get_settings()
_DB_URL = _normalize_database_url(_settings.DATABASE_URL)
_engine = create_engine(_DB_URL, future=True)
_SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, expire_on_commit=False)


def init_db() -> None:
	Base.metadata.create_all(bind=_engine)


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
