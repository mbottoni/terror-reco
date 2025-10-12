from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .settings import get_settings


class Base(DeclarativeBase):
	pass


_settings = get_settings()
_engine = create_engine(_settings.DATABASE_URL, future=True)
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
