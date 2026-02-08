"""Shared fixtures for all automated tests."""

from __future__ import annotations

import os

# Force DEBUG=true so the session cookie uses https_only=False.
# TestClient makes plain HTTP requests; a Secure cookie would never
# be sent back, breaking flash messages and session state.
os.environ.setdefault("DEBUG", "true")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import StaticPool, create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.db import Base, get_db  # noqa: E402
from app.main import app  # noqa: E402

# In-memory SQLite so tests never touch the real database.
_test_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
_TestSession = sessionmaker(bind=_test_engine, autoflush=False, autocommit=False)


@pytest.fixture(autouse=True)
def _setup_db():
    """Create all tables before each test and drop them after."""
    Base.metadata.create_all(bind=_test_engine)
    yield
    Base.metadata.drop_all(bind=_test_engine)


def _override_get_db():
    """Yield a test-scoped DB session."""
    session = _TestSession()
    try:
        yield session
    finally:
        session.close()


# Swap the real DB dependency for the in-memory one.
app.dependency_overrides[get_db] = _override_get_db


@pytest.fixture()
def client():
    """Provide a TestClient wired to the in-memory DB."""
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def db():
    """Provide a raw DB session for direct assertions."""
    session = _TestSession()
    try:
        yield session
    finally:
        session.close()
