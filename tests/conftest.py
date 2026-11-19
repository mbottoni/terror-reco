"""Shared fixtures for all automated tests."""

from __future__ import annotations

import os
from typing import Any

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


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Every test starts with a clean slate.

    All TestClient requests share one client address, so without this the
    suite as a whole would trip the per-client limits.
    """
    from app.middleware import limiter

    limiter.reset()
    yield
    limiter.reset()


def _override_get_db():
    """Yield a test-scoped DB session."""
    session = _TestSession()
    try:
        yield session
    finally:
        session.close()


# Swap the real DB dependency for the in-memory one.
app.dependency_overrides[get_db] = _override_get_db


def session_csrf_token(client: TestClient) -> str | None:
    """Read the CSRF token out of the client's session cookie.

    The cookie is ``base64(json).timestamp.signature``; the payload is
    readable without the key (it is signed, not encrypted).
    """
    import base64
    import json

    from app.settings import get_settings

    cookie = client.cookies.get(get_settings().SESSION_COOKIE_NAME)
    if not cookie:
        return None
    payload = cookie.split(".", 1)[0]
    try:
        data = json.loads(base64.b64decode(payload + "=" * (-len(payload) % 4)))
    except ValueError:
        return None
    token = data.get("csrf")
    return token if isinstance(token, str) else None


class BrowserLikeClient(TestClient):
    """A TestClient that sends the session's CSRF token on POST, as pages do.

    Every real page embeds the token and its forms and fetch() calls send it
    back. Tests written against plain ``client.post(...)`` would otherwise all
    need the same boilerplate. Pass an explicit ``X-CSRF-Token`` header (even
    an empty one), or use a plain ``TestClient``, to test the check itself.
    """

    def post(self, url: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        headers = dict(kwargs.pop("headers", None) or {})
        if not any(name.lower() == "x-csrf-token" for name in headers):
            token = session_csrf_token(self)
            if token is None:
                self.get("/")  # any page mints the token
                token = session_csrf_token(self)
            if token:
                headers["X-CSRF-Token"] = token
        return super().post(url, headers=headers, **kwargs)


@pytest.fixture()
def client():
    """Provide a browser-like TestClient wired to the in-memory DB."""
    with BrowserLikeClient(app) as c:
        yield c


@pytest.fixture()
def db():
    """Provide a raw DB session for direct assertions."""
    session = _TestSession()
    try:
        yield session
    finally:
        session.close()
