"""Every state-changing route requires the session's CSRF token."""

from __future__ import annotations

import re
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import session_csrf_token

# (path, kwargs for a body the route would otherwise accept)
PROTECTED: list[tuple[str, dict[str, Any]]] = [
    ("/auth/logout", {}),
    ("/auth/logout-all", {}),
    ("/api/feedback", {"json": {"imdb_id": "tt1", "rating": 1}}),
    ("/watchlist/toggle", {"json": {"imdb_id": "tt1"}}),
    ("/stripe/create-checkout-session", {}),
]


@pytest.fixture()
def raw():
    """A plain client: sends no CSRF token unless the test adds one."""
    with TestClient(app) as c:
        yield c


def _page_token(client: TestClient) -> str:
    html = client.get("/").text
    match = re.search(r'<meta name="csrf-token" content="([^"]+)">', html)
    assert match, "every page must carry the csrf-token meta tag"
    return match.group(1)


@pytest.mark.parametrize(("path", "kwargs"), PROTECTED)
class TestTokenRequired:
    def test_missing_token_is_403(self, raw: TestClient, path: str, kwargs: dict[str, Any]) -> None:
        raw.get("/")  # establish a session; still no token sent
        assert raw.post(path, follow_redirects=False, **kwargs).status_code == 403

    def test_wrong_token_is_403(self, raw: TestClient, path: str, kwargs: dict[str, Any]) -> None:
        raw.get("/")
        resp = raw.post(
            path, headers={"X-CSRF-Token": "deadbeef.deadbeef"}, follow_redirects=False, **kwargs
        )
        assert resp.status_code == 403

    def test_token_from_another_session_is_403(
        self, raw: TestClient, path: str, kwargs: dict[str, Any]
    ) -> None:
        with TestClient(app) as attacker:
            foreign = _page_token(attacker)  # validly signed, wrong session
        raw.get("/")
        resp = raw.post(path, headers={"X-CSRF-Token": foreign}, follow_redirects=False, **kwargs)
        assert resp.status_code == 403

    def test_page_token_in_header_passes_the_check(
        self, raw: TestClient, path: str, kwargs: dict[str, Any]
    ) -> None:
        token = _page_token(raw)
        resp = raw.post(path, headers={"X-CSRF-Token": token}, follow_redirects=False, **kwargs)
        # Whatever the route then decides (401 signed out, 303, 503...), it
        # is no longer the CSRF check refusing it.
        assert resp.status_code != 403


class TestFormField:
    def test_logout_form_carries_the_token_and_works(self, raw: TestClient) -> None:
        page = raw.get("/auth/register").text
        csrf = re.search(r'name="csrf" value="([^"]+)"', page).group(1)  # type: ignore[union-attr]
        data = {
            "email": "form@example.com",
            "password": "password123",
            "confirm_password": "password123",
            "csrf": csrf,
        }
        home = raw.post("/auth/register", data=data, follow_redirects=True).text
        field = re.search(
            r'action="/auth/logout" method="post">\s*<input[^>]*value="([^"]+)"', home
        )
        assert field, "the sign-out form must embed the token"
        resp = raw.post("/auth/logout", data={"csrf": field.group(1)}, follow_redirects=False)
        assert resp.status_code == 303


class TestTokenLifetime:
    def test_token_is_stable_across_page_loads(self, raw: TestClient) -> None:
        """A second tab must not invalidate the first tab's forms."""
        first = _page_token(raw)
        raw.get("/auth/login")
        raw.get("/auth/register")
        assert _page_token(raw) == first == session_csrf_token(raw)

    def test_token_changes_at_sign_in(self, raw: TestClient) -> None:
        before = _page_token(raw)
        page = raw.get("/auth/register").text
        csrf = re.search(r'name="csrf" value="([^"]+)"', page).group(1)  # type: ignore[union-attr]
        data = {
            "email": "rotate@example.com",
            "password": "password123",
            "confirm_password": "password123",
            "csrf": csrf,
        }
        raw.post("/auth/register", data=data, follow_redirects=True)
        assert _page_token(raw) != before


class TestExemptions:
    def test_webhook_is_authenticated_by_signature_not_csrf(self, raw: TestClient) -> None:
        resp = raw.post("/stripe/webhook", content=b"{}")
        assert resp.status_code != 403

    def test_safe_methods_need_no_token(self, raw: TestClient) -> None:
        assert raw.get("/").status_code == 200
        assert raw.get("/healthz").status_code == 200
