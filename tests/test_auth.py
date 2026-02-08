"""Tests for the full authentication flow (register -> login -> logout)."""

from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models import User  # noqa: I001 – grouped with app imports

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_csrf(client: TestClient, path: str = "/auth/register") -> str:
    """Fetch a page and extract the CSRF token from the hidden form field."""
    resp = client.get(path)
    assert resp.status_code == 200
    text = resp.text
    # <input type="hidden" name="csrf" value="TOKEN">
    marker = 'name="csrf" value="'
    start = text.index(marker) + len(marker)
    end = text.index('"', start)
    return text[start:end]


def _register(
    client: TestClient,
    email: str = "user@example.com",
    password: str = "Str0ngPass!",
    confirm: str | None = None,
):
    """Register a new user and return the response (follows redirects)."""
    csrf = _get_csrf(client, "/auth/register")
    return client.post(
        "/auth/register",
        data={
            "email": email,
            "password": password,
            "confirm_password": confirm if confirm is not None else password,
            "csrf": csrf,
        },
        follow_redirects=True,
    )


def _login(
    client: TestClient,
    email: str = "user@example.com",
    password: str = "Str0ngPass!",
):
    """Login an existing user and return the response (follows redirects)."""
    csrf = _get_csrf(client, "/auth/login")
    return client.post(
        "/auth/login",
        data={"email": email, "password": password, "csrf": csrf},
        follow_redirects=True,
    )


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register_success(self, client: TestClient, db: Session):
        """Registering with valid data creates the user and signs them in."""
        resp = _register(client)
        assert resp.status_code == 200
        # Flash success shown on homepage
        assert "Account created" in resp.text
        # Avatar visible (first letter of email)
        assert 'class="avatar-btn"' in resp.text

        # User exists in DB
        user = db.query(User).filter(User.email == "user@example.com").first()
        assert user is not None

    def test_register_short_password(self, client: TestClient):
        """Passwords shorter than 8 characters are rejected."""
        resp = _register(client, password="short", confirm="short")
        assert "at least 8 characters" in resp.text

    def test_register_mismatched_passwords(self, client: TestClient):
        """Mismatched password and confirm_password are rejected."""
        resp = _register(client, password="Str0ngPass!", confirm="Different1!")
        assert "do not match" in resp.text

    def test_register_invalid_email(self, client: TestClient):
        """Badly-formatted email is rejected."""
        resp = _register(client, email="not-an-email")
        assert "valid email" in resp.text

    def test_register_duplicate_email(self, client: TestClient):
        """Registering the same email twice shows an error."""
        _register(client, email="dup@example.com")
        # Log out first so we can visit /auth/register again
        client.post("/auth/logout", follow_redirects=True)
        resp = _register(client, email="dup@example.com")
        assert "already registered" in resp.text

    def test_register_preserves_email_on_error(self, client: TestClient):
        """Email field stays pre-filled when registration fails."""
        resp = _register(client, email="keep@example.com", password="short", confirm="short")
        assert 'value="keep@example.com"' in resp.text


# ---------------------------------------------------------------------------
# Login tests
# ---------------------------------------------------------------------------


class TestLogin:
    def test_login_success(self, client: TestClient):
        """Correct credentials sign the user in and redirect to home."""
        _register(client)
        client.post("/auth/logout", follow_redirects=True)

        resp = _login(client)
        assert resp.status_code == 200
        assert "Signed in successfully" in resp.text
        assert 'class="avatar-btn"' in resp.text

    def test_login_wrong_password(self, client: TestClient):
        """Wrong password shows a generic error (no info leakage)."""
        _register(client)
        client.post("/auth/logout", follow_redirects=True)

        resp = _login(client, password="WrongPassword1!")
        assert "Invalid email or password" in resp.text

    def test_login_nonexistent_email(self, client: TestClient):
        """Non-existent email shows the same generic error."""
        resp = _login(client, email="nobody@example.com")
        assert "Invalid email or password" in resp.text

    def test_login_preserves_email_on_error(self, client: TestClient):
        """Email field stays pre-filled after a failed login."""
        resp = _login(client, email="kept@example.com", password="whatever1234")
        assert 'value="kept@example.com"' in resp.text

    def test_login_redirect_when_already_logged_in(self, client: TestClient):
        """Visiting /auth/login while logged in redirects to /."""
        _register(client)
        resp = client.get("/auth/login", follow_redirects=True)
        # Should end up on homepage, not the login form
        assert "Welcome back" not in resp.text


# ---------------------------------------------------------------------------
# Logout tests
# ---------------------------------------------------------------------------


class TestLogout:
    def test_logout_clears_session(self, client: TestClient):
        """Signing out clears the session and shows a flash."""
        _register(client)
        resp = client.post("/auth/logout", follow_redirects=True)
        assert resp.status_code == 200
        assert "signed out" in resp.text.lower()
        # Login/signup buttons should reappear (no avatar)
        assert 'class="avatar-btn"' not in resp.text

    def test_logged_out_user_cannot_access_history(self, client: TestClient):
        """Visiting /history/ without a session redirects to login."""
        resp = client.get("/history/", follow_redirects=True)
        assert "Log in" in resp.text or "Sign in" in resp.text


# ---------------------------------------------------------------------------
# Flash message rendering
# ---------------------------------------------------------------------------


class TestFlashMessages:
    def test_success_flash_has_correct_class(self, client: TestClient):
        """Success flash uses the default (green) style, not flash-error."""
        resp = _register(client)
        assert "Account created" in resp.text
        # The actual flash div should NOT have the error class.
        # We check inside the flash container, not the CSS stylesheet.
        assert 'class="flash flash-error"' not in resp.text
        assert 'class="flash"' in resp.text

    def test_error_flash_has_error_class(self, client: TestClient):
        """Error flash uses the flash-error class."""
        resp = _login(client, email="nobody@x.com")
        assert 'class="flash flash-error"' in resp.text
