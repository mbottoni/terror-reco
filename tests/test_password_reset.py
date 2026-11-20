"""Password reset by emailed link."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.password_reset as reset_mod
import app.services.email as email_mod
from app.main import app
from app.models import PasswordResetToken, User
from app.password_reset import SENT_MESSAGE, hash_token
from app.security import verify_password

EMAIL = "owner@example.com"
PASSWORD = "Str0ngPass!"
NEW_PASSWORD = "An0therPass!"


@pytest.fixture()
def outbox(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, str]]:
    """Email is configured; messages are captured, never sent."""
    sent: list[dict[str, str]] = []

    def _send(to: str, subject: str, text: str) -> bool:
        sent.append({"to": to, "subject": subject, "text": text})
        return True

    monkeypatch.setattr(reset_mod, "email_configured", lambda: True)
    monkeypatch.setattr(reset_mod, "send_email", _send)
    return sent


@pytest.fixture()
def no_email(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reset_mod, "email_configured", lambda: False)


def _csrf(client: Any, path: str) -> str:
    page = client.get(path).text
    return re.search(r'name="csrf" value="([^"]+)"', page).group(1)  # type: ignore[union-attr]


def _register(client: Any, email: str = EMAIL) -> None:
    client.post(
        "/auth/register",
        data={
            "email": email,
            "password": PASSWORD,
            "confirm_password": PASSWORD,
            "csrf": _csrf(client, "/auth/register"),
        },
    )
    client.post("/auth/logout")


def _login(client: Any, password: str) -> str:
    resp = client.post(
        "/auth/login",
        data={"email": EMAIL, "password": password, "csrf": _csrf(client, "/auth/login")},
        follow_redirects=True,
    )
    return str(resp.text)


def _forgot(client: Any, email: str = EMAIL) -> Any:
    return client.post("/auth/forgot", data={"email": email}, follow_redirects=True)


def _link_token(outbox: list[dict[str, str]]) -> str:
    return re.search(r"/auth/reset/(\S+)", outbox[-1]["text"]).group(1)  # type: ignore[union-attr]


def _reset(client: Any, token: str, new: str = NEW_PASSWORD, confirm: str | None = None) -> Any:
    return client.post(
        f"/auth/reset/{token}",
        data={"new_password": new, "confirm_password": new if confirm is None else confirm},
        follow_redirects=True,
    )


class TestRequestingALink:
    def test_known_address_gets_an_email_with_a_working_link(
        self, client: Any, outbox: list[dict[str, str]], db: Any
    ) -> None:
        _register(client)
        resp = _forgot(client, "  Owner@Example.com ")  # normalised like login
        assert SENT_MESSAGE in resp.text
        assert len(outbox) == 1 and outbox[0]["to"] == EMAIL
        token = _link_token(outbox)
        assert "Choose a new password" in client.get(f"/auth/reset/{token}").text
        assert 'name="new_password"' in client.get(f"/auth/reset/{token}").text

    def test_unknown_address_gets_the_same_answer_and_no_email(
        self, client: Any, outbox: list[dict[str, str]], db: Any
    ) -> None:
        _register(client)
        known = _forgot(client).text
        unknown = _forgot(client, "nobody@example.com").text
        assert SENT_MESSAGE in unknown
        def strip(page: str) -> str:
            # The CSRF token differs per render; everything else must match.
            return re.sub(r'(csrf[^"]*" (?:content|value)=")[^"]+', r"\\1", page)
            r'(csrf[^"]*" (?:content|value)=")[^"]+', r"\\1", page
        )  # noqa: E731
        assert strip(known) == strip(unknown)
        assert len(outbox) == 1
        assert db.query(PasswordResetToken).count() == 1

    def test_only_a_hash_of_the_token_is_stored(
        self, client: Any, outbox: list[dict[str, str]], db: Any
    ) -> None:
        _register(client)
        _forgot(client)
        token = _link_token(outbox)
        row = db.query(PasswordResetToken).one()
        assert row.token_hash == hash_token(token) and token not in row.token_hash
        assert len(token) >= 40

    def test_link_uses_the_configured_origin(
        self, client: Any, outbox: list[dict[str, str]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(reset_mod.get_settings(), "PUBLIC_BASE_URL", "https://real.example")
        _register(client)
        client.post("/auth/forgot", data={"email": EMAIL}, headers={"Host": "evil.example"})
        assert "https://real.example/auth/reset/" in outbox[0]["text"]
        assert "evil.example" not in outbox[0]["text"]

    def test_repeat_requests_do_not_flood_the_inbox(
        self, client: Any, outbox: list[dict[str, str]]
    ) -> None:
        _register(client)
        for _ in range(4):
            assert SENT_MESSAGE in _forgot(client).text
        assert len(outbox) == 1

    def test_a_new_link_replaces_the_old_one(
        self, client: Any, outbox: list[dict[str, str]], monkeypatch: pytest.MonkeyPatch, db: Any
    ) -> None:
        _register(client)
        _forgot(client)
        first = _link_token(outbox)
        monkeypatch.setattr(reset_mod, "RESEND_INTERVAL", timedelta(0))
        _forgot(client)
        second = _link_token(outbox)
        assert first != second
        assert "expired or was already used" in client.get(f"/auth/reset/{first}").text
        assert db.query(PasswordResetToken).count() == 1

    def test_requires_the_csrf_token(self, client: Any, outbox: list[dict[str, str]]) -> None:
        resp = client.post("/auth/forgot", data={"email": EMAIL}, headers={"X-CSRF-Token": "no"})
        assert resp.status_code == 403


class TestUsingALink:
    def test_sets_the_password_and_signs_out_everywhere(
        self, client: Any, outbox: list[dict[str, str]], db: Any
    ) -> None:
        _register(client)
        with TestClient(app) as other:  # a device still signed in with the old password
            other.post(
                "/auth/login",
                data={"email": EMAIL, "password": PASSWORD, "csrf": _csrf(other, "/auth/login")},
            )
            assert other.get("/account/", follow_redirects=False).status_code == 200
            _forgot(client)
            resp = _reset(client, _link_token(outbox))
            assert "Password changed" in resp.text and resp.url.path == "/auth/login"
            assert other.get("/account/", follow_redirects=False).status_code == 303
        assert verify_password(NEW_PASSWORD, db.query(User).one().password_hash)
        assert "Invalid email or password" in _login(client, PASSWORD)
        assert "Signed in" in _login(client, NEW_PASSWORD)

    def test_link_works_once(self, client: Any, outbox: list[dict[str, str]], db: Any) -> None:
        _register(client)
        _forgot(client)
        token = _link_token(outbox)
        _reset(client, token)
        again = _reset(client, token, new="Third0nePass!")
        assert "expired or was already used" in again.text
        assert verify_password(NEW_PASSWORD, db.query(User).one().password_hash)

    def test_expired_link_is_refused(
        self, client: Any, outbox: list[dict[str, str]], db: Any
    ) -> None:
        _register(client)
        _forgot(client)
        token = _link_token(outbox)
        row = db.query(PasswordResetToken).one()
        row.expires_at = datetime.now(UTC) - timedelta(seconds=1)
        db.commit()
        assert "expired or was already used" in client.get(f"/auth/reset/{token}").text
        assert "expired or was already used" in _reset(client, token).text
        assert verify_password(PASSWORD, db.query(User).one().password_hash)

    @pytest.mark.parametrize("token", ["not-a-real-token", "a" * 43, "../../etc/passwd"])
    def test_made_up_tokens_are_refused(self, client: Any, outbox: Any, token: str) -> None:
        resp = client.post(
            f"/auth/reset/{token}",
            data={"new_password": NEW_PASSWORD, "confirm_password": NEW_PASSWORD},
            follow_redirects=True,
        )
        assert resp.status_code in (200, 404)
        assert "Password changed" not in resp.text

    def test_weak_or_mismatched_password_keeps_the_link_usable(
        self, client: Any, outbox: list[dict[str, str]], db: Any
    ) -> None:
        _register(client)
        _forgot(client)
        token = _link_token(outbox)
        assert "at least" in _reset(client, token, new="short").text
        assert "do not match" in _reset(client, token, confirm="Different1!").text
        assert verify_password(PASSWORD, db.query(User).one().password_hash)
        assert "Password changed" in _reset(client, token).text

    def test_deleting_the_account_removes_its_tokens(
        self, client: Any, outbox: list[dict[str, str]], db: Any
    ) -> None:
        _register(client)
        _forgot(client)
        _login(client, PASSWORD)
        client.post("/account/delete", data={"current_password": PASSWORD})
        assert db.query(PasswordResetToken).count() == 0


class TestWithoutAnEmailProvider:
    def test_page_says_it_is_unavailable_and_offers_no_form(
        self, client: Any, no_email: None
    ) -> None:
        page = client.get("/auth/forgot").text
        assert "not available on this site yet" in page
        assert 'action="/auth/forgot"' not in page

    def test_never_claims_to_have_sent_anything(self, client: Any, no_email: None, db: Any) -> None:
        _register(client)
        resp = _forgot(client)
        assert SENT_MESSAGE not in resp.text
        assert "not available yet" in resp.text
        assert db.query(PasswordResetToken).count() == 0

    def test_login_page_links_to_it(self, client: Any) -> None:
        assert 'href="/auth/forgot"' in client.get("/auth/login").text


class TestEmailService:
    def test_unconfigured_sends_nothing_and_reports_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = email_mod.get_settings()
        monkeypatch.setattr(settings, "RESEND_API_KEY", None)
        called: list[Any] = []
        monkeypatch.setattr(email_mod.httpx, "post", lambda *a, **k: called.append(a))
        assert email_mod.send_email("a@example.com", "s", "t") is False
        assert called == [] and email_mod.email_configured() is False

    def test_configured_posts_to_the_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = email_mod.get_settings()
        monkeypatch.setattr(settings, "RESEND_API_KEY", "re_test")
        monkeypatch.setattr(settings, "EMAIL_FROM", "TerrorReco <noreply@example.com>")
        seen: dict[str, Any] = {}

        class _Ok:
            status_code = 200

        def _post(url: str, **kwargs: Any) -> Any:
            seen.update(url=url, **kwargs)
            return _Ok()

        monkeypatch.setattr(email_mod.httpx, "post", _post)
        assert email_mod.send_email("a@example.com", "Subject", "Body") is True
        assert seen["url"] == email_mod.RESEND_URL
        assert seen["headers"]["Authorization"] == "Bearer re_test"
        assert seen["json"]["to"] == ["a@example.com"] and seen["json"]["text"] == "Body"

    def test_provider_errors_are_swallowed_and_reported(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        import httpx

        settings = email_mod.get_settings()
        monkeypatch.setattr(settings, "RESEND_API_KEY", "re_test")
        monkeypatch.setattr(settings, "EMAIL_FROM", "x@example.com")

        def _boom(*args: Any, **kwargs: Any) -> Any:
            raise httpx.ConnectError("down")

        monkeypatch.setattr(email_mod.httpx, "post", _boom)
        assert email_mod.send_email("private.person@example.com", "s", "t") is False

        class _Rejected:
            status_code = 422

        monkeypatch.setattr(email_mod.httpx, "post", lambda *a, **k: _Rejected())
        assert email_mod.send_email("private.person@example.com", "s", "t") is False
        assert "private.person" not in caplog.text  # the recipient is never logged
