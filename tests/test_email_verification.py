"""Confirming email addresses, and a registration that does not reveal accounts."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

import app.email_verification as ev
import app.password_reset as reset_mod
from app.models import EmailVerificationToken, User
from tests.conftest import _TestSession

EMAIL = "owner@example.com"
PASSWORD = "Str0ngPass!"


@pytest.fixture()
def outbox(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, str]]:
    """An email provider is configured; messages are captured, never sent."""
    sent: list[dict[str, str]] = []

    def _send(to: str, subject: str, text: str) -> bool:
        sent.append({"to": to, "subject": subject, "text": text})
        return True

    for module in (ev, reset_mod):
        monkeypatch.setattr(module, "email_configured", lambda: True)
        monkeypatch.setattr(module, "send_email", _send)
    ev._last_existing_notice.clear()
    return sent


def _csrf(client: Any, path: str) -> str:
    return re.search(r'name="csrf" value="([^"]+)"', client.get(path).text).group(1)  # type: ignore[union-attr]


def _register(client: Any, email: str = EMAIL, password: str = PASSWORD) -> Any:
    return client.post(
        "/auth/register",
        data={
            "email": email,
            "password": password,
            "confirm_password": password,
            "csrf": _csrf(client, "/auth/register"),
        },
        follow_redirects=False,
    )


def _login(client: Any, email: str = EMAIL, password: str = PASSWORD) -> Any:
    return client.post(
        "/auth/login",
        data={"email": email, "password": password, "csrf": _csrf(client, "/auth/login")},
        follow_redirects=False,
    )


def _link(message: dict[str, str]) -> str:
    return re.search(r"(/auth/verify/[\w-]+)", message["text"]).group(1)  # type: ignore[union-attr]


def _user(email: str = EMAIL) -> User:
    with _TestSession() as db:
        return db.query(User).filter(User.email == email).one()


def _signed_in(client: Any) -> bool:
    return client.get("/account/", follow_redirects=False).status_code == 200


class TestWithoutAnEmailProvider:
    """Nothing can be confirmed, so nothing is held back."""

    def test_registration_signs_in_at_once(self, client: Any) -> None:
        response = _register(client)
        assert response.headers["location"] == "/onboarding/"
        assert _signed_in(client)
        assert _user().email_verified_at is None

    def test_an_unconfirmed_account_can_sign_in(self, client: Any) -> None:
        _register(client)
        client.post("/auth/logout")
        assert _login(client).headers["location"] == "/"

    def test_the_account_page_does_not_mention_confirmation(self, client: Any) -> None:
        _register(client)
        assert "not confirmed" not in client.get("/account/").text


class TestRegistrationRevealsNothing:
    def test_a_new_address_lands_on_check_your_inbox_and_is_not_signed_in(
        self, client: Any, outbox: list[dict[str, str]]
    ) -> None:
        response = _register(client)
        assert response.headers["location"] == "/auth/check-email"
        assert not _signed_in(client)
        assert [m["to"] for m in outbox] == [EMAIL]
        assert "Confirm" in outbox[0]["subject"]

    def test_an_existing_address_gets_the_same_response(
        self, client: Any, outbox: list[dict[str, str]]
    ) -> None:
        first = _register(client)
        outbox.clear()
        second = _register(client, password="Different1!")
        assert (second.status_code, second.headers["location"]) == (
            first.status_code,
            first.headers["location"],
        )
        page = client.get("/auth/check-email").text
        assert "already" not in page.lower()

    def test_the_owner_of_an_existing_address_is_told_by_email(
        self, client: Any, outbox: list[dict[str, str]]
    ) -> None:
        _register(client)
        outbox.clear()
        _register(client, password="Different1!")
        assert len(outbox) == 1
        assert "already have" in outbox[0]["subject"]
        assert "/auth/forgot" in outbox[0]["text"]

    def test_a_second_attempt_does_not_change_the_existing_password(
        self, client: Any, outbox: list[dict[str, str]]
    ) -> None:
        _register(client)
        client.get(_link(outbox[0]))
        _register(client, password="Different1!")
        assert _login(client, password="Different1!").headers["location"] == "/auth/login"
        assert _login(client).headers["location"] == "/"

    def test_the_existing_account_notice_is_sent_once_an_hour(
        self, client: Any, outbox: list[dict[str, str]]
    ) -> None:
        _register(client)
        outbox.clear()
        for _ in range(3):
            _register(client)
        assert len(outbox) == 1

    def test_no_email_address_is_logged(
        self, client: Any, outbox: list[dict[str, str]], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("INFO"):
            _register(client)
            _register(client)
        assert EMAIL not in caplog.text


class TestSigningInNeedsAConfirmedAddress:
    def test_an_unconfirmed_account_is_held_and_sent_a_new_link(
        self, client: Any, outbox: list[dict[str, str]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _register(client)
        monkeypatch.setattr(ev, "RESEND_INTERVAL", timedelta(0))
        outbox.clear()
        response = _login(client)
        assert response.headers["location"] == "/auth/login"
        assert not _signed_in(client)
        assert "Confirm your email address first" in client.get("/auth/login").text
        assert len(outbox) == 1

    def test_a_wrong_password_on_an_unconfirmed_account_says_nothing_about_it(
        self, client: Any, outbox: list[dict[str, str]]
    ) -> None:
        _register(client)
        outbox.clear()
        _login(client, password="WrongPass1!")
        page = client.get("/auth/login").text
        assert "Invalid email or password" in page
        assert "Confirm" not in page
        assert outbox == []

    def test_the_link_confirms_the_address_and_then_sign_in_works(
        self, client: Any, outbox: list[dict[str, str]]
    ) -> None:
        _register(client)
        response = client.get(_link(outbox[0]), follow_redirects=False)
        assert response.headers["location"] == "/auth/login"
        assert _user().email_verified_at is not None
        assert _login(client).headers["location"] == "/"

    def test_the_link_does_not_sign_anyone_in(
        self, client: Any, outbox: list[dict[str, str]]
    ) -> None:
        _register(client)
        client.get(_link(outbox[0]))
        assert not _signed_in(client)


class TestTheLink:
    def test_it_works_once(self, client: Any, outbox: list[dict[str, str]]) -> None:
        _register(client)
        link = _link(outbox[0])
        client.get(link)
        client.get(link, follow_redirects=False)
        assert "expired or was already used" in client.get("/auth/login").text

    def test_it_expires(self, client: Any, outbox: list[dict[str, str]]) -> None:
        _register(client)
        with _TestSession() as db:
            db.query(EmailVerificationToken).update(
                {"expires_at": datetime.now(UTC) - timedelta(minutes=1)}
            )
            db.commit()
        client.get(_link(outbox[0]))
        assert _user().email_verified_at is None

    def test_a_made_up_token_confirms_nothing(
        self, client: Any, outbox: list[dict[str, str]]
    ) -> None:
        _register(client)
        client.get("/auth/verify/not-a-real-token")
        assert _user().email_verified_at is None

    def test_only_the_hash_is_stored(self, client: Any, outbox: list[dict[str, str]]) -> None:
        _register(client)
        token = _link(outbox[0]).rsplit("/", 1)[1]
        with _TestSession() as db:
            row = db.query(EmailVerificationToken).one()
        assert row.token_hash != token and len(row.token_hash) == 64

    def test_asking_again_within_a_minute_sends_nothing(
        self, client: Any, outbox: list[dict[str, str]]
    ) -> None:
        _register(client)
        _login(client)  # held; would send another link
        assert len(outbox) == 1

    def test_the_link_uses_the_configured_origin(
        self, client: Any, outbox: list[dict[str, str]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.settings import get_settings

        monkeypatch.setattr(get_settings(), "PUBLIC_BASE_URL", "https://example.test")
        _register(client)
        assert "https://example.test/auth/verify/" in outbox[0]["text"]


class TestChangingTheAddress:
    def _confirmed_and_signed_in(self, client: Any, outbox: list[dict[str, str]]) -> None:
        _register(client)
        client.get(_link(outbox[0]))
        _login(client)
        outbox.clear()

    def test_the_new_address_starts_unconfirmed_and_gets_a_link(
        self, client: Any, outbox: list[dict[str, str]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ev, "RESEND_INTERVAL", timedelta(0))
        self._confirmed_and_signed_in(client, outbox)
        client.post(
            "/account/email", data={"new_email": "new@example.com", "current_password": PASSWORD}
        )
        assert _user("new@example.com").email_verified_at is None
        assert [m["to"] for m in outbox] == ["new@example.com"]
        assert _signed_in(client)  # this session carries on
        assert "not confirmed yet" in client.get("/account/").text

    def test_a_link_sent_to_the_old_address_cannot_confirm_the_new_one(
        self, client: Any, outbox: list[dict[str, str]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _register(client)
        old_link = _link(outbox[0])
        with _TestSession() as db:  # the address changes before the link is opened
            db.query(User).update({"email": "new@example.com"})
            db.commit()
        client.get(old_link)
        assert _user("new@example.com").email_verified_at is None

    def test_the_account_page_can_resend_the_link(
        self, client: Any, outbox: list[dict[str, str]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ev, "RESEND_INTERVAL", timedelta(0))
        self._confirmed_and_signed_in(client, outbox)
        client.post(
            "/account/email", data={"new_email": "new@example.com", "current_password": PASSWORD}
        )
        outbox.clear()
        client.post("/auth/verify/resend")
        assert len(outbox) == 1
        client.get(_link(outbox[0]), follow_redirects=False)
        assert _user("new@example.com").email_verified_at is not None
        assert "Email address confirmed." in client.get("/account/").text

    def test_resend_needs_a_signed_in_user(self, client: Any, outbox: list[dict[str, str]]) -> None:
        response = client.post("/auth/verify/resend", follow_redirects=False)
        assert response.headers["location"] == "/auth/login"
        assert outbox == []


class TestOtherProofsOfOwnership:
    def test_a_password_reset_by_email_confirms_the_address(
        self, client: Any, outbox: list[dict[str, str]]
    ) -> None:
        _register(client)
        outbox.clear()
        client.post("/auth/forgot", data={"email": EMAIL})
        reset_link = re.search(r"(/auth/reset/[\w-]+)", outbox[0]["text"]).group(1)  # type: ignore[union-attr]
        client.post(
            reset_link, data={"new_password": "An0therPass!", "confirm_password": "An0therPass!"}
        )
        assert _user().email_verified_at is not None
        assert _login(client, password="An0therPass!").headers["location"] == "/"

    def test_the_export_says_whether_the_address_is_confirmed(self, client: Any) -> None:
        _register(client)
        assert client.get("/account/export").json()["account"]["email_verified_at"] is None

    def test_deleting_the_account_removes_its_tokens(
        self, client: Any, outbox: list[dict[str, str]]
    ) -> None:
        _register(client)
        client.get(_link(outbox[0]))
        _login(client)
        client.post(
            "/account/email", data={"new_email": "n@example.com", "current_password": PASSWORD}
        )
        client.post("/account/delete", data={"current_password": PASSWORD})
        with _TestSession() as db:
            assert db.query(EmailVerificationToken).count() == 0
