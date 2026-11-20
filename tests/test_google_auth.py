"""Sign in with Google. Google itself is replaced by a stub; nothing leaves the process."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

import app.google_auth as google
from app.models import User
from app.security import NO_PASSWORD, hash_password
from app.settings import get_settings
from tests.conftest import _TestSession

EMAIL = "ana@example.com"
PASSWORD = "Str0ngPass!"
PROFILE = {"sub": "1234567890", "email": "Ana@Example.com", "email_verified": True}


@pytest.fixture()
def google_on(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Google sign-in is configured, and Google answers with ``calls['profile']``."""
    settings = get_settings()
    monkeypatch.setattr(settings, "GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.setattr(settings, "GOOGLE_CLIENT_SECRET", "client-secret")
    calls: dict[str, Any] = {"profile": dict(PROFILE), "exchanged": []}

    def _fetch(code: str, verifier: str, redirect_uri: str) -> dict[str, Any] | None:
        calls["exchanged"].append({"code": code, "verifier": verifier, "redirect": redirect_uri})
        profile: dict[str, Any] | None = calls["profile"]
        return profile

    monkeypatch.setattr(google, "fetch_profile", _fetch)
    return calls


def _start(client: Any) -> dict[str, str]:
    response = client.get("/auth/google", follow_redirects=False)
    assert response.status_code == 303
    url = urlparse(response.headers["location"])
    assert f"{url.scheme}://{url.netloc}{url.path}" == google.AUTHORIZE_URL
    return {key: values[0] for key, values in parse_qs(url.query).items()}


def _finish(client: Any, state: str, code: str = "the-code") -> Any:
    return client.get(
        "/auth/google/callback", params={"code": code, "state": state}, follow_redirects=False
    )


def _sign_in(client: Any) -> Any:
    return _finish(client, _start(client)["state"])


def _users() -> list[User]:
    with _TestSession() as db:
        return db.query(User).all()


def _signed_in(client: Any) -> bool:
    return client.get("/account/", follow_redirects=False).status_code == 200


def _add_user(**fields: Any) -> None:
    with _TestSession() as db:
        db.add(User(email=EMAIL, password_hash=hash_password(PASSWORD), **fields))
        db.commit()


class TestSwitchedOff:
    def test_routes_do_not_exist(self, client: Any) -> None:
        assert client.get("/auth/google", follow_redirects=False).status_code == 404
        assert client.get("/auth/google/callback?code=x&state=y").status_code == 404

    def test_no_button_is_shown(self, client: Any) -> None:
        assert "Continue with Google" not in client.get("/auth/login").text
        assert "Continue with Google" not in client.get("/auth/register").text


class TestStartingTheFlow:
    def test_the_button_is_shown(self, client: Any, google_on: dict[str, Any]) -> None:
        assert "Continue with Google" in client.get("/auth/login").text
        assert "Continue with Google" in client.get("/auth/register").text

    def test_the_redirect_carries_state_pkce_and_a_minimal_scope(
        self, client: Any, google_on: dict[str, Any]
    ) -> None:
        params = _start(client)
        assert params["client_id"] == "client-id"
        assert params["response_type"] == "code"
        assert params["scope"] == "openid email"
        assert params["code_challenge_method"] == "S256"
        assert len(params["state"]) >= 24 and len(params["code_challenge"]) >= 43
        assert params["redirect_uri"].endswith("/auth/google/callback")
        assert "client-secret" not in str(params)

    def test_the_redirect_uri_comes_from_configuration_not_the_host_header(
        self, client: Any, google_on: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(get_settings(), "PUBLIC_BASE_URL", "https://example.test")
        response = client.get(
            "/auth/google", headers={"host": "evil.example"}, follow_redirects=False
        )
        query = parse_qs(urlparse(response.headers["location"]).query)
        assert query["redirect_uri"] == ["https://example.test/auth/google/callback"]


class TestANewPerson:
    def test_an_account_is_created_confirmed_and_signed_in(
        self, client: Any, google_on: dict[str, Any]
    ) -> None:
        response = _sign_in(client)
        assert response.headers["location"] == "/onboarding/"
        assert _signed_in(client)
        (user,) = _users()
        assert user.email == EMAIL  # normalised
        assert user.google_sub == PROFILE["sub"]
        assert user.email_verified_at is not None
        assert user.password_hash == NO_PASSWORD

    def test_the_code_is_exchanged_with_the_verifier_from_this_browser(
        self, client: Any, google_on: dict[str, Any]
    ) -> None:
        params = _start(client)
        _finish(client, params["state"])
        (call,) = google_on["exchanged"]
        assert call["code"] == "the-code"
        assert google._challenge(call["verifier"]) == params["code_challenge"]

    def test_signing_in_again_finds_the_same_account(
        self, client: Any, google_on: dict[str, Any]
    ) -> None:
        _sign_in(client)
        client.post("/auth/logout")
        assert _sign_in(client).headers["location"] == "/"
        assert len(_users()) == 1

    def test_the_account_follows_googles_id_when_the_address_changes(
        self, client: Any, google_on: dict[str, Any]
    ) -> None:
        _sign_in(client)
        client.post("/auth/logout")
        google_on["profile"]["email"] = "renamed@example.com"
        _sign_in(client)
        assert [u.email for u in _users()] == [EMAIL]


class TestRefusals:
    def test_a_wrong_state_is_refused_and_nothing_is_exchanged(
        self, client: Any, google_on: dict[str, Any]
    ) -> None:
        _start(client)
        response = _finish(client, "not-the-state")
        assert response.headers["location"] == "/auth/login"
        assert google_on["exchanged"] == [] and _users() == []
        assert "did not work" in client.get("/auth/login").text

    def test_a_callback_nobody_started_is_refused(
        self, client: Any, google_on: dict[str, Any]
    ) -> None:
        _finish(client, "anything")
        assert google_on["exchanged"] == [] and not _signed_in(client)

    def test_the_state_works_once(self, client: Any, google_on: dict[str, Any]) -> None:
        state = _start(client)["state"]
        _finish(client, state)
        client.post("/auth/logout")
        _finish(client, state)
        assert not _signed_in(client)

    def test_an_address_google_has_not_verified_is_refused(
        self, client: Any, google_on: dict[str, Any]
    ) -> None:
        google_on["profile"]["email_verified"] = False
        _sign_in(client)
        assert _users() == [] and not _signed_in(client)

    def test_a_failed_exchange_is_refused(self, client: Any, google_on: dict[str, Any]) -> None:
        google_on["profile"] = None
        assert _sign_in(client).headers["location"] == "/auth/login"
        assert _users() == []

    def test_cancelling_on_googles_page_returns_to_sign_in(
        self, client: Any, google_on: dict[str, Any]
    ) -> None:
        state = _start(client)["state"]
        client.get(
            f"/auth/google/callback?error=access_denied&state={state}", follow_redirects=False
        )
        assert "cancelled" in client.get("/auth/login").text
        assert google_on["exchanged"] == []


class TestLinkingToAnExistingAccount:
    def test_a_confirmed_account_is_linked_and_keeps_its_password(
        self, client: Any, google_on: dict[str, Any]
    ) -> None:
        from datetime import UTC, datetime

        _add_user(email_verified_at=datetime.now(UTC))
        assert _sign_in(client).headers["location"] == "/"
        (user,) = _users()
        assert user.google_sub == PROFILE["sub"]
        assert user.password_hash != NO_PASSWORD

    def test_an_unconfirmed_account_loses_its_password_and_sessions(
        self, client: Any, google_on: dict[str, Any]
    ) -> None:
        """Someone else may have registered this address with their own password."""
        _add_user()
        before = _users()[0].session_version
        _sign_in(client)
        (user,) = _users()
        assert user.password_hash == NO_PASSWORD
        assert user.session_version == before + 1
        assert user.email_verified_at is not None
        assert _signed_in(client)  # the Google owner is in, on a fresh session


class TestAnAccountWithNoPassword:
    def _csrf(self, client: Any) -> str:
        return re.search(r'name="csrf" value="([^"]+)"', client.get("/auth/login").text).group(1)  # type: ignore[union-attr]

    def test_it_cannot_be_signed_in_to_with_any_password(
        self, client: Any, google_on: dict[str, Any]
    ) -> None:
        _sign_in(client)
        client.post("/auth/logout")
        for attempt in ("!", "", "anything1!"):
            client.post(
                "/auth/login",
                data={"email": EMAIL, "password": attempt, "csrf": self._csrf(client)},
            )
            assert not _signed_in(client)

    def test_the_account_page_offers_to_set_a_password(
        self, client: Any, google_on: dict[str, Any]
    ) -> None:
        _sign_in(client)
        page = client.get("/account/").text
        assert "Set a password" in page and "Google sign-in connected" in page
        assert 'name="current_password"' not in page

    def test_a_first_password_can_be_set_and_then_used(
        self, client: Any, google_on: dict[str, Any]
    ) -> None:
        _sign_in(client)
        client.post(
            "/account/password", data={"new_password": PASSWORD, "confirm_password": PASSWORD}
        )
        client.post("/auth/logout")
        client.post(
            "/auth/login", data={"email": EMAIL, "password": PASSWORD, "csrf": self._csrf(client)}
        )
        assert _signed_in(client)

    def test_email_cannot_be_changed_until_a_password_is_set(
        self, client: Any, google_on: dict[str, Any]
    ) -> None:
        _sign_in(client)
        client.post("/account/email", data={"new_email": "other@example.com"})
        assert [u.email for u in _users()] == [EMAIL]

    def test_deletion_needs_the_address_typed(self, client: Any, google_on: dict[str, Any]) -> None:
        _sign_in(client)
        client.post("/account/delete", data={"confirm_email": "wrong@example.com"})
        assert len(_users()) == 1
        client.post("/account/delete", data={"confirm_email": EMAIL})
        assert _users() == []


def test_fetch_profile_never_raises_and_never_logs_the_code(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import httpx

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise httpx.ConnectError("no route")

    monkeypatch.setattr(google.httpx, "post", _boom)
    with caplog.at_level("INFO"):
        assert google.fetch_profile("secret-code", "verifier", "https://x/cb") is None
    assert "secret-code" not in caplog.text


def test_fetch_profile_sends_the_verifier_and_reads_userinfo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    class _Reply:
        def __init__(self, payload: dict[str, Any]) -> None:
            self.status_code = 200
            self._payload = payload

        def json(self) -> dict[str, Any]:
            return self._payload

    def _post(url: str, data: dict[str, Any], timeout: float) -> _Reply:
        seen["token"] = (url, data)
        return _Reply({"access_token": "at"})

    def _get(url: str, headers: dict[str, str], timeout: float) -> _Reply:
        seen["userinfo"] = (url, headers)
        return _Reply(PROFILE)

    monkeypatch.setattr(google.httpx, "post", _post)
    monkeypatch.setattr(google.httpx, "get", _get)
    assert google.fetch_profile("c", "v", "https://x/cb") == PROFILE
    assert seen["token"][0] == google.TOKEN_URL
    assert seen["token"][1]["code_verifier"] == "v"
    assert seen["token"][1]["grant_type"] == "authorization_code"
    assert seen["userinfo"] == (google.USERINFO_URL, {"Authorization": "Bearer at"})
