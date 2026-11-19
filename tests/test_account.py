"""Account settings: password, email, export, deletion."""

from __future__ import annotations

import re
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import MovieFeedback, Payment, SearchHistory, SearchResult, User, WatchlistItem
from app.security import verify_password

EMAIL = "owner@example.com"
PASSWORD = "Str0ngPass!"
NEW_PASSWORD = "An0therPass!"


def _csrf(client: Any, path: str) -> str:
    page = client.get(path).text
    return re.search(r'name="csrf" value="([^"]+)"', page).group(1)  # type: ignore[union-attr]


def _register(client: Any, email: str = EMAIL, password: str = PASSWORD) -> None:
    client.post(
        "/auth/register",
        data={
            "email": email,
            "password": password,
            "confirm_password": password,
            "csrf": _csrf(client, "/auth/register"),
        },
        follow_redirects=True,
    )


def _login(client: Any, email: str = EMAIL, password: str = PASSWORD) -> Any:
    return client.post(
        "/auth/login",
        data={"email": email, "password": password, "csrf": _csrf(client, "/auth/login")},
        follow_redirects=True,
    )


def _signed_in(client: Any) -> bool:
    return bool(client.get("/account/", follow_redirects=False).status_code == 200)


class TestAccountPage:
    def test_signed_out_is_sent_to_login(self, client: Any) -> None:
        resp = client.get("/account/", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/auth/login"

    def test_shows_the_address_and_counts(self, client: Any) -> None:
        _register(client)
        client.post("/watchlist/toggle", json={"imdb_id": "tt1", "title": "Saved"})
        page = client.get("/account/").text
        assert EMAIL in page
        assert "1 saved films" in page

    def test_menu_links_to_it(self, client: Any) -> None:
        _register(client)
        assert 'href="/account/"' in client.get("/").text

    @pytest.mark.parametrize("path", ["/account/password", "/account/email", "/account/delete"])
    def test_every_form_requires_the_csrf_token(self, client: Any, path: str) -> None:
        _register(client)
        resp = client.post(
            path, data={"current_password": PASSWORD}, headers={"X-CSRF-Token": "wrong"}
        )
        assert resp.status_code == 403


class TestChangePassword:
    def _change(self, client: Any, current: str, new: str, confirm: str | None = None) -> Any:
        return client.post(
            "/account/password",
            data={
                "current_password": current,
                "new_password": new,
                "confirm_password": new if confirm is None else confirm,
            },
            follow_redirects=True,
        )

    def test_changes_it_and_keeps_this_device_signed_in(self, client: Any, db: Any) -> None:
        _register(client)
        resp = self._change(client, PASSWORD, NEW_PASSWORD)
        assert "Password changed" in resp.text
        assert _signed_in(client)
        user = db.query(User).one()
        assert verify_password(NEW_PASSWORD, user.password_hash)
        assert not verify_password(PASSWORD, user.password_hash)

    def test_other_devices_are_signed_out(self, client: Any) -> None:
        _register(client)
        with TestClient(app) as other:
            _login(other)
            assert _signed_in(other)
            self._change(client, PASSWORD, NEW_PASSWORD)
            assert not _signed_in(other)

    def test_old_password_no_longer_signs_in(self, client: Any) -> None:
        _register(client)
        self._change(client, PASSWORD, NEW_PASSWORD)
        client.post("/auth/logout")
        assert "Invalid email or password" in _login(client, password=PASSWORD).text
        assert "Signed in" in _login(client, password=NEW_PASSWORD).text

    def test_wrong_current_password_changes_nothing(self, client: Any, db: Any) -> None:
        _register(client)
        resp = self._change(client, "not-my-password", NEW_PASSWORD)
        assert "current password is not correct" in resp.text
        assert verify_password(PASSWORD, db.query(User).one().password_hash)

    def test_short_or_mismatched_new_password_is_refused(self, client: Any, db: Any) -> None:
        _register(client)
        assert "at least" in self._change(client, PASSWORD, "short").text
        assert "do not match" in self._change(client, PASSWORD, NEW_PASSWORD, "Different1!").text
        assert verify_password(PASSWORD, db.query(User).one().password_hash)


class TestChangeEmail:
    def _change(self, client: Any, new_email: str, password: str = PASSWORD) -> Any:
        return client.post(
            "/account/email",
            data={"new_email": new_email, "current_password": password},
            follow_redirects=True,
        )

    def test_changes_and_normalises(self, client: Any, db: Any) -> None:
        _register(client)
        assert "Email address updated" in self._change(client, "  New.Owner@Example.com ").text
        assert db.query(User).one().email == "new.owner@example.com"
        client.post("/auth/logout")
        assert "Signed in" in _login(client, email="new.owner@example.com").text

    def test_requires_the_current_password(self, client: Any, db: Any) -> None:
        _register(client)
        assert "not correct" in self._change(client, "x@example.com", password="wrong").text
        assert db.query(User).one().email == EMAIL

    def test_cannot_take_another_accounts_address(self, client: Any, db: Any) -> None:
        _register(client, email="taken@example.com")
        client.post("/auth/logout")
        _register(client)
        assert "already in use" in self._change(client, "Taken@Example.com").text
        assert db.query(User).filter(User.email == "taken@example.com").count() == 1

    @pytest.mark.parametrize("bad", ["not-an-email", "a@b", ""])
    def test_rejects_malformed_addresses(self, client: Any, db: Any, bad: str) -> None:
        _register(client)
        self._change(client, bad)
        assert db.query(User).one().email == EMAIL


def _fill_account(client: Any, db: Any) -> int:
    """Give the signed-in user one of everything. Returns the user id."""
    user = db.query(User).filter(User.email == EMAIL).one()
    client.post("/watchlist/toggle", json={"imdb_id": "tt1", "title": "Saved Film"})
    client.post("/api/feedback", json={"imdb_id": "tt2", "title": "Liked Film", "rating": 1})
    search = SearchHistory(user_id=user.id, mood="folk horror", strategy="semantic")
    search.results.append(SearchResult(position=0, imdb_id="tt3", title="Found Film"))
    db.add(search)
    db.add(
        Payment(
            checkout_session_id="cs_1",
            stripe_event_id="evt_1",
            amount_total=300,
            currency="usd",
            status="paid",
            user_id=user.id,
        )
    )
    db.commit()
    return int(user.id)


class TestExport:
    def test_signed_out_is_sent_to_login(self, client: Any) -> None:
        assert client.get("/account/export", follow_redirects=False).status_code == 303

    def test_contains_everything_stored_and_no_secrets(self, client: Any, db: Any) -> None:
        _register(client)
        _fill_account(client, db)
        resp = client.get("/account/export")
        assert resp.status_code == 200
        assert "attachment" in resp.headers["content-disposition"]
        data = resp.json()
        assert data["account"]["email"] == EMAIL
        assert data["searches"][0]["mood"] == "folk horror"
        assert data["searches"][0]["results"] == [{"imdb_id": "tt3", "title": "Found Film"}]
        assert data["ratings"][0]["imdb_id"] == "tt2" and data["ratings"][0]["rating"] == 1
        assert data["watchlist"][0]["imdb_id"] == "tt1"
        assert data["payments"][0]["amount"] == 3.0
        assert "password" not in resp.text and "argon2" not in resp.text

    def test_never_includes_another_users_rows(self, client: Any, db: Any) -> None:
        _register(client, email="other@example.com")
        client.post("/watchlist/toggle", json={"imdb_id": "tt9", "title": "Not Yours"})
        client.post("/auth/logout")
        _register(client)
        assert "Not Yours" not in client.get("/account/export").text


class TestDeleteAccount:
    def _delete(self, client: Any, password: str = PASSWORD) -> Any:
        return client.post(
            "/account/delete", data={"current_password": password}, follow_redirects=True
        )

    def test_wrong_password_deletes_nothing(self, client: Any, db: Any) -> None:
        _register(client)
        assert "not correct" in self._delete(client, password="wrong").text
        assert db.query(User).count() == 1

    def test_removes_the_user_and_every_row_that_references_them(
        self, client: Any, db: Any
    ) -> None:
        _register(client)
        user_id = _fill_account(client, db)
        resp = self._delete(client)
        assert "have been deleted" in resp.text
        db.expire_all()
        assert db.query(User).count() == 0
        assert db.query(SearchHistory).filter(SearchHistory.user_id == user_id).count() == 0
        assert db.query(SearchResult).count() == 0
        assert db.query(MovieFeedback).filter(MovieFeedback.user_id == user_id).count() == 0
        assert db.query(WatchlistItem).filter(WatchlistItem.user_id == user_id).count() == 0

    def test_payment_record_survives_unlinked(self, client: Any, db: Any) -> None:
        _register(client)
        _fill_account(client, db)
        self._delete(client)
        db.expire_all()
        payment = db.query(Payment).one()
        assert payment.user_id is None and payment.amount_total == 300

    def test_signed_out_afterwards_and_cannot_sign_back_in(self, client: Any) -> None:
        _register(client)
        self._delete(client)
        assert not _signed_in(client)
        assert "Invalid email or password" in _login(client).text

    def test_leaves_other_accounts_untouched(self, client: Any, db: Any) -> None:
        _register(client, email="bystander@example.com")
        client.post("/watchlist/toggle", json={"imdb_id": "tt9", "title": "Theirs"})
        client.post("/auth/logout")
        _register(client)
        self._delete(client)
        db.expire_all()
        assert [u.email for u in db.query(User).all()] == ["bystander@example.com"]
        assert db.query(WatchlistItem).count() == 1
