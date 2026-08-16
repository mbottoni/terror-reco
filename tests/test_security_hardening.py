"""Tests for CSRF session binding and the health endpoint."""

from __future__ import annotations

import re
from typing import Any

from fastapi.testclient import TestClient

from app.main import app
from app.security import generate_csrf_token, validate_csrf_token


def _csrf_from(html: str) -> str:
    match = re.search(r'name="csrf" value="([^"]+)"', html)
    assert match, "no csrf token in form"
    return match.group(1)


class TestCsrfIsSessionBound:
    def test_token_from_another_session_is_rejected(self) -> None:
        """The vulnerability this closes.

        Every token is signed with the same application secret, so before
        binding, a token an attacker minted in their *own* session verified
        perfectly inside a victim's. Signature validity alone is not identity.
        """
        with TestClient(app) as attacker, TestClient(app) as victim:
            stolen = _csrf_from(attacker.get("/auth/register").text)
            victim.get("/auth/register")  # victim's session has a different token

            resp = victim.post(
                "/auth/register",
                data={
                    "email": "victim@example.com",
                    "password": "password123",
                    "confirm_password": "password123",
                    "csrf": stolen,
                },
                follow_redirects=True,
            )
            assert "Security check failed" in resp.text

    def test_matching_session_token_is_accepted(self) -> None:
        with TestClient(app) as c:
            csrf = _csrf_from(c.get("/auth/register").text)
            resp = c.post(
                "/auth/register",
                data={
                    "email": "ok@example.com",
                    "password": "password123",
                    "confirm_password": "password123",
                    "csrf": csrf,
                },
                follow_redirects=True,
            )
            assert "Security check failed" not in resp.text

    def test_valid_signature_without_a_session_is_rejected(self) -> None:
        """A correctly-signed token still needs a session to match against."""
        assert not validate_csrf_token(generate_csrf_token(), None)

    def test_malformed_and_empty_tokens(self) -> None:
        assert not validate_csrf_token("", "anything")
        assert not validate_csrf_token("no-dot-separator", "no-dot-separator")

    def test_tampered_signature_is_rejected(self) -> None:
        token = generate_csrf_token()
        salt, sig = token.split(".", 1)
        forged = f"{salt}.{'0' * len(sig)}"
        assert not validate_csrf_token(forged, forged)


class TestHealthEndpoint:
    def test_reports_status_and_corpus(self, client: Any) -> None:
        body = client.get("/healthz").json()
        assert body["status"] in {"ok", "degraded"}
        assert "corpus_films" in body
