"""The payment path, with the Stripe SDK mocked.

No test here talks to Stripe. The webhook tests build a real signature with
the same scheme Stripe uses, so signature verification itself is exercised.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from types import SimpleNamespace
from typing import Any

import pytest
import stripe
from fastapi.testclient import TestClient

import app.stripe_payments as sp

WEBHOOK_SECRET = "whsec_test_secret"


@pytest.fixture()
def configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sp.settings, "STRIPE_SECRET_KEY", "sk_test_x")
    monkeypatch.setattr(sp.settings, "STRIPE_PUBLISHABLE_KEY", "pk_test_x")
    monkeypatch.setattr(sp.settings, "COFFEE_PRICE_ID", "price_test")
    monkeypatch.setattr(sp.settings, "STRIPE_WEBHOOK_SECRET", WEBHOOK_SECRET)


@pytest.fixture()
def unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "STRIPE_SECRET_KEY",
        "STRIPE_PUBLISHABLE_KEY",
        "COFFEE_PRICE_ID",
        "STRIPE_WEBHOOK_SECRET",
    ):
        monkeypatch.setattr(sp.settings, name, None)


def _session(**overrides: Any) -> SimpleNamespace:
    base: dict[str, Any] = {
        "id": "cs_test_123456",
        "url": "https://checkout.stripe.com/c/pay/cs_test_123456",
        "payment_status": "paid",
        "amount_total": 300,
        "currency": "usd",
        "metadata": {"app": "terror-reco", "type": "coffee"},
    }
    return SimpleNamespace(**{**base, **overrides})


class TestDebugEndpoint:
    def test_is_gone(self, client: TestClient, configured: None) -> None:
        assert client.get("/stripe/debug").status_code == 404


class TestCoffeePage:
    def test_renders_when_configured(self, client: TestClient, configured: None) -> None:
        resp = client.get("/stripe/coffee")
        assert resp.status_code == 200
        assert "pk_test_x" in resp.text

    def test_unconfigured_is_a_page_not_a_500(self, client: TestClient, unconfigured: None) -> None:
        resp = client.get("/stripe/coffee")
        assert resp.status_code == 503
        assert "not set up" in resp.text


class TestCreateCheckoutSession:
    def test_returns_the_checkout_url(
        self, client: TestClient, configured: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}

        def _create(**kwargs: Any) -> SimpleNamespace:
            seen.update(kwargs)
            return _session()

        monkeypatch.setattr(stripe.checkout.Session, "create", _create)
        resp = client.post("/stripe/create-checkout-session")
        assert resp.status_code == 200
        assert resp.json() == {"checkout_url": _session().url}
        assert seen["metadata"]["app"] == "terror-reco"
        assert seen["line_items"] == [{"price": "price_test", "quantity": 1}]
        assert "{CHECKOUT_SESSION_ID}" in seen["success_url"]
        # Left unset so Stripe offers wallets and local methods, not cards only.
        assert "payment_method_types" not in seen

    def test_unconfigured_is_503(self, client: TestClient, unconfigured: None) -> None:
        assert client.post("/stripe/create-checkout-session").status_code == 503

    def test_stripe_error_text_never_reaches_the_browser(
        self, client: TestClient, configured: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(**kwargs: Any) -> Any:
            raise stripe.StripeError("Invalid API Key provided: sk_test_SECRETSECRET")

        monkeypatch.setattr(stripe.checkout.Session, "create", _boom)
        resp = client.post("/stripe/create-checkout-session")
        assert resp.status_code == 502
        assert "SECRETSECRET" not in resp.text
        assert "Invalid API Key" not in resp.text


class TestSuccessPage:
    def _retrieve(self, monkeypatch: pytest.MonkeyPatch, session: Any) -> None:
        def _fake(session_id: str, **kwargs: Any) -> Any:
            if isinstance(session, Exception):
                raise session
            return session

        monkeypatch.setattr(stripe.checkout.Session, "retrieve", _fake)

    def test_paid_session_is_thanked_with_the_amount(
        self, client: TestClient, configured: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._retrieve(monkeypatch, _session())
        resp = client.get("/stripe/success", params={"session_id": "cs_test_123456"})
        assert resp.status_code == 200
        assert "Thank you" in resp.text
        assert "$3.00" in resp.text

    def test_other_currencies_are_labelled(
        self, client: TestClient, configured: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._retrieve(monkeypatch, _session(currency="eur", amount_total=500))
        resp = client.get("/stripe/success", params={"session_id": "cs_test_123456"})
        assert "5.00 EUR" in resp.text

    @pytest.mark.parametrize("status", ["unpaid", "no_payment_required"])
    def test_unpaid_session_is_not_thanked(
        self,
        client: TestClient,
        configured: None,
        monkeypatch: pytest.MonkeyPatch,
        status: str,
    ) -> None:
        self._retrieve(monkeypatch, _session(payment_status=status))
        resp = client.get("/stripe/success", params={"session_id": "cs_test_123456"})
        assert resp.status_code == 402
        assert "Thank you" not in resp.text

    def test_session_from_another_app_is_not_thanked(
        self, client: TestClient, configured: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._retrieve(monkeypatch, _session(metadata={"app": "something-else"}))
        resp = client.get("/stripe/success", params={"session_id": "cs_test_123456"})
        assert resp.status_code == 402

    def test_unknown_session_id_is_a_page_without_stripe_text(
        self, client: TestClient, configured: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._retrieve(monkeypatch, stripe.StripeError("No such checkout.session: cs_INTERNAL"))
        resp = client.get("/stripe/success", params={"session_id": "cs_bogus"})
        assert resp.status_code == 400
        assert "cs_INTERNAL" not in resp.text

    def test_missing_session_id_is_400(self, client: TestClient, configured: None) -> None:
        assert client.get("/stripe/success").status_code == 400


class TestCancelPage:
    def test_renders(self, client: TestClient) -> None:
        resp = client.get("/stripe/cancel")
        assert resp.status_code == 200
        assert "cancelled" in resp.text.lower()


def _signed(payload: bytes, secret: str = WEBHOOK_SECRET) -> str:
    timestamp = int(time.time())
    signed_payload = f"{timestamp}.".encode() + payload
    signature = hmac.new(secret.encode(), signed_payload, hashlib.sha256).hexdigest()
    return f"t={timestamp},v1={signature}"


def _event(event_type: str = "checkout.session.completed") -> bytes:
    return json.dumps(
        {
            "id": "evt_test_1",
            "object": "event",
            "type": event_type,
            "data": {"object": {"id": "cs_test_123456", "object": "checkout.session"}},
        }
    ).encode()


class TestWebhook:
    def test_valid_signature_is_accepted(self, client: TestClient, configured: None) -> None:
        payload = _event()
        resp = client.post(
            "/stripe/webhook", content=payload, headers={"stripe-signature": _signed(payload)}
        )
        assert resp.status_code == 200
        assert resp.json() == {"status": "success"}

    def test_other_event_types_are_acknowledged(self, client: TestClient, configured: None) -> None:
        payload = _event("charge.refunded")
        resp = client.post(
            "/stripe/webhook", content=payload, headers={"stripe-signature": _signed(payload)}
        )
        assert resp.status_code == 200

    def test_wrong_secret_is_rejected(self, client: TestClient, configured: None) -> None:
        payload = _event()
        resp = client.post(
            "/stripe/webhook",
            content=payload,
            headers={"stripe-signature": _signed(payload, secret="whsec_wrong")},
        )
        assert resp.status_code == 400

    def test_tampered_payload_is_rejected(self, client: TestClient, configured: None) -> None:
        signature = _signed(_event())
        resp = client.post(
            "/stripe/webhook",
            content=_event("charge.refunded"),
            headers={"stripe-signature": signature},
        )
        assert resp.status_code == 400

    def test_missing_signature_is_rejected(self, client: TestClient, configured: None) -> None:
        assert client.post("/stripe/webhook", content=_event()).status_code == 400

    def test_unconfigured_secret_is_500(self, client: TestClient, unconfigured: None) -> None:
        assert client.post("/stripe/webhook", content=_event()).status_code == 500
