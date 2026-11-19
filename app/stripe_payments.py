"""
Stripe payment integration for TerrorReco.
Handles "Buy me a coffee" payments.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import stripe
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool
from starlette.responses import Response

from .auth import get_current_user
from .db import get_db
from .models import Payment, User
from .security import require_csrf
from .settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stripe", tags=["stripe"])
settings = get_settings()

# Written into every checkout session and checked on the way back, so the
# success page only thanks someone for a session this app created.
APP_METADATA = {"app": "terror-reco", "type": "coffee"}


def _page(
    request: Request,
    template: str,
    context: dict[str, Any],
    user: User | None,
    status_code: int = 200,
) -> Response:
    """Render through the shared templates, so the nav knows who is signed in."""
    resp: Response = request.app.state.templates.TemplateResponse(
        request, template, {**context, "user": user}, status_code=status_code
    )
    return resp


def _problem(request: Request, user: User | None, *, status_code: int, message: str) -> Response:
    return _page(
        request,
        "error.html",
        {"icon": "☕", "heading": "Payment problem", "message": message},
        user,
        status_code=status_code,
    )


# The handlers that call Stripe are plain ``def``: the SDK is synchronous, so
# FastAPI runs them in its threadpool. As ``async def`` each call blocked the
# event loop for a full network round trip.


@router.get("/coffee", response_class=HTMLResponse)
def buy_coffee_page(request: Request, user: User | None = Depends(get_current_user)) -> Response:
    """Display the buy coffee page."""
    if not settings.STRIPE_PUBLISHABLE_KEY:
        return _problem(
            request, user, status_code=503, message="Payments are not set up on this site yet."
        )
    return _page(
        request,
        "coffee.html",
        {
            "stripe_publishable_key": settings.STRIPE_PUBLISHABLE_KEY,
            "price_id": settings.COFFEE_PRICE_ID,
        },
        user,
    )


@router.post("/create-checkout-session", dependencies=[Depends(require_csrf)])
def create_checkout_session(
    request: Request, user: User | None = Depends(get_current_user)
) -> dict[str, Any]:
    """Create a Stripe checkout session for coffee purchase."""
    if not settings.STRIPE_SECRET_KEY or not settings.COFFEE_PRICE_ID:
        logger.error("Checkout requested but Stripe is not configured")
        raise HTTPException(status_code=503, detail="Payments are not available right now.")

    base_url = str(request.base_url).rstrip("/")
    # No payment_method_types: Stripe then offers every method enabled in the
    # dashboard (wallets, local methods), not cards alone.
    params: dict[str, Any] = {
        "api_key": settings.STRIPE_SECRET_KEY,
        "line_items": [{"price": settings.COFFEE_PRICE_ID, "quantity": 1}],
        "mode": "payment",
        "success_url": f"{base_url}/stripe/success?session_id={{CHECKOUT_SESSION_ID}}",
        "cancel_url": f"{base_url}/stripe/cancel",
        "metadata": APP_METADATA,
    }
    if user:
        # Comes back on the webhook, which is how a payment is tied to the
        # account that made it. Absent for signed-out supporters.
        params["client_reference_id"] = str(user.id)
    try:
        checkout_session = stripe.checkout.Session.create(**params)
    except stripe.StripeError:
        # The exception text can carry key and account details; it goes to
        # the log, never to the browser.
        logger.exception("Stripe refused to create a checkout session")
        raise HTTPException(
            status_code=502, detail="Could not start the payment. Please try again."
        ) from None

    logger.info("Checkout session created: ...%s", checkout_session.id[-6:])
    return {"checkout_url": checkout_session.url}


@router.get("/success", response_class=HTMLResponse)
def stripe_success(
    request: Request,
    session_id: str | None = None,
    user: User | None = Depends(get_current_user),
) -> Response:
    """Thank the supporter, but only for a payment Stripe confirms was made."""
    unconfirmed = (
        "We could not confirm that payment. If you were charged, it will still go through."
    )
    if not session_id or not settings.STRIPE_SECRET_KEY:
        return _problem(request, user, status_code=400, message=unconfirmed)

    try:
        session = stripe.checkout.Session.retrieve(session_id, api_key=settings.STRIPE_SECRET_KEY)
    except stripe.StripeError:
        logger.warning("Could not retrieve checkout session for the success page", exc_info=True)
        return _problem(request, user, status_code=400, message=unconfirmed)

    metadata = dict(session.metadata or {})
    if session.payment_status != "paid" or metadata.get("app") != APP_METADATA["app"]:
        # An abandoned, unpaid or foreign session id must not render as a
        # completed donation.
        return _problem(request, user, status_code=402, message=unconfirmed)

    # Only the two values the page shows; the session object stays here.
    return _page(
        request,
        "coffee_success.html",
        {
            "amount": (session.amount_total or 0) / 100,  # minor units
            "currency": (session.currency or "usd").upper(),
        },
        user,
    )


@router.get("/cancel", response_class=HTMLResponse)
def stripe_cancel(request: Request, user: User | None = Depends(get_current_user)) -> Response:
    """Handle cancelled payment."""
    return _page(request, "coffee_cancel.html", {}, user)


def record_checkout_completed(db: Session, event_id: str, session: dict[str, Any]) -> bool:
    """Store a completed checkout. Returns False if it was already recorded.

    Stripe delivers at least once and retries on any non-2xx, so the same
    session can arrive several times, possibly concurrently. The unique
    constraint on ``checkout_session_id`` is the arbiter; the lookup first is
    only there to make the common repeat cheap.
    """
    session_id = str(session.get("id") or "")
    if not session_id:
        return False
    if db.query(Payment.id).filter(Payment.checkout_session_id == session_id).first():
        return False

    user_id: int | None = None
    reference = session.get("client_reference_id")
    if reference and str(reference).isdigit() and db.get(User, int(reference)):
        user_id = int(reference)

    db.add(
        Payment(
            checkout_session_id=session_id,
            stripe_event_id=event_id,
            payment_intent_id=session.get("payment_intent") or None,
            amount_total=int(session.get("amount_total") or 0),
            currency=str(session.get("currency") or "usd")[:8],
            status="paid",
            user_id=user_id,
        )
    )
    try:
        db.commit()
    except IntegrityError:
        db.rollback()  # a concurrent delivery of the same event won the race
        return False
    return True


def record_refund(db: Session, charge: dict[str, Any]) -> bool:
    """Mark the payment behind a fully refunded charge. False if unknown."""
    intent = charge.get("payment_intent")
    if not intent or not charge.get("refunded"):
        return False
    payment = db.query(Payment).filter(Payment.payment_intent_id == intent).first()
    if payment is None or payment.status == "refunded":
        return False
    payment.status = "refunded"
    db.commit()
    return True


@router.post("/webhook")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)) -> dict[str, str]:
    """Handle Stripe webhooks."""
    if not settings.STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="Webhook secret not configured")

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    if not sig_header:
        # Without a signature there is nothing to verify against, so reject
        # rather than handing None to construct_event.
        raise HTTPException(status_code=400, detail="Missing stripe-signature header")

    try:
        stripe.Webhook.construct_event(  # type: ignore[no-untyped-call]
            payload, sig_header, settings.STRIPE_WEBHOOK_SECRET
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid payload") from e
    except stripe.SignatureVerificationError as e:
        raise HTTPException(status_code=400, detail="Invalid signature") from e

    # The payload is verified at this point; read it as plain JSON rather than
    # through the SDK's object wrapper.
    event: dict[str, Any] = json.loads(payload)
    obj: dict[str, Any] = (event.get("data") or {}).get("object") or {}
    event_id = str(event.get("id") or "")

    if event.get("type") == "checkout.session.completed":
        if obj.get("payment_status") != "paid":
            # Delayed payment methods complete the session before the money
            # arrives; only a paid session is a payment.
            logger.info("Checkout completed but not paid yet: ...%s", str(obj.get("id"))[-6:])
        elif await run_in_threadpool(record_checkout_completed, db, event_id, obj):
            logger.info("Payment recorded: ...%s", str(obj.get("id"))[-6:])
        else:
            logger.info("Payment already recorded: ...%s", str(obj.get("id"))[-6:])
    elif event.get("type") == "charge.refunded":
        if await run_in_threadpool(record_refund, db, obj):
            logger.info("Payment refunded: ...%s", str(obj.get("payment_intent"))[-6:])

    # Always 2xx for a verified event, including ones not handled here;
    # anything else makes Stripe retry it for days.
    return {"status": "success"}
