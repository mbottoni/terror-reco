"""
Stripe payment integration for TerrorReco.
Handles "Buy me a coffee" payments.
"""

from __future__ import annotations

import logging
from typing import Any

import stripe
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from starlette.responses import Response

from .auth import get_current_user
from .models import User
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


@router.post("/create-checkout-session")
def create_checkout_session(request: Request) -> dict[str, Any]:
    """Create a Stripe checkout session for coffee purchase."""
    if not settings.STRIPE_SECRET_KEY or not settings.COFFEE_PRICE_ID:
        logger.error("Checkout requested but Stripe is not configured")
        raise HTTPException(status_code=503, detail="Payments are not available right now.")

    base_url = str(request.base_url).rstrip("/")
    try:
        # No payment_method_types: Stripe then offers every method enabled in
        # the dashboard (wallets, local methods), not cards alone.
        checkout_session = stripe.checkout.Session.create(
            api_key=settings.STRIPE_SECRET_KEY,
            line_items=[{"price": settings.COFFEE_PRICE_ID, "quantity": 1}],
            mode="payment",
            success_url=f"{base_url}/stripe/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{base_url}/stripe/cancel",
            metadata=APP_METADATA,
        )
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


@router.post("/webhook")
async def stripe_webhook(request: Request) -> dict[str, str]:
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
        event = stripe.Webhook.construct_event(  # type: ignore[no-untyped-call]
            payload, sig_header, settings.STRIPE_WEBHOOK_SECRET
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid payload") from e
    except stripe.SignatureVerificationError as e:
        raise HTTPException(status_code=400, detail="Invalid signature") from e

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        # Recording the payment is tracked separately (issue #58); until then
        # the event is only logged.
        logger.info("Payment completed: ...%s", str(session["id"])[-6:])

    return {"status": "success"}
