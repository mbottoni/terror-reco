"""Confirming that an account's owner can read mail at its address.

Two problems, one mechanism:

* Any string shaped like an address became an account. A typo made an account
  nobody could recover, and nothing stopped registering someone else's address.
* Registration answered "Email already registered", so anyone could test
  whether an address had an account.

With an email provider configured, registration answers the same way for a new
and an existing address ("check your inbox") and says the rest by email: a
confirmation link to a new account, a "you already have an account" note to an
existing one. An account cannot be signed in to with a password until its
address is confirmed; without that, registering and then trying to sign in
would tell the two cases apart again.

Without a provider none of this can work, so nothing changes: accounts are
usable at once and the old message stays. The module says which mode it is in
through :func:`verification_enabled`.
"""

from __future__ import annotations

import logging
import secrets
import time
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from starlette.responses import Response

from .db import get_db
from .models import EmailVerificationToken, User
from .password_reset import _aware, _now, hash_token
from .security import require_csrf
from .services.email import email_configured, send_email
from .settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth")

TOKEN_LIFETIME = timedelta(hours=24)
# One confirmation email per account per minute, however often it is asked for.
RESEND_INTERVAL = timedelta(seconds=60)
# One "you already have an account" note per account per hour, so the sign-up
# form cannot be used to fill someone's inbox.
EXISTING_NOTICE_INTERVAL = 3600.0
_last_existing_notice: dict[int, float] = {}

CHECK_EMAIL_PATH = "/auth/check-email"


def verification_enabled() -> bool:
    """Whether addresses can be confirmed at all (an email provider is set)."""
    return email_configured()


def is_verified(user: User) -> bool:
    return user.email_verified_at is not None


def mark_verified(db: Session, user: User) -> None:
    if user.email_verified_at is None:
        user.email_verified_at = datetime.now(UTC)
    db.commit()


def _base_url(request: Request) -> str:
    return get_settings().PUBLIC_BASE_URL or str(request.base_url).rstrip("/")


def issue_token(db: Session, user: User) -> str | None:
    """A confirmation token for the user's current address.

    Replaces earlier unused ones. Returns None when one was issued within
    RESEND_INTERVAL.
    """
    unused = db.query(EmailVerificationToken).filter(
        EmailVerificationToken.user_id == user.id, EmailVerificationToken.used_at.is_(None)
    )
    recent = unused.order_by(EmailVerificationToken.created_at.desc()).first()
    if recent is not None and _now() - _aware(recent.created_at) < RESEND_INTERVAL:
        return None
    unused.delete()
    token = secrets.token_urlsafe(32)
    db.add(
        EmailVerificationToken(
            user_id=user.id,
            token_hash=hash_token(token),
            email=user.email,
            expires_at=_now() + TOKEN_LIFETIME,
        )
    )
    db.commit()
    return token


def send_verification(
    request: Request, background: BackgroundTasks, db: Session, user: User
) -> bool:
    """Queue a confirmation email. False if one went out a moment ago."""
    token = issue_token(db, user)
    if token is None:
        return False
    hours = int(TOKEN_LIFETIME.total_seconds() // 3600)
    body = (
        "Confirm this email address for your TerrorReco account by opening this link "
        f"within {hours} hours:\n{_base_url(request)}/auth/verify/{token}\n\n"
        "If you did not create an account, ignore this message and nothing will happen."
    )
    background.add_task(send_email, user.email, "Confirm your TerrorReco email address", body)
    logger.info("VERIFICATION SENT: user_id=%s", user.id)
    return True


def send_existing_account_notice(request: Request, background: BackgroundTasks, user: User) -> bool:
    """Tell an address's owner that someone tried to register with it."""
    now = time.monotonic()
    last = _last_existing_notice.get(user.id)
    if last is not None and now - last < EXISTING_NOTICE_INTERVAL:
        return False
    _last_existing_notice[user.id] = now
    base = _base_url(request)
    body = (
        "Someone tried to create a TerrorReco account with this email address, "
        "but it already has one.\n\n"
        f"If that was you, sign in here: {base}/auth/login\n"
        f"If you have forgotten the password: {base}/auth/forgot\n\n"
        "If it was not you, there is nothing to do: no new account was created "
        "and your password has not changed."
    )
    background.add_task(send_email, user.email, "You already have a TerrorReco account", body)
    logger.info("EXISTING ACCOUNT NOTICE: user_id=%s", user.id)
    return True


def _flash(request: Request, message: str, kind: str = "success") -> None:
    request.session["flash"] = message
    request.session["flash_type"] = kind


@router.get("/check-email", response_class=HTMLResponse, response_model=None)
async def check_email(request: Request) -> Response:
    """Where registration lands. Says the same thing for every address."""
    page: Response = request.app.state.templates.TemplateResponse(
        request,
        "check_email.html",
        {
            "user": None,
            "flash": request.session.pop("flash", None),
            "flash_type": request.session.pop("flash_type", "success"),
        },
    )
    return page


@router.get("/verify/{token}", response_model=None)
async def verify(request: Request, token: str, db: Session = Depends(get_db)) -> RedirectResponse:
    """Confirm an address from its emailed link.

    Does not sign anyone in. A link that signed its opener in could be sent to
    a victim to put them in an attacker's account; confirming is harmless.
    """
    row = (
        db.query(EmailVerificationToken)
        .filter(EmailVerificationToken.token_hash == hash_token(token))
        .first()
    )
    user = db.get(User, row.user_id) if row is not None else None
    stale = (
        row is None
        or user is None
        or row.used_at is not None
        or _aware(row.expires_at) <= _now()
        # Issued for an address the account no longer has.
        or row.email != user.email
    )
    if stale or row is None or user is None:
        _flash(
            request,
            "That confirmation link has expired or was already used. "
            "Sign in and we will send you a new one.",
            "error",
        )
        return RedirectResponse(url="/auth/login", status_code=303)

    row.used_at = _now()
    mark_verified(db, user)
    logger.info("EMAIL VERIFIED: user_id=%s", user.id)
    signed_in = request.session.get("user_id") == user.id
    _flash(request, "Email address confirmed." + ("" if signed_in else " You can sign in now."))
    return RedirectResponse(url="/account/" if signed_in else "/auth/login", status_code=303)


@router.post("/verify/resend", dependencies=[Depends(require_csrf)])
async def resend(
    request: Request,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
) -> RedirectResponse:
    """Send the signed-in user a fresh confirmation link."""
    from .auth import get_current_user

    user = get_current_user(request, db)
    if user is None:
        return RedirectResponse(url="/auth/login", status_code=303)
    if not verification_enabled():
        _flash(request, "Email confirmation is not available yet.", "error")
    elif is_verified(user):
        _flash(request, "Your email address is already confirmed.")
    elif send_verification(request, background, db, user):
        _flash(request, "We have sent a confirmation link to your address.")
    else:
        _flash(request, "We sent a link a moment ago. Check your inbox, then try again.", "error")
    return RedirectResponse(url="/account/", status_code=303)
