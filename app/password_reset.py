"""Password reset by emailed link.

Without this a forgotten password meant a lost account: watchlist, history and
taste profile gone, recoverable only by editing the database.

The design choices that matter:

* The reset token is random and only its SHA-256 is stored, so a leaked
  database does not hand out working links.
* Links last 30 minutes and work once. Asking again invalidates earlier ones.
* The "forgot" form answers identically whether or not the address has an
  account, and the email is sent after the response, so neither the wording
  nor the timing reveals who is registered.
* A successful reset revokes every existing session for that user.
* The whole feature is off until an email provider is configured, and says so
  rather than claiming to have sent a message it could not send.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, BackgroundTasks, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from starlette.responses import Response

from .auth import MIN_PASSWORD_LENGTH, _set_flash, normalize_email, revoke_sessions
from .db import get_db
from .models import PasswordResetToken, User
from .security import hash_password, require_csrf
from .services.email import email_configured, send_email
from .settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", dependencies=[Depends(require_csrf)])

TOKEN_LIFETIME = timedelta(minutes=30)
# One email per address per minute, however often the form is submitted.
RESEND_INTERVAL = timedelta(seconds=60)
SENT_MESSAGE = "If that address has an account, we have sent it a link to reset the password."


def _now() -> datetime:
    return datetime.now(UTC)


def _aware(value: datetime) -> datetime:
    # SQLite hands back naive datetimes for timezone-aware columns.
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def issue_token(db: Session, user: User) -> str | None:
    """Create a reset token for *user*, replacing any earlier unused ones.

    Returns None when one was issued within RESEND_INTERVAL, so repeated
    submissions cannot be used to flood someone's inbox.
    """
    recent = (
        db.query(PasswordResetToken)
        .filter(PasswordResetToken.user_id == user.id, PasswordResetToken.used_at.is_(None))
        .order_by(PasswordResetToken.created_at.desc())
        .first()
    )
    if recent is not None and _now() - _aware(recent.created_at) < RESEND_INTERVAL:
        return None
    db.query(PasswordResetToken).filter(
        PasswordResetToken.user_id == user.id, PasswordResetToken.used_at.is_(None)
    ).delete()
    token = secrets.token_urlsafe(32)
    db.add(
        PasswordResetToken(
            user_id=user.id, token_hash=hash_token(token), expires_at=_now() + TOKEN_LIFETIME
        )
    )
    db.commit()
    return token


def find_valid_token(db: Session, token: str) -> PasswordResetToken | None:
    row = (
        db.query(PasswordResetToken)
        .filter(PasswordResetToken.token_hash == hash_token(token))
        .first()
    )
    if row is None or row.used_at is not None or _aware(row.expires_at) <= _now():
        return None
    return row


def _reset_link(request: Request, token: str) -> str:
    base = get_settings().PUBLIC_BASE_URL or str(request.base_url).rstrip("/")
    return f"{base}/auth/reset/{token}"


def _page(request: Request, template: str, **context: object) -> Response:
    page: Response = request.app.state.templates.TemplateResponse(
        request,
        template,
        {
            **context,
            "user": None,
            "flash": request.session.pop("flash", None),
            "flash_type": request.session.pop("flash_type", "success"),
        },
    )
    return page


@router.get("/forgot", response_class=HTMLResponse, response_model=None)
async def forgot_form(request: Request) -> Response:
    return _page(request, "forgot.html", available=email_configured())


@router.post("/forgot")
async def forgot(
    request: Request,
    background: BackgroundTasks,
    email: str = Form(...),
    db: Session = Depends(get_db),
) -> RedirectResponse:
    if not email_configured():
        # Never claim an email was sent when none can be.
        _set_flash(request, "Password reset by email is not available yet.", "error")
        return RedirectResponse(url="/auth/forgot", status_code=303)

    user = db.query(User).filter(User.email == normalize_email(email)).first()
    if user is not None:
        token = issue_token(db, user)
        if token is not None:
            link = _reset_link(request, token)
            minutes = int(TOKEN_LIFETIME.total_seconds() // 60)
            body = (
                "Someone asked to reset the password for your TerrorReco account.\n\n"
                f"To choose a new password, open this link within {minutes} minutes:\n{link}\n\n"
                "If that was not you, ignore this message: your password has not changed."
            )
            # After the response: the reply takes the same time whether or
            # not the address has an account.
            background.add_task(send_email, user.email, "Reset your TerrorReco password", body)
            logger.info("PASSWORD RESET REQUESTED: user_id=%s", user.id)
    # The same answer either way.
    _set_flash(request, SENT_MESSAGE)
    return RedirectResponse(url="/auth/forgot", status_code=303)


@router.get("/reset/{token}", response_class=HTMLResponse, response_model=None)
async def reset_form(request: Request, token: str, db: Session = Depends(get_db)) -> Response:
    valid = find_valid_token(db, token) is not None
    return _page(
        request,
        "reset.html",
        token=token,
        valid=valid,
        min_password_length=MIN_PASSWORD_LENGTH,
    )


@router.post("/reset/{token}")
async def reset(
    request: Request,
    token: str,
    new_password: str = Form(...),
    confirm_password: str = Form(...),
    db: Session = Depends(get_db),
) -> RedirectResponse:
    row = find_valid_token(db, token)
    if row is None:
        _set_flash(request, "That reset link has expired or was already used.", "error")
        return RedirectResponse(url="/auth/forgot", status_code=303)
    back = RedirectResponse(url=f"/auth/reset/{token}", status_code=303)
    if len(new_password) < MIN_PASSWORD_LENGTH:
        _set_flash(request, f"Password must be at least {MIN_PASSWORD_LENGTH} characters.", "error")
        return back
    if new_password != confirm_password:
        _set_flash(request, "The passwords do not match.", "error")
        return back

    user = db.get(User, row.user_id)
    if user is None:
        _set_flash(request, "That reset link has expired or was already used.", "error")
        return RedirectResponse(url="/auth/forgot", status_code=303)

    user.password_hash = hash_password(new_password)
    row.used_at = _now()
    # The link arrived by email, so its opener can read mail at this address.
    if user.email_verified_at is None:
        user.email_verified_at = _now()
    # Whoever had a session with the old password is signed out everywhere.
    revoke_sessions(db, user)
    logger.info("PASSWORD RESET COMPLETED: user_id=%s", user.id)
    request.session.clear()
    request.session["flash"] = "Password changed. Sign in with your new password."
    request.session["flash_type"] = "success"
    return RedirectResponse(url="/auth/login", status_code=303)
