"""Account settings: change password or email, export or delete your data.

Every action that weakens or ends access to the account asks for the current
password first, so a borrowed, unlocked session cannot take the account over
or destroy it.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy.orm import Session
from starlette.responses import Response

from .auth import (
    MIN_PASSWORD_LENGTH,
    _set_flash,
    get_current_user,
    normalize_email,
    revoke_sessions,
    start_session,
)
from .db import get_db
from .models import MovieFeedback, Payment, SearchHistory, User, WatchlistItem
from .security import hash_password, require_csrf, verify_password

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/account", dependencies=[Depends(require_csrf)])

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _to_login() -> RedirectResponse:
    # A fresh response each time: middleware writes headers (including the
    # session cookie) onto the response object, so one must never be shared.
    return RedirectResponse(url="/auth/login", status_code=303)


def _back(request: Request, message: str, kind: str = "success") -> RedirectResponse:
    _set_flash(request, message, kind)
    return RedirectResponse(url="/account/", status_code=303)


@router.get("/", response_class=HTMLResponse, response_model=None)
async def account_page(
    request: Request,
    user: User | None = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    if not user:
        return _to_login()
    counts = {
        "searches": db.query(SearchHistory).filter(SearchHistory.user_id == user.id).count(),
        "ratings": db.query(MovieFeedback).filter(MovieFeedback.user_id == user.id).count(),
        "saved": db.query(WatchlistItem).filter(WatchlistItem.user_id == user.id).count(),
    }
    page: Response = request.app.state.templates.TemplateResponse(
        request,
        "account.html",
        {
            "user": user,
            "counts": counts,
            "min_password_length": MIN_PASSWORD_LENGTH,
            "flash": request.session.pop("flash", None),
            "flash_type": request.session.pop("flash_type", "success"),
        },
    )
    return page


@router.post("/password")
async def change_password(
    request: Request,
    current_password: str = Form(...),
    new_password: str = Form(...),
    confirm_password: str = Form(...),
    user: User | None = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> RedirectResponse:
    if not user:
        return _to_login()
    if not verify_password(current_password, user.password_hash):
        logger.warning("PASSWORD CHANGE FAIL: wrong current password (user_id=%s)", user.id)
        return _back(request, "Your current password is not correct.", "error")
    if len(new_password) < MIN_PASSWORD_LENGTH:
        return _back(
            request, f"Password must be at least {MIN_PASSWORD_LENGTH} characters.", "error"
        )
    if new_password != confirm_password:
        return _back(request, "The new passwords do not match.", "error")

    user.password_hash = hash_password(new_password)
    # Ends every other session: whoever knew the old password is signed out.
    revoke_sessions(db, user)
    start_session(request, user)  # this device stays signed in
    logger.info("PASSWORD CHANGED: user_id=%s", user.id)
    return _back(request, "Password changed. Other devices have been signed out.")


@router.post("/email")
async def change_email(
    request: Request,
    new_email: str = Form(...),
    current_password: str = Form(...),
    user: User | None = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> RedirectResponse:
    if not user:
        return _to_login()
    if not verify_password(current_password, user.password_hash):
        logger.warning("EMAIL CHANGE FAIL: wrong current password (user_id=%s)", user.id)
        return _back(request, "Your current password is not correct.", "error")
    email = normalize_email(new_email)
    if not _EMAIL_RE.match(email):
        return _back(request, "Please enter a valid email address.", "error")
    if email == user.email:
        return _back(request, "That is already your email address.", "error")
    if db.query(User.id).filter(User.email == email).first():
        return _back(request, "That email address is already in use.", "error")

    user.email = email
    db.commit()
    logger.info("EMAIL CHANGED: user_id=%s", user.id)
    return _back(request, "Email address updated.")


def export_user_data(db: Session, user: User) -> dict[str, Any]:
    """Everything stored about *user*, as plain JSON-serialisable data."""

    def when(value: Any) -> str | None:
        return value.isoformat() if value else None

    searches = (
        db.query(SearchHistory)
        .filter(SearchHistory.user_id == user.id)
        .order_by(SearchHistory.created_at)
        .all()
    )
    ratings = db.query(MovieFeedback).filter(MovieFeedback.user_id == user.id).all()
    saved = db.query(WatchlistItem).filter(WatchlistItem.user_id == user.id).all()
    payments = db.query(Payment).filter(Payment.user_id == user.id).all()
    return {
        "account": {"email": user.email, "created_at": when(user.created_at)},
        "searches": [
            {
                "mood": s.mood,
                "strategy": s.strategy,
                "created_at": when(s.created_at),
                "results": [{"imdb_id": r.imdb_id, "title": r.title} for r in s.results],
            }
            for s in searches
        ],
        "ratings": [
            {
                "imdb_id": f.imdb_id,
                "title": f.title,
                "rating": f.rating,
                "mood": f.mood,
                "updated_at": when(f.updated_at),
            }
            for f in ratings
        ],
        "watchlist": [
            {"imdb_id": w.imdb_id, "title": w.title, "saved_at": when(w.created_at)} for w in saved
        ],
        "payments": [
            {
                "amount": p.amount_total / 100,
                "currency": p.currency,
                "status": p.status,
                "created_at": when(p.created_at),
            }
            for p in payments
        ],
    }


@router.get("/export", response_model=None)
async def export_data(
    user: User | None = Depends(get_current_user), db: Session = Depends(get_db)
) -> Response:
    if not user:
        return _to_login()
    return JSONResponse(
        export_user_data(db, user),
        headers={"Content-Disposition": 'attachment; filename="terrorreco-data.json"'},
    )


@router.post("/delete")
async def delete_account(
    request: Request,
    current_password: str = Form(...),
    user: User | None = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> RedirectResponse:
    if not user:
        return _to_login()
    if not verify_password(current_password, user.password_hash):
        logger.warning("ACCOUNT DELETE FAIL: wrong current password (user_id=%s)", user.id)
        return _back(request, "Your current password is not correct.", "error")

    user_id = user.id
    # Payments are a financial record and outlive the account, unlinked. Done
    # explicitly because SQLite does not enforce the FK's ON DELETE SET NULL.
    db.query(Payment).filter(Payment.user_id == user_id).update({Payment.user_id: None})
    # ORM delete, so the cascades on history, feedback and watchlist run (and
    # history's cascade removes its result rows). A bulk DELETE would skip them.
    db.delete(user)
    db.commit()
    logger.info("ACCOUNT DELETED: user_id=%s", user_id)

    request.session.clear()
    request.session["flash"] = "Your account and its data have been deleted."
    request.session["flash_type"] = "success"
    return RedirectResponse(url="/", status_code=303)
