from __future__ import annotations

import logging
import re

from fastapi import APIRouter, Depends, Form, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from .db import get_db
from .models import User
from .security import (
    DUMMY_PASSWORD_HASH,
    hash_password,
    require_csrf,
    session_csrf_token,
    validate_csrf_token,
    verify_password,
)

logger = logging.getLogger(__name__)

MIN_PASSWORD_LENGTH = 8


def normalize_email(email: str) -> str:
    """The one form an address is stored and looked up in.

    Without this ``Ana@x.com`` and ``ana@x.com`` are two accounts, and someone
    who registered with a capital letter cannot sign in from a keyboard that
    lowercases it.
    """
    return email.strip().lower()


SESSION_USER_KEY = "user_id"
SESSION_VERSION_KEY = "sv"


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User | None:
    """The signed-in user, or None if the session is absent, stale or revoked."""
    user_id = request.session.get(SESSION_USER_KEY)
    if not user_id:
        return None
    user = db.get(User, user_id)
    # A cookie issued before the user's session_version was bumped is revoked.
    # Cookies from before this field existed carry no version; they count as
    # version 0, so nobody is signed out by the deploy that introduced it.
    if user is None or request.session.get(SESSION_VERSION_KEY, 0) != user.session_version:
        request.session.pop(SESSION_USER_KEY, None)
        request.session.pop(SESSION_VERSION_KEY, None)
        return None
    return user


def start_session(request: Request, user: User) -> None:
    """Sign *user* in on a clean session.

    Whatever the pre-login session held is dropped first, so nothing planted
    in it before authentication survives into the authenticated one.
    """
    request.session.clear()
    request.session[SESSION_USER_KEY] = user.id
    request.session[SESSION_VERSION_KEY] = user.session_version


def revoke_sessions(db: Session, user: User) -> None:
    """Invalidate every session cookie this user holds, on every device.

    Call on password change, password reset and "sign out everywhere".
    """
    user.session_version += 1
    db.commit()


def _set_flash(request: Request, message: str, kind: str = "success") -> None:
    """Store a flash message with a type (success | error)."""
    request.session["flash"] = message
    request.session["flash_type"] = kind


router = APIRouter(prefix="/auth")


@router.get("/login", response_class=HTMLResponse, response_model=None)
async def login_form(request: Request, db: Session = Depends(get_db)) -> Response:
    # Redirect if already logged in
    if get_current_user(request, db):
        return RedirectResponse(url="/", status_code=303)

    csrf = session_csrf_token(request)
    flash = request.session.pop("flash", None)
    flash_type = request.session.pop("flash_type", "success")
    email = request.session.pop("auth_email", "")
    resp: Response = request.app.state.templates.TemplateResponse(
        request,
        "login.html",
        {
            "csrf": csrf,
            "flash": flash,
            "flash_type": flash_type,
            "email": email,
        },
    )
    return resp


@router.post("/login")
async def login(
    request: Request,
    response: Response,
    email: str = Form(...),
    password: str = Form(...),
    csrf: str = Form(...),
    db: Session = Depends(get_db),
) -> RedirectResponse:
    # Preserve email so the form can be pre-filled on error
    request.session["auth_email"] = email
    email = normalize_email(email)

    # Log lines here never carry the address: it is personal data, and which
    # addresses fail to sign in is exactly what an attacker wants from a log.
    if not validate_csrf_token(csrf, request.session.get("csrf")):
        logger.warning("LOGIN FAIL: CSRF token invalid")
        _set_flash(request, "Security check failed. Please try again.", "error")
        return RedirectResponse(url="/auth/login", status_code=303)

    user = db.query(User).filter(User.email == email).first()
    # Always spend one Argon2 verification. Returning early for an unknown
    # address made "no such user" tens of milliseconds faster than "wrong
    # password", which reveals who has an account despite the shared message.
    password_ok = verify_password(password, user.password_hash if user else DUMMY_PASSWORD_HASH)
    if not user or not password_ok:
        logger.warning("LOGIN FAIL: invalid credentials")
        _set_flash(request, "Invalid email or password.", "error")
        return RedirectResponse(url="/auth/login", status_code=303)

    start_session(request, user)
    logger.info("LOGIN OK: user_id=%s", user.id)
    _set_flash(request, "Signed in successfully.")
    return RedirectResponse(url="/", status_code=303)


@router.get("/register", response_class=HTMLResponse, response_model=None)
async def register_form(request: Request, db: Session = Depends(get_db)) -> Response:
    # Redirect if already logged in
    if get_current_user(request, db):
        return RedirectResponse(url="/", status_code=303)

    csrf = session_csrf_token(request)
    flash = request.session.pop("flash", None)
    flash_type = request.session.pop("flash_type", "success")
    email = request.session.pop("auth_email", "")
    resp: Response = request.app.state.templates.TemplateResponse(
        request,
        "register.html",
        {
            "csrf": csrf,
            "flash": flash,
            "flash_type": flash_type,
            "email": email,
        },
    )
    return resp


@router.post("/register")
async def register(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(""),
    csrf: str = Form(...),
    db: Session = Depends(get_db),
) -> RedirectResponse:
    request.session["auth_email"] = email
    email = normalize_email(email)

    if not validate_csrf_token(csrf, request.session.get("csrf")):
        logger.warning("REGISTER FAIL: CSRF invalid")
        _set_flash(request, "Security check failed. Please try again.", "error")
        return RedirectResponse(url="/auth/register", status_code=303)

    # Validate email format
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        logger.warning("REGISTER FAIL: invalid email format")
        _set_flash(request, "Please enter a valid email address.", "error")
        return RedirectResponse(url="/auth/register", status_code=303)

    # Validate password length
    if len(password) < MIN_PASSWORD_LENGTH:
        logger.warning("REGISTER FAIL: password too short")
        _set_flash(
            request,
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters.",
            "error",
        )
        return RedirectResponse(url="/auth/register", status_code=303)

    # Validate password confirmation
    if password != confirm_password:
        logger.warning("REGISTER FAIL: passwords do not match")
        _set_flash(request, "Passwords do not match.", "error")
        return RedirectResponse(url="/auth/register", status_code=303)

    existing = db.query(User).filter(User.email == email).first()
    if existing:
        logger.warning("REGISTER FAIL: address already registered (user_id=%s)", existing.id)
        _set_flash(request, "Email already registered. Please sign in.", "error")
        return RedirectResponse(url="/auth/login", status_code=303)

    user = User(email=email, password_hash=hash_password(password))
    db.add(user)
    db.commit()
    db.refresh(user)

    start_session(request, user)
    logger.info("REGISTER OK: user_id=%s", user.id)
    _set_flash(request, "Account created. Welcome!")
    return RedirectResponse(url="/", status_code=303)


@router.post("/logout", dependencies=[Depends(require_csrf)])
async def logout(request: Request) -> RedirectResponse:
    request.session.clear()
    # Set flash AFTER clear so it survives the redirect
    request.session["flash"] = "You've been signed out."
    request.session["flash_type"] = "success"
    return RedirectResponse(url="/", status_code=303)


<<<<<<< HEAD
@router.post("/logout-all", dependencies=[Depends(require_csrf)])
=======
@router.post("/logout-all")
>>>>>>> origin/main
async def logout_everywhere(
    request: Request,
    user: User | None = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> RedirectResponse:
    """Sign out of every device, including this one."""
    if user:
        revoke_sessions(db, user)
        logger.info("SESSIONS REVOKED: user_id=%s", user.id)
    request.session.clear()
    request.session["flash"] = "You've been signed out everywhere."
    request.session["flash_type"] = "success"
    return RedirectResponse(url="/", status_code=303)
