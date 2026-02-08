from __future__ import annotations

import logging
import re

from fastapi import APIRouter, Depends, Form, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from .db import get_db
from .models import User
from .security import generate_csrf_token, hash_password, validate_csrf_token, verify_password

logger = logging.getLogger(__name__)

MIN_PASSWORD_LENGTH = 8


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User | None:
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    return db.get(User, user_id)


def _set_flash(request: Request, message: str, kind: str = "success") -> None:
    """Store a flash message with a type (success | error)."""
    request.session["flash"] = message
    request.session["flash_type"] = kind


router = APIRouter(prefix="/auth")


@router.get("/login", response_class=HTMLResponse, response_model=None)
async def login_form(request: Request, db: Session = Depends(get_db)):
    # Redirect if already logged in
    user_id = request.session.get("user_id")
    if user_id and db.get(User, user_id):
        return RedirectResponse(url="/", status_code=303)

    csrf = generate_csrf_token()
    request.session["csrf"] = csrf
    flash = request.session.pop("flash", None)
    flash_type = request.session.pop("flash_type", "success")
    email = request.session.pop("auth_email", "")
    return request.app.state.templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "csrf": csrf,
            "flash": flash,
            "flash_type": flash_type,
            "email": email,
        },
    )


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

    if not validate_csrf_token(csrf):
        logger.warning("LOGIN FAIL: CSRF token invalid for %s", email)
        _set_flash(request, "Security check failed. Please try again.", "error")
        return RedirectResponse(url="/auth/login", status_code=303)

    user = db.query(User).filter(User.email == email).first()
    if not user:
        logger.warning("LOGIN FAIL: No user found with email %s", email)
        _set_flash(request, "Invalid email or password.", "error")
        return RedirectResponse(url="/auth/login", status_code=303)

    if not verify_password(password, user.password_hash):
        logger.warning("LOGIN FAIL: Wrong password for %s", email)
        _set_flash(request, "Invalid email or password.", "error")
        return RedirectResponse(url="/auth/login", status_code=303)

    # Clear preserved email on success
    request.session.pop("auth_email", None)
    request.session["user_id"] = user.id
    logger.info("LOGIN OK: user_id=%s email=%s", user.id, email)
    _set_flash(request, "Signed in successfully.")
    return RedirectResponse(url="/", status_code=303)


@router.get("/register", response_class=HTMLResponse, response_model=None)
async def register_form(request: Request, db: Session = Depends(get_db)):
    # Redirect if already logged in
    user_id = request.session.get("user_id")
    if user_id and db.get(User, user_id):
        return RedirectResponse(url="/", status_code=303)

    csrf = generate_csrf_token()
    request.session["csrf"] = csrf
    flash = request.session.pop("flash", None)
    flash_type = request.session.pop("flash_type", "success")
    email = request.session.pop("auth_email", "")
    return request.app.state.templates.TemplateResponse(
        "register.html",
        {
            "request": request,
            "csrf": csrf,
            "flash": flash,
            "flash_type": flash_type,
            "email": email,
        },
    )


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
    logger.info(
        "REGISTER attempt: email=%s, pw_len=%d, confirm_len=%d",
        email,
        len(password),
        len(confirm_password),
    )

    if not validate_csrf_token(csrf):
        logger.warning("REGISTER FAIL: CSRF invalid for %s", email)
        _set_flash(request, "Security check failed. Please try again.", "error")
        return RedirectResponse(url="/auth/register", status_code=303)

    # Validate email format
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        logger.warning("REGISTER FAIL: invalid email format %s", email)
        _set_flash(request, "Please enter a valid email address.", "error")
        return RedirectResponse(url="/auth/register", status_code=303)

    # Validate password length
    if len(password) < MIN_PASSWORD_LENGTH:
        logger.warning("REGISTER FAIL: password too short (%d chars)", len(password))
        _set_flash(
            request,
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters.",
            "error",
        )
        return RedirectResponse(url="/auth/register", status_code=303)

    # Validate password confirmation
    if password != confirm_password:
        logger.warning("REGISTER FAIL: passwords don't match for %s", email)
        _set_flash(request, "Passwords do not match.", "error")
        return RedirectResponse(url="/auth/register", status_code=303)

    existing = db.query(User).filter(User.email == email).first()
    if existing:
        logger.warning("REGISTER FAIL: email %s already exists", email)
        _set_flash(request, "Email already registered. Please sign in.", "error")
        return RedirectResponse(url="/auth/login", status_code=303)

    user = User(email=email, password_hash=hash_password(password))
    db.add(user)
    db.commit()
    db.refresh(user)

    # Clear preserved email on success
    request.session.pop("auth_email", None)
    request.session["user_id"] = user.id
    logger.info("REGISTER OK: user_id=%s email=%s", user.id, email)
    _set_flash(request, "Account created. Welcome!")
    return RedirectResponse(url="/", status_code=303)


@router.post("/logout")
async def logout(request: Request) -> RedirectResponse:
    request.session.clear()
    # Set flash AFTER clear so it survives the redirect
    request.session["flash"] = "You've been signed out."
    request.session["flash_type"] = "success"
    return RedirectResponse(url="/", status_code=303)
