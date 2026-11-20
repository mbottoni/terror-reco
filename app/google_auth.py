"""Sign in with Google (OpenID Connect, authorisation-code flow with PKCE).

Off unless ``GOOGLE_CLIENT_ID`` and ``GOOGLE_CLIENT_SECRET`` are set.

What is trusted, and why:

* The ``state`` value ties the callback to the browser that started the flow,
  and PKCE ties the code to it as well.
* The code is exchanged, and the profile fetched, by this server talking to
  Google over TLS. Nothing the browser carries is believed beyond the code.
* An account is identified by Google's ``sub``. The address is used only to
  link a first Google sign-in to an existing account, and only when Google
  says the address is verified.

Linking has one trap. If someone registered a victim's address with a password
of their own and never confirmed it, the victim signing in with Google would
land in an account the attacker can also open. So when a Google sign-in links
to an account whose address was never confirmed, the old password is discarded
and every existing session is revoked first.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import secrets
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .auth import _set_flash, get_current_user, normalize_email, revoke_sessions, start_session
from .db import get_db
from .models import User
from .security import NO_PASSWORD
from .settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth/google")

AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
STATE_KEY = "google_state"
VERIFIER_KEY = "google_verifier"
FAILED = "Signing in with Google did not work. Please try again."


def google_enabled() -> bool:
    settings = get_settings()
    return bool(settings.GOOGLE_CLIENT_ID and settings.GOOGLE_CLIENT_SECRET)


def _redirect_uri(request: Request) -> str:
    base = get_settings().PUBLIC_BASE_URL or str(request.base_url).rstrip("/")
    return f"{base}/auth/google/callback"


def _challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def fetch_profile(code: str, verifier: str, redirect_uri: str) -> dict[str, Any] | None:
    """Exchange the code and return Google's profile, or None on any failure."""
    settings = get_settings()
    try:
        token = httpx.post(
            TOKEN_URL,
            data={
                "code": code,
                "client_id": settings.GOOGLE_CLIENT_ID,
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
                "code_verifier": verifier,
            },
            timeout=10.0,
        )
        if token.status_code != 200:
            logger.warning("GOOGLE SIGN-IN: token exchange answered %s", token.status_code)
            return None
        access_token = token.json().get("access_token")
        if not access_token:
            return None
        info = httpx.get(
            USERINFO_URL, headers={"Authorization": f"Bearer {access_token}"}, timeout=10.0
        )
        if info.status_code != 200:
            logger.warning("GOOGLE SIGN-IN: userinfo answered %s", info.status_code)
            return None
        profile: dict[str, Any] = info.json()
        return profile
    except (httpx.HTTPError, ValueError):
        logger.exception("GOOGLE SIGN-IN: could not reach Google")
        return None


def user_for_profile(db: Session, sub: str, email: str) -> tuple[User, bool]:
    """The account for a verified Google identity; ``(user, created)``."""
    user = db.query(User).filter(User.google_sub == sub).first()
    if user is not None:
        return user, False

    user = db.query(User).filter(User.email == email).first()
    if user is not None:
        if user.email_verified_at is None:
            # Nobody ever proved they own this address, so whoever set the
            # password may not be the person Google just vouched for.
            user.password_hash = NO_PASSWORD
            revoke_sessions(db, user)
            logger.info("GOOGLE LINK: discarded unconfirmed password (user_id=%s)", user.id)
            user.email_verified_at = datetime.now(UTC)
        user.google_sub = sub
        db.commit()
        logger.info("GOOGLE LINKED: user_id=%s", user.id)
        return user, False

    user = User(
        email=email,
        password_hash=NO_PASSWORD,
        google_sub=sub,
        email_verified_at=datetime.now(UTC),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info("REGISTER OK via Google: user_id=%s", user.id)
    return user, True


@router.get("")
def start(request: Request, db: Session = Depends(get_db)) -> RedirectResponse:
    if not google_enabled():
        raise HTTPException(status_code=404)
    if get_current_user(request, db):
        return RedirectResponse(url="/", status_code=303)
    state = secrets.token_urlsafe(24)
    verifier = secrets.token_urlsafe(48)
    request.session[STATE_KEY] = state
    request.session[VERIFIER_KEY] = verifier
    query = urlencode(
        {
            "client_id": get_settings().GOOGLE_CLIENT_ID,
            "redirect_uri": _redirect_uri(request),
            "response_type": "code",
            "scope": "openid email",
            "state": state,
            "code_challenge": _challenge(verifier),
            "code_challenge_method": "S256",
            "prompt": "select_account",
        }
    )
    return RedirectResponse(url=f"{AUTHORIZE_URL}?{query}", status_code=303)


# A plain ``def``: the two calls to Google are synchronous, so FastAPI runs
# this in its threadpool instead of blocking the event loop.
@router.get("/callback")
def callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    db: Session = Depends(get_db),
) -> RedirectResponse:
    if not google_enabled():
        raise HTTPException(status_code=404)
    to_login = RedirectResponse(url="/auth/login", status_code=303)
    expected = request.session.pop(STATE_KEY, None)
    verifier = request.session.pop(VERIFIER_KEY, None)

    if error:
        # The person pressed "cancel" on Google's page, most often.
        _set_flash(request, "Google sign-in was cancelled.", "error")
        return to_login
    if not (code and state and expected and verifier) or not secrets.compare_digest(
        state, expected
    ):
        logger.warning("GOOGLE SIGN-IN: state missing or wrong")
        _set_flash(request, FAILED, "error")
        return to_login

    profile = fetch_profile(code, verifier, _redirect_uri(request))
    sub = str((profile or {}).get("sub") or "")
    email = normalize_email(str((profile or {}).get("email") or ""))
    if not profile or not sub or not email:
        _set_flash(request, FAILED, "error")
        return to_login
    if profile.get("email_verified") is not True:
        logger.warning("GOOGLE SIGN-IN: address not verified by Google")
        _set_flash(
            request, "Google has not verified that email address, so we cannot use it.", "error"
        )
        return to_login

    try:
        user, created = user_for_profile(db, sub, email)
    except IntegrityError:
        db.rollback()  # two callbacks for one new account raced
        _set_flash(request, FAILED, "error")
        return to_login

    start_session(request, user)
    logger.info("LOGIN OK via Google: user_id=%s", user.id)
    _set_flash(request, "Account created. Welcome!" if created else "Signed in successfully.")
    return RedirectResponse(url="/onboarding/" if created else "/", status_code=303)
