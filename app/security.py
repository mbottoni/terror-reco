from __future__ import annotations

import hmac
import secrets
from hashlib import sha256

from argon2 import PasswordHasher

from .settings import get_settings

_ph = PasswordHasher()


def hash_password(password: str) -> str:
    return _ph.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        _ph.verify(password_hash, password)
        return True
    except Exception:
        return False


def safe_url(url: str | None) -> str:
    """Return *url* if it is safe to put in ``src``/``href``, else ``""``.

    Jinja's autoescaping makes a value safe as *text*; it does nothing about
    the *scheme*, so ``src="{{ poster_url }}"`` will happily render
    ``javascript:alert(1)`` as a live handler. Poster URLs reach us from TMDB
    and -- via ``/watchlist/toggle`` -- straight from a request body, so the
    scheme has to be checked rather than assumed.

    Only absolute http(s) URLs and same-origin paths are allowed. Anything
    else (``javascript:``, ``data:``, ``vbscript:``, a protocol-relative
    ``//host``) returns empty, which renders as a broken image rather than as
    code.
    """
    candidate = (url or "").strip()
    if not candidate:
        return ""
    lowered = candidate.lower()
    if lowered.startswith(("http://", "https://")):
        return candidate
    # A single leading slash is same-origin; "//host" is protocol-relative and
    # points somewhere else entirely.
    if candidate.startswith("/") and not candidate.startswith("//"):
        return candidate
    return ""


def generate_csrf_token() -> str:
    secret = get_settings().SECRET_KEY.encode()
    salt = secrets.token_hex(16)
    sig = hmac.new(secret, msg=salt.encode(), digestmod=sha256).hexdigest()
    return f"{salt}.{sig}"


def validate_csrf_token(token: str, session_token: str | None = None) -> bool:
    """Verify a CSRF token's signature *and* that it belongs to this session.

    The signature check alone is not sufficient: every token is signed with
    the same application secret, so a token an attacker mints in their own
    session verifies perfectly in a victim's.  Comparing against the value
    stashed in the victim's session is what actually binds the two together
    and makes the token unforgeable by a third party.
    """
    if not token:
        return False
    try:
        salt, sig = token.split(".", 1)
    except ValueError:
        return False

    secret = get_settings().SECRET_KEY.encode()
    expected = hmac.new(secret, msg=salt.encode(), digestmod=sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return False

    if session_token is None:
        return False
    return hmac.compare_digest(token, session_token)
