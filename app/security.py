from __future__ import annotations

import hmac
import re
import secrets
from hashlib import sha256

from argon2 import PasswordHasher
from fastapi import HTTPException, Request

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


# Stored in ``password_hash`` for an account that has no password (created by
# signing in with Google). Not a valid Argon2 hash, so nothing verifies
# against it.
NO_PASSWORD = "!"


def has_password(password_hash: str) -> bool:
    return password_hash != NO_PASSWORD


# A real Argon2 hash of a random password nobody knows. Login verifies against
# it when the address is unknown, so that path costs the same as a wrong
# password instead of returning measurably sooner.
DUMMY_PASSWORD_HASH = hash_password(secrets.token_urlsafe(32))


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


_TMDB_IMAGE_RE = re.compile(r"^(https://image\.tmdb\.org/t/p/)w\d+(/.+)$")


def poster_variant(url: str | None, size: str) -> str:
    """The same TMDB poster at another width (``w185``, ``w342``, ``w500``).

    The corpus stores one fully-qualified ``w500`` URL per film. Grid cards
    render far narrower than 500px, so they ask for a smaller rendition of the
    same file. Anything that is not a TMDB image URL comes back unchanged
    (after the usual scheme check), so OMDb posters and test fixtures still
    work.
    """
    safe = safe_url(url)
    match = _TMDB_IMAGE_RE.match(safe)
    if not match or not re.fullmatch(r"w\d+", size):
        return safe
    return f"{match.group(1)}{size}{match.group(2)}"


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


CSRF_SESSION_KEY = "csrf"
CSRF_HEADER = "x-csrf-token"
CSRF_FORM_FIELD = "csrf"


def session_csrf_token(request: Request) -> str:
    """This session's CSRF token, minted on first use.

    One token per session rather than one per form render: regenerating on
    every render meant opening the login page in a second tab silently
    invalidated the form in the first.
    """
    if "session" not in request.scope:
        # Rendered outside the session middleware: the rate limiter's 429
        # page is produced before the session exists. Such a page has no
        # token, and nothing on it needs one.
        return ""
    token = request.session.get(CSRF_SESSION_KEY)
    if not token or not validate_csrf_token(token, token):
        token = generate_csrf_token()
        request.session[CSRF_SESSION_KEY] = token
    return str(token)


async def require_csrf(request: Request) -> None:
    """Dependency for state-changing routes: the request must carry the token.

    ``fetch`` callers send it in the ``X-CSRF-Token`` header (read from the
    ``<meta name="csrf-token">`` every page carries); plain forms send it as a
    hidden ``csrf`` field. ``SameSite=Lax`` on the session cookie already
    blocks cross-site POSTs in current browsers; this is the second lock.
    """
    if request.method in ("GET", "HEAD", "OPTIONS"):
        # Safe methods change nothing, so a router can apply this dependency
        # to all of its routes without breaking its pages.
        return
    token = request.headers.get(CSRF_HEADER)
    if not token:
        content_type = request.headers.get("content-type", "")
        if content_type.startswith(("application/x-www-form-urlencoded", "multipart/form-data")):
            form = await request.form()
            value = form.get(CSRF_FORM_FIELD)
            token = value if isinstance(value, str) else None
    if not validate_csrf_token(token or "", request.session.get(CSRF_SESSION_KEY)):
        raise HTTPException(
            status_code=403, detail="Security check failed. Reload the page and try again."
        )
