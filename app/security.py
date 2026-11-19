from __future__ import annotations

import hmac
import re
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
