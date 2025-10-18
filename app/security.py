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


def generate_csrf_token() -> str:
    secret = get_settings().SECRET_KEY.encode()
    salt = secrets.token_hex(16)
    sig = hmac.new(secret, msg=salt.encode(), digestmod=sha256).hexdigest()
    return f"{salt}.{sig}"


def validate_csrf_token(token: str) -> bool:
    try:
        salt, sig = token.split(".", 1)
    except ValueError:
        return False
    secret = get_settings().SECRET_KEY.encode()
    expected = hmac.new(secret, msg=salt.encode(), digestmod=sha256).hexdigest()
    return hmac.compare_digest(sig, expected)
