"""Transactional email, behind one function.

Sends through Resend's HTTP API when ``RESEND_API_KEY`` is set. When it is not
(development, tests, and production until a provider is chosen) nothing is
sent and a log line says so: features that depend on email check
:func:`email_configured` and stay switched off rather than pretending.
"""

from __future__ import annotations

import logging

import httpx

from ..settings import get_settings

logger = logging.getLogger(__name__)

RESEND_URL = "https://api.resend.com/emails"


def email_configured() -> bool:
    settings = get_settings()
    return bool(settings.RESEND_API_KEY and settings.EMAIL_FROM)


def send_email(to: str, subject: str, text: str) -> bool:
    """Send a plain-text email. Returns whether the provider accepted it.

    Never raises: callers run this after the response has been sent, where an
    exception would be lost anyway. The recipient is not logged.
    """
    settings = get_settings()
    if not email_configured():
        logger.warning("EMAIL NOT SENT: no provider configured (subject: %s)", subject)
        return False
    try:
        response = httpx.post(
            RESEND_URL,
            headers={"Authorization": f"Bearer {settings.RESEND_API_KEY}"},
            json={"from": settings.EMAIL_FROM, "to": [to], "subject": subject, "text": text},
            timeout=10.0,
        )
    except httpx.HTTPError:
        logger.exception("EMAIL FAILED: could not reach the provider (subject: %s)", subject)
        return False
    if response.status_code >= 300:
        logger.error(
            "EMAIL FAILED: provider answered %s (subject: %s)", response.status_code, subject
        )
        return False
    logger.info("EMAIL SENT (subject: %s)", subject)
    return True
