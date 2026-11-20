"""Application logging and optional error tracking.

Two gaps this closes. Nothing configured the app's own loggers, so they
inherited the root logger's WARNING level and every INFO line -- sign-ins,
account deletions, recorded payments, backups of intent -- was silently
dropped. And an unhandled exception left a traceback in a container log that
nobody reads; with ``SENTRY_DSN`` set it is reported instead.
"""

from __future__ import annotations

import logging
import os
import sys

from .settings import AppSettings

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_HANDLER_NAME = "terror-reco-stderr"


def configure_logging(level: str = "INFO") -> None:
    """Send ``app.*`` log records at *level* and above to stderr.

    Only the ``app`` logger is touched: uvicorn keeps its own access and
    error logs, and third-party libraries stay at the root default. Records
    still propagate, so test capture and any root handler keep working.
    Safe to call more than once.
    """
    logger = logging.getLogger("app")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not any(handler.get_name() == _HANDLER_NAME for handler in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.set_name(_HANDLER_NAME)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%dT%H:%M:%S%z"))
        logger.addHandler(handler)


def configure_error_tracking(settings: AppSettings) -> bool:
    """Start Sentry when a DSN is configured. Returns whether it was started.

    Deliberately conservative: no performance tracing, and no request bodies,
    cookies or user details (``send_default_pii=False``), because this app's
    requests carry passwords and session cookies.
    """
    if not settings.SENTRY_DSN:
        return False
    import sentry_sdk

    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        environment="development" if settings.DEBUG else "production",
        # The deploy job exports IMAGE_TAG (the commit sha) into .env.prod.
        release=os.getenv("IMAGE_TAG") or None,
        traces_sample_rate=0.0,
        send_default_pii=False,
    )
    logging.getLogger(__name__).info("Error tracking enabled")
    return True
