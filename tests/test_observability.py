"""App logging is switched on, and error tracking only with a DSN."""

from __future__ import annotations

import logging
from typing import Any

import pytest

import app.observability as obs
from app.observability import configure_error_tracking, configure_logging
from app.settings import AppSettings


def _settings(**overrides: Any) -> AppSettings:
    return AppSettings(_env_file=None, DEBUG=True, **overrides)  # type: ignore[arg-type]


class TestLogging:
    def test_app_loggers_emit_info(self) -> None:
        configure_logging("INFO")
        assert logging.getLogger("app").level == logging.INFO
        assert logging.getLogger("app.auth").isEnabledFor(logging.INFO)

    def test_info_lines_reach_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The regression: INFO lines such as REGISTER OK were silently dropped."""
        logger = logging.getLogger("app")
        for handler in list(logger.handlers):
            if handler.get_name() == obs._HANDLER_NAME:
                logger.removeHandler(handler)
        configure_logging("INFO")  # re-attach to the stderr pytest is capturing
        logging.getLogger("app.account").info("ACCOUNT DELETED: user_id=%s", 7)
        err = capsys.readouterr().err
        assert "INFO app.account: ACCOUNT DELETED: user_id=7" in err

    def test_calling_twice_does_not_duplicate_the_handler(self) -> None:
        configure_logging("INFO")
        configure_logging("INFO")
        handlers = [
            h for h in logging.getLogger("app").handlers if h.get_name() == obs._HANDLER_NAME
        ]
        assert len(handlers) == 1

    def test_level_is_configurable_and_bad_values_fall_back(self) -> None:
        configure_logging("warning")
        assert logging.getLogger("app").level == logging.WARNING
        configure_logging("not-a-level")
        assert logging.getLogger("app").level == logging.INFO

    def test_third_party_loggers_are_left_alone(self) -> None:
        before = logging.getLogger("httpx").level
        configure_logging("INFO")
        assert logging.getLogger("httpx").level == before

    def test_running_app_has_logging_configured(self, client: Any) -> None:
        client.get("/")
        assert logging.getLogger("app.main").isEnabledFor(logging.INFO)


class TestErrorTracking:
    def test_off_without_a_dsn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sentry_sdk

        called: list[Any] = []
        monkeypatch.setattr(sentry_sdk, "init", lambda **kw: called.append(kw))
        assert configure_error_tracking(_settings()) is False
        assert called == []

    def test_on_with_a_dsn_and_without_personal_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sentry_sdk

        seen: dict[str, Any] = {}
        monkeypatch.setattr(sentry_sdk, "init", lambda **kw: seen.update(kw))
        monkeypatch.setenv("IMAGE_TAG", "abc123")
        dsn = "https://public@o0.ingest.sentry.io/0"
        assert configure_error_tracking(_settings(SENTRY_DSN=dsn)) is True
        assert seen["dsn"] == dsn
        assert seen["send_default_pii"] is False
        assert seen["traces_sample_rate"] == 0.0
        assert seen["release"] == "abc123"
        assert seen["environment"] == "development"
