"""SECRET_KEY must be real whenever DEBUG is off."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.settings import DEFAULT_SECRET_KEY, MIN_SECRET_KEY_LENGTH, AppSettings

STRONG_KEY = "k" * MIN_SECRET_KEY_LENGTH


def _settings(**overrides: object) -> AppSettings:
    # _env_file=None so a developer's local .env cannot decide the outcome.
    return AppSettings(_env_file=None, **overrides)  # type: ignore[arg-type]


class TestSecretKeyGuard:
    def test_default_key_refused_in_production(self) -> None:
        with pytest.raises(ValidationError, match="SECRET_KEY"):
            _settings(DEBUG=False, SECRET_KEY=DEFAULT_SECRET_KEY)

    def test_short_key_refused_in_production(self) -> None:
        with pytest.raises(ValidationError, match="SECRET_KEY"):
            _settings(DEBUG=False, SECRET_KEY="k" * (MIN_SECRET_KEY_LENGTH - 1))

    def test_strong_key_accepted_in_production(self) -> None:
        assert _settings(DEBUG=False, SECRET_KEY=STRONG_KEY).SECRET_KEY == STRONG_KEY

    def test_default_key_allowed_in_debug(self) -> None:
        assert _settings(DEBUG=True, SECRET_KEY=DEFAULT_SECRET_KEY).DEBUG is True
