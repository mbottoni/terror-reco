"""SECRET_KEY must be real whenever DEBUG is off."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.settings import DEFAULT_SECRET_KEY, MIN_SECRET_KEY_LENGTH, AppSettings

STRONG_KEY = "k" * MIN_SECRET_KEY_LENGTH


BASE_URL = "https://example.com"


def _settings(**overrides: object) -> AppSettings:
    # _env_file=None so a developer's local .env cannot decide the outcome.
    # Production needs a base URL too; tests of other rules supply a valid one.
    if overrides.get("DEBUG") is False:
        overrides.setdefault("PUBLIC_BASE_URL", BASE_URL)
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


class TestPublicBaseUrl:
    def test_required_in_production(self) -> None:
        with pytest.raises(ValidationError, match="PUBLIC_BASE_URL"):
            _settings(DEBUG=False, SECRET_KEY=STRONG_KEY, PUBLIC_BASE_URL=None)

    def test_optional_in_debug(self) -> None:
        assert _settings(DEBUG=True).PUBLIC_BASE_URL is None

    def test_trailing_slash_is_stripped(self) -> None:
        settings = _settings(DEBUG=False, SECRET_KEY=STRONG_KEY, PUBLIC_BASE_URL=BASE_URL + "/")
        assert settings.PUBLIC_BASE_URL == BASE_URL

    @pytest.mark.parametrize(
        "url", ["http://example.com", "example.com", "https://example.com/app", "ftp://x.y"]
    )
    def test_production_rejects_anything_but_an_https_origin(self, url: str) -> None:
        with pytest.raises(ValidationError, match="PUBLIC_BASE_URL"):
            _settings(DEBUG=False, SECRET_KEY=STRONG_KEY, PUBLIC_BASE_URL=url)

    def test_plain_http_allowed_in_debug(self) -> None:
        settings = _settings(DEBUG=True, PUBLIC_BASE_URL="http://localhost:8000")
        assert settings.PUBLIC_BASE_URL == "http://localhost:8000"
