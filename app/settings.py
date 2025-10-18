from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        env_file=(".env",),
        env_file_encoding="utf-8",
    )

    # OMDb only
    OMDB_API_KEY: str | None = Field(default=None)
    OMDB_BASE_URL: str = Field(default="https://www.omdbapi.com/")

    # App
    APP_NAME: str = Field(default="TerrorReco")
    DEBUG: bool = Field(default=False)

    # Unified recommender toggles
    USE_UNIFIED_RECOMMENDER: bool = Field(default=False)
    UNIFIED_USE_CROSS_ENCODER: bool = Field(default=False)
    UNIFIED_DIVERSITY_LAMBDA: float = Field(default=0.7)

    # Auth / DB
    DATABASE_URL: str = Field(default="sqlite:///./app.db")
    SECRET_KEY: str = Field(default="change-me-please")
    SESSION_COOKIE_NAME: str = Field(default="terror_session")

    # Stripe
    STRIPE_PUBLISHABLE_KEY: str | None = Field(default=None)
    STRIPE_SECRET_KEY: str | None = Field(default=None)
    STRIPE_WEBHOOK_SECRET: str | None = Field(default=None)
    COFFEE_PRICE_ID: str | None = Field(default=None)  # Stripe Price ID for coffee


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()  # type: ignore[call-arg]
