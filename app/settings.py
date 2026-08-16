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

    # OMDb (detail enrichment: IMDb rating/votes, Metascore, awards)
    OMDB_API_KEY: str | None = Field(default=None)
    OMDB_BASE_URL: str = Field(default="https://www.omdbapi.com/")

    # TMDB (corpus discovery: real genre browsing, unlike OMDb's title-only search)
    TMDB_API_KEY: str | None = Field(default=None)
    TMDB_BEARER_TOKEN: str | None = Field(default=None)
    TMDB_BASE_URL: str = Field(default="https://api.themoviedb.org/3")
    TMDB_IMAGE_BASE: str = Field(default="https://image.tmdb.org/t/p/w500")

    # App
    APP_NAME: str = Field(default="TerrorReco")
    DEBUG: bool = Field(default=False)

    # Unified recommender toggles
    # Make "unified" the strategy used when the UI does not specify one.
    USE_UNIFIED_RECOMMENDER: bool = Field(default=False)
    # Rerank the shortlist with a cross-encoder. Off by default: measured at
    # +0.0026 NDCG (inside run-to-run noise), *worse* precision, and ~1.1s of
    # extra latency per query. See docs/evaluation-baseline.md.
    UNIFIED_USE_CROSS_ENCODER: bool = Field(default=False)
    UNIFIED_DIVERSITY_LAMBDA: float = Field(default=0.7)
    # Blend weights as a JSON object, e.g.
    #   UNIFIED_WEIGHTS='{"semantic": 0.45, "keyword": 0.2, "popularity": 0.25, "recency": 0.1}'
    # Unset uses DEFAULT_WEIGHTS. A grid search over 77 combinations beat the
    # default by only +0.0127 NDCG, which is inside the pipeline's own noise,
    # so the default is deliberately left alone -- see scripts/tune_weights.py.
    UNIFIED_WEIGHTS: dict[str, float] | None = Field(default=None)

    # Personalisation from like/dislike feedback
    PERSONALIZATION_ENABLED: bool = Field(default=True)
    # Weight of the taste signal (cosine to the mean embedding of liked films)
    # when a logged-in user has rated anything.
    PERSONALIZATION_TASTE_WEIGHT: float = Field(default=0.15)

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
    return AppSettings()
