from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# SECRET_KEY signs the session cookie and every CSRF token, so a deployment
# running on the placeholder has forgeable sessions.
DEFAULT_SECRET_KEY = "change-me-please"
MIN_SECRET_KEY_LENGTH = 32


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
    # libpq sslmode for Postgres. Managed Postgres needs "require"; a Postgres
    # container on the same Docker network has no TLS and needs "disable".
    DATABASE_SSLMODE: str = Field(default="require")
    SECRET_KEY: str = Field(default=DEFAULT_SECRET_KEY)
    SESSION_COOKIE_NAME: str = Field(default="terror_session")
    # The site's canonical origin, e.g. "https://example.com" (no trailing
    # slash). Every absolute URL the app hands to someone else -- Stripe
    # return URLs today; email links and OAuth redirects later -- is built
    # from this, never from the request's Host header. Required in production.
    PUBLIC_BASE_URL: str | None = Field(default=None)

    # Per-client request limits (app/middleware.py). In-process counters, so
    # correct only for a single worker.
    RATE_LIMIT_ENABLED: bool = Field(default=True)

    # Stripe
    STRIPE_PUBLISHABLE_KEY: str | None = Field(default=None)
    STRIPE_SECRET_KEY: str | None = Field(default=None)
    STRIPE_WEBHOOK_SECRET: str | None = Field(default=None)
    COFFEE_PRICE_ID: str | None = Field(default=None)  # Stripe Price ID for coffee

    @model_validator(mode="after")
    def _refuse_weak_secret_in_production(self) -> "AppSettings":
        """Fail at startup rather than serve forgeable sessions.

        Gated on DEBUG so local development and the test suite (which forces
        DEBUG=true in conftest.py) keep working without a real key.
        """
        if self.DEBUG:
            return self
        if self.SECRET_KEY == DEFAULT_SECRET_KEY or len(self.SECRET_KEY) < MIN_SECRET_KEY_LENGTH:
            raise ValueError(
                "SECRET_KEY is unset or too short for production. Set SECRET_KEY to a random "
                f"string of at least {MIN_SECRET_KEY_LENGTH} characters "
                "(python -c 'import secrets; print(secrets.token_urlsafe(48))'), "
                "or set DEBUG=1 for local development."
            )
        return self

    @model_validator(mode="after")
    def _normalise_public_base_url(self) -> "AppSettings":
        """Require a well-formed PUBLIC_BASE_URL whenever DEBUG is off."""
        url = (self.PUBLIC_BASE_URL or "").strip().rstrip("/")
        if not url:
            if self.DEBUG:
                self.PUBLIC_BASE_URL = None
                return self
            raise ValueError(
                "PUBLIC_BASE_URL is required in production: set it to the site's origin, "
                "e.g. PUBLIC_BASE_URL=https://example.com"
            )
        scheme_ok = url.startswith("https://") or (self.DEBUG and url.startswith("http://"))
        if not scheme_ok or "/" in url.split("://", 1)[1]:
            raise ValueError(
                "PUBLIC_BASE_URL must be an origin with no path, e.g. https://example.com "
                "(http:// is accepted only when DEBUG is on)"
            )
        self.PUBLIC_BASE_URL = url
        return self


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
