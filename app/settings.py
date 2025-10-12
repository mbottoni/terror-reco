from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
	model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)

	# OMDb only
	OMDB_API_KEY: str | None = Field(default=None)
	OMDB_BASE_URL: str = Field(default="https://www.omdbapi.com/")

	# App
	APP_NAME: str = Field(default="TerrorReco")
	DEBUG: bool = Field(default=False)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
	return AppSettings()  # type: ignore[call-arg]
