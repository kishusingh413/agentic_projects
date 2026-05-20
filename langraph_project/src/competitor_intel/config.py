from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from competitor_intel.errors import ConfigurationError


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    tavily_api_key: str | None = Field(default=None, alias="TAVILY_API_KEY")

    @field_validator("openai_api_key")
    @classmethod
    def key_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("OPENAI_API_KEY must be non-empty")
        return stripped

    @field_validator("tavily_api_key")
    @classmethod
    def tavily_key_trimmed(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


@lru_cache
def get_settings() -> Settings:
    try:
        return Settings()
    except Exception as exc:
        raise ConfigurationError(
            "Invalid configuration. Set OPENAI_API_KEY in the environment or .env file."
        ) from exc
