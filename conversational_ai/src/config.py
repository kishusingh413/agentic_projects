"""Validated environment configuration loaded once at startup."""

from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.errors import ConfigurationError


def _secret_or_none(value: object) -> SecretStr | None:
    if value is None:
        return None
    if isinstance(value, SecretStr):
        if not value.get_secret_value().strip():
            return None
        return value
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, str):
        return SecretStr(value)
    return None


class Settings(BaseSettings):
    """Application settings from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: SecretStr = Field(..., alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    tavily_api_key: SecretStr | None = Field(default=None, alias="TAVILY_API_KEY")

    @field_validator("tavily_api_key", mode="before")
    @classmethod
    def optional_tavily(cls, value: object) -> SecretStr | None:
        return _secret_or_none(value)

    @field_validator("openai_model")
    @classmethod
    def model_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("OPENAI_MODEL must be a non-empty string")
        return stripped


@lru_cache
def get_settings() -> Settings:
    """Return cached, validated settings or raise ConfigurationError."""
    try:
        return Settings()
    except Exception as exc:
        raise ConfigurationError(
            "Invalid environment configuration. "
            "Ensure OPENAI_API_KEY is set and optional vars match expected types. "
            f"Details: {exc}"
        ) from exc
