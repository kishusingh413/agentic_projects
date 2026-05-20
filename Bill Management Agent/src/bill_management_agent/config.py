"""Validated application configuration (startup)."""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from bill_management_agent.errors import ConfigurationError


class Settings(BaseSettings):
    """Environment variables — validated at load time."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    bill_agent_demo: bool = Field(default=False, alias="BILL_AGENT_DEMO")
    bill_use_sidecar_json: bool = Field(default=False, alias="BILL_USE_SIDECAR_JSON")

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def strip_empty_key(cls, value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped if stripped else None
        return None


_settings: Settings | None = None


def get_settings() -> Settings:
    """Singleton settings; raises ConfigurationError if invalid."""
    global _settings
    if _settings is None:
        try:
            _settings = Settings()
        except Exception as exc:
            raise ConfigurationError(f"Invalid configuration: {exc}") from exc
    return _settings


def reset_settings_for_tests() -> None:
    """Clear cached settings (tests only)."""
    global _settings
    _settings = None
