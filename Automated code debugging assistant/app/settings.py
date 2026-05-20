"""Validated application settings (environment variables)."""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Load and validate configuration at startup — do not trust raw os.environ."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(
        ...,
        description="API key for the OpenAI-compatible LLM used by CrewAI.",
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="Model name passed to CrewAI agents.",
    )
    code_interpreter_unsafe: bool = Field(
        default=False,
        description="If true, CodeInterpreterTool runs on the host (trusted dev only).",
    )

    @field_validator("openai_api_key")
    @classmethod
    def api_key_not_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("OPENAI_API_KEY must be non-empty")
        return stripped

    @field_validator("code_interpreter_unsafe", mode="before")
    @classmethod
    def parse_bool_flag(cls, value: object) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).strip().lower() in ("1", "true", "yes", "on")


def load_settings() -> AppSettings:
    """Parse and validate settings; raises if required variables are missing or invalid."""
    return AppSettings()
