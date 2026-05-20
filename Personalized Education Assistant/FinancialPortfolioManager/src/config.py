"""Validated environment configuration (startup only)."""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Loads and validates environment variables required for the LLM client."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(
        ...,
        alias="OPENAI_API_KEY",
        description="API key for OpenAI-compatible chat completions.",
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        alias="OPENAI_MODEL",
        description="Chat model name for agents and group chat selection.",
    )

    @field_validator("openai_api_key")
    @classmethod
    def strip_api_key(cls, value: str) -> str:
        stripped = value.strip()
        if len(stripped) < 8:
            raise ValueError("OPENAI_API_KEY appears invalid (too short).")
        return stripped


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings()
