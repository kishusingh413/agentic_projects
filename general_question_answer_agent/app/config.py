"""Load and validate environment-driven settings at startup."""

from __future__ import annotations

import logging
from enum import Enum
from functools import lru_cache
from typing import Self

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.errors import ConfigurationError

logger = logging.getLogger(__name__)


class LlmProvider(str, Enum):
    """Supported LLM backends."""

    AUTO = "auto"
    GROQ = "groq"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    OPENAI = "openai"


class Settings(BaseSettings):
    """Application configuration parsed from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: LlmProvider = Field(
        default=LlmProvider.AUTO,
        alias="LLM_PROVIDER",
        description="Use 'auto' to pick the first provider with credentials (openai, groq, gemini, then ollama).",
    )

    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")

    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    gemini_model: str = Field(default="gemini-2.0-flash", alias="GEMINI_MODEL")

    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.2", alias="OLLAMA_MODEL")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")

    @field_validator("llm_provider", mode="before")
    @classmethod
    def coerce_provider(cls, value: object) -> LlmProvider:
        if isinstance(value, LlmProvider):
            return value
        if isinstance(value, str):
            key = value.strip().lower()
            if key == "":
                return LlmProvider.AUTO
            for member in LlmProvider:
                if member.value == key:
                    return member
            raise ValueError(f"Unknown LLM_PROVIDER: {value!r}")
        raise ValueError(f"Invalid LLM_PROVIDER: {value!r}")

    def _has_openai_credentials(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key.strip())

    def _has_groq_credentials(self) -> bool:
        return bool(self.groq_api_key and self.groq_api_key.strip())

    def _has_gemini_credentials(self) -> bool:
        return bool(self.google_api_key and self.google_api_key.strip())

    def _has_ollama_config(self) -> bool:
        return bool(self.ollama_base_url and self.ollama_base_url.strip())

    def _credentials_match_provider(self, provider: LlmProvider) -> bool:
        if provider is LlmProvider.OPENAI:
            return self._has_openai_credentials()
        if provider is LlmProvider.GROQ:
            return self._has_groq_credentials()
        if provider is LlmProvider.GEMINI:
            return self._has_gemini_credentials()
        if provider is LlmProvider.OLLAMA:
            return self._has_ollama_config()
        return False

    def _pick_first_available_provider(self) -> LlmProvider | None:
        """Prefer cloud APIs in a fixed order; Ollama last (local, no API key)."""
        if self._has_openai_credentials():
            return LlmProvider.OPENAI
        if self._has_groq_credentials():
            return LlmProvider.GROQ
        if self._has_gemini_credentials():
            return LlmProvider.GEMINI
        if self._has_ollama_config():
            return LlmProvider.OLLAMA
        return None

    @model_validator(mode="after")
    def resolve_llm_provider(self) -> Self:
        """Resolve AUTO or fall back when the requested provider has no credentials."""
        requested = self.llm_provider

        if requested is LlmProvider.AUTO:
            resolved = self._pick_first_available_provider()
            if resolved is None:
                raise ConfigurationError(
                    "No LLM credentials found. Set at least one of OPENAI_API_KEY, "
                    "GROQ_API_KEY, GOOGLE_API_KEY, or configure OLLAMA_BASE_URL for local Ollama.",
                )
            logger.info(
                "LLM_PROVIDER is auto; using %s (first available).",
                resolved.value,
            )
            self.llm_provider = resolved
            return self

        if self._credentials_match_provider(requested):
            return self

        fallback = self._pick_first_available_provider()
        if fallback is None:
            raise ConfigurationError(
                f"LLM_PROVIDER is {requested.value} but the matching credentials are missing, "
                "and no other provider credentials were found.",
            )
        logger.warning(
            "LLM_PROVIDER is %s but credentials for that provider are missing; "
            "using %s instead.",
            requested.value,
            fallback.value,
        )
        self.llm_provider = fallback
        return self

    def validate_provider_credentials(self) -> None:
        """Ensure required secrets exist for the resolved provider (after AUTO / fallback)."""
        if self.llm_provider is LlmProvider.AUTO:
            raise ConfigurationError("LLM provider was not resolved; this should not happen.")
        if self.llm_provider is LlmProvider.GROQ:
            if not self._has_groq_credentials():
                raise ConfigurationError(
                    "LLM_PROVIDER is groq but GROQ_API_KEY is missing or empty.",
                )
        elif self.llm_provider is LlmProvider.GEMINI:
            if not self._has_gemini_credentials():
                raise ConfigurationError(
                    "LLM_PROVIDER is gemini but GOOGLE_API_KEY is missing or empty.",
                )
        elif self.llm_provider is LlmProvider.OLLAMA:
            if not self._has_ollama_config():
                raise ConfigurationError(
                    "LLM_PROVIDER is ollama but OLLAMA_BASE_URL is missing or empty.",
                )
        elif self.llm_provider is LlmProvider.OPENAI:
            if not self._has_openai_credentials():
                raise ConfigurationError(
                    "LLM_PROVIDER is openai but OPENAI_API_KEY is missing or empty.",
                )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return validated settings singleton."""
    settings = Settings()
    settings.validate_provider_credentials()
    return settings
