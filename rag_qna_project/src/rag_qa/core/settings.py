from __future__ import annotations

import os
from typing import Literal

from pydantic import Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag_qa.core.errors import ConfigurationError


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # If empty, system falls back to extractive answering.
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    model_name: str = Field(default="gpt-4o-mini", alias="RAG_MODEL_NAME")
    llm_mode: Literal["openai", "extractive"] = Field(default="extractive", alias="RAG_LLM_MODE")
    chunk_size: int = Field(default=900, alias="RAG_CHUNK_SIZE", ge=200, le=3000)
    chunk_overlap: int = Field(default=120, alias="RAG_CHUNK_OVERLAP", ge=0, le=500)

    @model_validator(mode="after")
    def validate_combinations(self) -> "AppSettings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE.")
        if self.llm_mode == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when RAG_LLM_MODE=openai.")
        return self


def load_settings() -> AppSettings:
    try:
        return AppSettings()
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid configuration: {exc}") from exc
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise ConfigurationError(f"Failed to load settings: {exc}") from exc


def mask_api_key_for_logs(api_key: str | None) -> str:
    if api_key is None or not api_key.strip():
        return "not-set"
    cleaned_key: str = api_key.strip()
    if len(cleaned_key) <= 8:
        return "****"
    return f"{cleaned_key[:4]}...{cleaned_key[-4:]}"


def export_settings_debug(settings: AppSettings) -> dict[str, str]:
    return {
        "llm_mode": settings.llm_mode,
        "model_name": settings.model_name,
        "chunk_size": str(settings.chunk_size),
        "chunk_overlap": str(settings.chunk_overlap),
        "openai_key": mask_api_key_for_logs(settings.openai_api_key),
        "python_env": os.environ.get("VIRTUAL_ENV", "system"),
    }
