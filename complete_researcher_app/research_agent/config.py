from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from research_agent.errors import ConfigurationError


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL"
    )

    # Tavily (web search)
    tavily_api_key: Optional[str] = Field(default=None, alias="TAVILY_API_KEY")

    # RAG
    chroma_persist_dir: str = Field(default="./chroma_db", alias="CHROMA_PERSIST_DIR")
    knowledge_base_dir: str = Field(
        default="./data/knowledge_base", alias="KNOWLEDGE_BASE_DIR"
    )

    # Web
    web_max_results: int = Field(default=5, alias="WEB_MAX_RESULTS")
    web_max_chars_per_snippet: int = Field(
        default=2500, alias="WEB_MAX_CHARS"
    )

    # RAG
    rag_top_k: int = Field(default=4, alias="RAG_TOP_K")

    # Memory / checkpointing
    checkpoint_db: str = Field(default="./checkpoints.sqlite", alias="CHECKPOINT_DB")


_cached_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _cached_settings
    if _cached_settings is not None:
        return _cached_settings

    settings = Settings()

    # Validate critical env vars explicitly (do not trust blindly)
    # If the key is missing, we run in an offline fallback mode.
    if settings.openai_api_key is not None and not settings.openai_api_key.strip():
        raise ConfigurationError(
            "OPENAI_API_KEY must be non-empty if provided",
            details={"env_var": "OPENAI_API_KEY"},
        )

    if settings.tavily_api_key is not None and not settings.tavily_api_key.strip():
        raise ConfigurationError(
            "TAVILY_API_KEY must be non-empty if provided",
            details={"env_var": "TAVILY_API_KEY"},
        )

    # Validate numeric bounds
    if settings.web_max_results < 1 or settings.web_max_results > 20:
        raise ConfigurationError(
            "WEB_MAX_RESULTS out of bounds",
            details={"value": settings.web_max_results},
        )
    if settings.rag_top_k < 1 or settings.rag_top_k > 10:
        raise ConfigurationError(
            "RAG_TOP_K out of bounds",
            details={"value": settings.rag_top_k},
        )

    # Validate and normalize directories
    kb_dir = Path(settings.knowledge_base_dir)
    if not kb_dir.exists():
        # Create knowledge base directory as it is part of app setup
        kb_dir.mkdir(parents=True, exist_ok=True)

    chroma_dir = Path(settings.chroma_persist_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    chk_path = Path(settings.checkpoint_db)
    if chk_path.parent and str(chk_path.parent) != ".":
        chk_path.parent.mkdir(parents=True, exist_ok=True)

    _cached_settings = settings
    return settings

