"""Construct chat models for configured providers."""

from __future__ import annotations

import logging

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import LlmProvider, Settings, get_settings
from app.errors import ConfigurationError

logger = logging.getLogger(__name__)


def build_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """
    Instantiate the chat model for the configured LLM_PROVIDER.

    Raises ConfigurationError if the provider cannot be initialized.
    """
    cfg = settings if settings is not None else get_settings()

    try:
        if cfg.llm_provider is LlmProvider.GROQ:
            from langchain_groq import ChatGroq

            return ChatGroq(
                model=cfg.groq_model,
                api_key=cfg.groq_api_key,
                temperature=0,
            )

        if cfg.llm_provider is LlmProvider.GEMINI:
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=cfg.gemini_model,
                google_api_key=cfg.google_api_key,
                temperature=0,
            )

        if cfg.llm_provider is LlmProvider.OLLAMA:
            from langchain_ollama import ChatOllama

            return ChatOllama(
                base_url=cfg.ollama_base_url,
                model=cfg.ollama_model,
                temperature=0,
            )

        if cfg.llm_provider is LlmProvider.OPENAI:
            from langchain_openai import ChatOpenAI

            base = cfg.openai_base_url.strip() if cfg.openai_base_url else ""
            if base:
                return ChatOpenAI(
                    model=cfg.openai_model,
                    api_key=cfg.openai_api_key,
                    temperature=0,
                    base_url=base,
                )
            return ChatOpenAI(
                model=cfg.openai_model,
                api_key=cfg.openai_api_key,
                temperature=0,
            )
    except Exception as exc:
        logger.exception(
            "Failed to build chat model",
            extra={"provider": cfg.llm_provider.value},
        )
        raise ConfigurationError(
            f"Could not initialize LLM for provider {cfg.llm_provider.value}: {exc}",
        ) from exc

    raise ConfigurationError(f"Unsupported LLM provider: {cfg.llm_provider}")
