from __future__ import annotations

import os
from typing import Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from research_agent.config import get_settings
from research_agent.errors import ExternalAPIError
from research_agent.logging_utils import log_exception


def get_chat_model() -> ChatOpenAI:
    settings = get_settings()

    api_key = settings.openai_api_key
    if api_key is None or not api_key.strip():
        raise ExternalAPIError(
            "OpenAI API key missing",
            details={"env_var": "OPENAI_API_KEY"},
        )

    # Do not rely on env being set; pass explicitly where possible.
    try:
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=api_key,
            temperature=0.2,
        )
    except Exception as exc:  # noqa: BLE001
        log_exception("get_chat_model", exc)
        raise ExternalAPIError("Failed to initialize chat model") from exc


def get_embedding_model() -> OpenAIEmbeddings:
    settings = get_settings()
    api_key = settings.openai_api_key
    if api_key is None or not api_key.strip():
        raise ExternalAPIError(
            "OpenAI API key missing",
            details={"env_var": "OPENAI_API_KEY"},
        )

    try:
        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=api_key,
        )
    except Exception as exc:  # noqa: BLE001
        log_exception("get_embedding_model", exc)
        raise ExternalAPIError("Failed to initialize embedding model") from exc

