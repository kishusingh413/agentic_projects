from __future__ import annotations

from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from research_agent.errors import ExternalAPIError
from research_agent.logging_utils import log_exception
from research_agent.services.llm_service import get_chat_model
from research_agent.config import get_settings


def generate_llm_draft(query: str, *, conversation_hint: Optional[str] = None) -> str:
    if not query.strip():
        raise ExternalAPIError("Empty query passed to generate_llm_draft")

    settings = get_settings()
    if settings.openai_api_key is None:
        # Offline fallback: produce a minimal deterministic draft.
        hint = f" (Context: {conversation_hint})" if conversation_hint else ""
        return (
            f"Draft answer for: `{query}`{hint}\n\n"
            "Offline mode: no LLM is available. "
            "The summarization step will format any retrieved web/RAG evidence."
        )

    model = get_chat_model()
    system = SystemMessage(
        content=(
            "You are a helpful assistant. Draft a structured answer to the user's question. "
            "Do not include sources yet; the summarization agent will handle formatting and citations. "
            "Keep it concise but complete."
        )
    )
    hint_text = f"\nConversation hint: {conversation_hint}\n" if conversation_hint else ""
    human = HumanMessage(content=f"User query:\n{query}\n{hint_text}")

    try:
        response = model.invoke([system, human])
        content = str(getattr(response, "content", "")).strip()
        if not content:
            return "I could not generate a draft response."
        return content
    except ExternalAPIError:
        raise
    except Exception as exc:  # noqa: BLE001
        log_exception("llm_draft", exc, extra={"query": query})
        raise ExternalAPIError("LLM draft generation failed") from exc

