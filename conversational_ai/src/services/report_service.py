"""LLM-backed synthesis of competitor intelligence reports."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import Settings
from src.errors import ReportGenerationError

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an analyst assistant for clothing retail operators. Your job is to \
produce clear, actionable competitor intelligence from web search snippets.

Rules:
- Ground claims in the provided snippets when possible; label uncertainty explicitly.
- Footfall and "busiest hours" are often estimates from reviews, blogs, or data vendors— \
  never present them as verified meter readings unless the snippet states that.
- Structure the final answer as a professional brief with sections: \
  Executive Summary, Key Competitors, Footfall & Peak-Time Signals, Strategic Implications, \
  Data Limitations & Next Steps.
- If the user names a neighborhood (e.g., Koramangala, Bangalore), prioritize that geography.
- Use bullet points where helpful. Keep tone professional and concise.
"""


def build_report_messages(
    *,
    conversation: list[BaseMessage],
    location_focus: str,
    search_snippets: list[str],
) -> list[BaseMessage]:
    """Compose chat messages for report generation."""
    context_block = "\n\n---\n\n".join(search_snippets[:40])
    user_block = (
        f"Location / focus: {location_focus}\n\n"
        f"Web snippets (may be incomplete or contradictory):\n\n{context_block}\n\n"
        "Generate the competitor report following the system rules."
    )
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        *conversation,
        HumanMessage(content=user_block),
    ]


def synthesize_report(
    *,
    settings: Settings,
    conversation: list[BaseMessage],
    location_focus: str,
    search_snippets: list[str],
) -> str:
    """Call the chat model to produce the markdown report."""
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.openai_base_url,
        temperature=0.2,
    )
    messages = build_report_messages(
        conversation=conversation,
        location_focus=location_focus,
        search_snippets=search_snippets,
    )
    try:
        response = llm.invoke(messages)
    except Exception as exc:
        logger.exception("Report generation LLM call failed")
        raise ReportGenerationError(f"LLM report generation failed: {exc}") from exc

    content = getattr(response, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        merged = "\n".join(parts).strip()
        if merged:
            return merged

    logger.error(
        "Unexpected LLM response shape",
        extra={"content_type": type(content).__name__},
    )
    raise ReportGenerationError("Model returned an empty or unsupported response format")
