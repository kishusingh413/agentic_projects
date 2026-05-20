"""Derive structured search plans from conversational context."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.config import Settings
from src.errors import ReportGenerationError

logger = logging.getLogger(__name__)


class AudienceSegment(str, Enum):
    """Who primarily benefits from the answer."""

    OWNER_MANAGER = "owner_manager"
    MARKETING = "marketing"
    LOCATION_ANALYST = "location_analyst"
    INVESTOR = "investor"
    GENERAL = "general"


class QueryPlan(BaseModel):
    """Structured plan for competitor research searches."""

    location_focus: str = Field(
        description="Geographic focus, e.g. 'Koramangala, Bangalore' or 'user-specified area'",
    )
    audience: AudienceSegment = Field(
        default=AudienceSegment.GENERAL,
        description="Tailor emphasis to this stakeholder group when relevant",
    )
    search_queries: list[str] = Field(
        min_length=1,
        max_length=6,
        description="Distinct web search strings about apparel competitors, footfall, peak hours",
    )
    notes_for_analyst: str = Field(
        default="",
        description="Short internal note on ambiguity, follow-ups, or assumptions",
    )


PLANNER_SYSTEM = """You plan web searches for clothing-store competitive intelligence.

Given the conversation, infer:
- The target neighborhood, city, or mall context (ask yourself what location to emphasize).
- 2 to 6 precise search queries that would surface: nearby apparel competitors, shopping \
streets or malls, foot traffic or busy-hour clues (reviews, blogs, local guides, news).

Queries should be in English unless the user writes in another language (then mirror their language).
Do not include personal data or instructions to break laws. Prefer neutral, factual phrasing.
"""


def extract_query_plan(
    *,
    settings: Settings,
    messages: list[BaseMessage],
) -> QueryPlan:
    """Use structured output to build search queries from chat history."""
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.openai_base_url,
        temperature=0,
    )
    structured = llm.with_structured_output(QueryPlan)
    planner_messages: list[BaseMessage] = [
        SystemMessage(content=PLANNER_SYSTEM),
        *messages,
        HumanMessage(
            content=(
                "Produce the QueryPlan JSON for the latest user request. "
                "If location is missing, infer from context or use a reasonable default "
                "mentioned earlier in the thread; if impossible, set location_focus to "
                "'unspecified' and craft queries that help disambiguate."
            )
        ),
    ]
    try:
        result: Any = structured.invoke(planner_messages)
    except Exception as exc:
        logger.exception("Query planning LLM call failed")
        raise ReportGenerationError(f"Query planning failed: {exc}") from exc

    if isinstance(result, QueryPlan):
        return result
    raise ReportGenerationError("Structured planner returned an unexpected type")
