from __future__ import annotations

from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser

from research_agent.config import get_settings
from research_agent.errors import RoutingError
from research_agent.logging_utils import log_exception
from research_agent.schemas import RouterDecision

from research_agent.services.llm_service import get_chat_model
from research_agent.types import RouteType


ROUTER_SYSTEM_PROMPT: str = (
    "You are a routing agent for a multi-agent research system. "
    "Decide how the user's query should be answered: "
    "1) 'web' if it needs up-to-date/current information (latest, today, current events, news, prices, sports standings). "
    "2) 'rag' if it should be answered using the provided knowledge base dataset. "
    "3) 'llm' for general reasoning or questions that do not require external web or the knowledge base. "
    "Return ONLY the requested structured output."
)


def decide_route(query: str, *, conversation_hint: Optional[str] = None) -> RouterDecision:
    """
    Decide the routing path for a query using an LLM classifier.
    """
    settings = get_settings()
    _ = settings  # ensures settings validation at startup
    parser = PydanticOutputParser(pydantic_object=RouterDecision)
    format_instructions = parser.get_format_instructions()

    model = None
    try:
        model = get_chat_model()
    except Exception as exc:  # noqa: BLE001
        # Offline fallback if OpenAI key is missing.
        if settings.openai_api_key is None:
            return _heuristic_route(query)
        log_exception("router_get_chat_model", exc, extra={"query": query})
        raise

    suffix = ""
    if conversation_hint:
        suffix = f"\nConversation context hint: {conversation_hint}"

    system = SystemMessage(content=ROUTER_SYSTEM_PROMPT)
    human = HumanMessage(
        content=(
            f"User query:\n{query}\n\n"
            f"{suffix}\n"
            f"{format_instructions}"
        )
    )

    try:
        response = model.invoke([system, human])  # type: ignore[union-attr]
        parsed: RouterDecision = parser.parse(response.content)
        if parsed.route not in {RouteType.LLM, RouteType.WEB, RouteType.RAG}:
            raise RoutingError(
                "Router returned an invalid route",
                details={"route": str(parsed.route)},
            )
        return parsed
    except RoutingError:
        raise
    except Exception as exc:  # noqa: BLE001
        log_exception("router_decide_route", exc, extra={"query": query})
        if settings.openai_api_key is None:
            return _heuristic_route(query)
        raise RoutingError("Failed to decide route") from exc


def _heuristic_route(query: str) -> RouterDecision:
    q = query.lower()
    web_markers = [
        "latest",
        "today",
        "current",
        "news",
        "price",
        "stock",
        "score",
        "standings",
        "weather",
        "events",
    ]
    rag_markers = ["langgraph", "rag", "vector", "knowledge base", "chroma", "openai api"]

    if any(m in q for m in web_markers):
        return RouterDecision(route=RouteType.WEB, rationale="Keyword match (offline heuristic)")

    if any(m in q for m in rag_markers):
        return RouterDecision(route=RouteType.RAG, rationale="Keyword match (offline heuristic)")

    return RouterDecision(route=RouteType.LLM, rationale="Default to LLM (offline heuristic)")

