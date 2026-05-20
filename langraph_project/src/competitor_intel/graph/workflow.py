import logging
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from competitor_intel.graph.state import CompetitorIntelState
from competitor_intel.services.intent_service import parse_user_intent
from competitor_intel.services.llm_factory import get_chat_model
from competitor_intel.services.report_service import generate_competitor_report
from competitor_intel.services.search_service import format_hits_for_prompt, run_text_searches

logger = logging.getLogger(__name__)


def _latest_human_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            if isinstance(content, str):
                return content.strip()
            return str(content).strip()
    return ""


def node_parse_intent(state: CompetitorIntelState) -> dict[str, object]:
    messages = state.get("messages") or []
    user_text = _latest_human_text(messages)
    if not user_text:
        return {
            "needs_clarification": True,
            "clarification_question": "What location or neighborhood should we analyze?",
            "last_user_text": "",
            "messages": [
                AIMessage(
                    content="Please describe the area (neighborhood and city) you want to analyze."
                )
            ],
        }
    try:
        llm = get_chat_model()
        parsed = parse_user_intent(llm, user_text)
    except Exception as exc:
        logger.exception("Intent parsing failed", extra={"user_text_preview": user_text[:200]})
        return {
            "needs_clarification": True,
            "clarification_question": None,
            "last_user_text": user_text,
            "messages": [
                AIMessage(
                    content=(
                        "Something went wrong while understanding your request. "
                        "Please rephrase with a clear neighborhood and city."
                    )
                )
            ],
        }

    updates: dict[str, object] = {
        "location": parsed.location,
        "store_focus": parsed.store_focus,
        "audience": parsed.audience,
        "needs_clarification": parsed.needs_clarification,
        "clarification_question": parsed.clarification_question,
        "last_user_text": user_text,
    }
    if parsed.needs_clarification:
        question = (
            parsed.clarification_question
            or "Which neighborhood or city should we focus on for clothing competitors?"
        )
        updates["messages"] = [AIMessage(content=question)]
    return updates


def node_search(state: CompetitorIntelState) -> dict[str, object]:
    location = (state.get("location") or "").strip()
    focus = (state.get("store_focus") or "clothing stores").strip()
    if not location:
        return {
            "search_context": "",
            "messages": [
                AIMessage(content="A specific location is required before running competitor search.")
            ],
        }
    queries = [
        f"{location} {focus} stores competitors shopping",
        f"{location} clothing apparel retail stores list",
        f"{location} popular times busy hours retail foot traffic shopping",
        f"{location} mall high street fashion brands footfall",
    ]
    try:
        hits = run_text_searches(queries, max_results_per_query=5)
        context = format_hits_for_prompt(hits, limit=30)
    except Exception as exc:
        logger.exception(
            "Search failed",
            extra={"location": location, "focus": focus, "error": str(exc)},
        )
        return {
            "search_context": "",
            "messages": [
                AIMessage(
                    content=(
                        "Competitor search is temporarily unavailable. "
                        "Please try again in a few minutes."
                    )
                )
            ],
        }
    return {"search_context": context}


def node_synthesize_report(state: CompetitorIntelState) -> dict[str, object]:
    location = (state.get("location") or "").strip()
    context = (state.get("search_context") or "").strip()
    user_request = (state.get("last_user_text") or "").strip()
    if not location:
        return {
            "report": None,
            "messages": [AIMessage(content="Missing location; cannot build a competitor report.")],
        }
    if not context:
        return {
            "report": None,
            "messages": [
                AIMessage(
                    content=(
                        "No usable search results were found for that location. "
                        "Try a broader area name or a nearby landmark."
                    )
                )
            ],
        }
    try:
        llm = get_chat_model()
        report = generate_competitor_report(
            llm,
            location=location,
            store_focus=(state.get("store_focus") or "clothing stores"),
            audience=state.get("audience"),
            search_context=context,
            user_request=user_request or location,
        )
    except Exception as exc:
        logger.exception(
            "Report generation failed",
            extra={"location": location, "error": str(exc)},
        )
        return {
            "report": None,
            "messages": [
                AIMessage(content="Report generation failed. Please try again with a shorter query.")
            ],
        }
    return {"report": report, "messages": [AIMessage(content=report)]}


def route_after_parse(state: CompetitorIntelState) -> Literal["end", "search"]:
    if state.get("needs_clarification"):
        return "end"
    return "search"


def route_after_search(state: CompetitorIntelState) -> Literal["report", "end"]:
    context = (state.get("search_context") or "").strip()
    if not context:
        return "end"
    return "report"


def build_competitor_intel_graph():
    graph = StateGraph(CompetitorIntelState)
    graph.add_node("parse_intent", node_parse_intent)
    graph.add_node("search", node_search)
    graph.add_node("synthesize_report", node_synthesize_report)

    graph.add_edge(START, "parse_intent")
    graph.add_conditional_edges(
        "parse_intent",
        route_after_parse,
        {"end": END, "search": "search"},
    )
    graph.add_conditional_edges(
        "search",
        route_after_search,
        {"report": "synthesize_report", "end": END},
    )
    graph.add_edge("synthesize_report", END)
    return graph.compile()


_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_competitor_intel_graph()
    return _compiled_graph
