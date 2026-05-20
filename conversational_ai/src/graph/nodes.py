"""LangGraph node callables — orchestration only; logic lives in services."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from src.config import get_settings
from src.errors import AppError
from src.graph.state import GraphState
from src.services.plan_service import extract_query_plan
from src.services.report_service import synthesize_report
from src.services.search_service import run_searches

logger = logging.getLogger(__name__)


def _latest_human_text(state: GraphState) -> str:
    messages = state.get("messages") or []
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            if isinstance(content, str) and content.strip():
                return content.strip()
    return ""


def plan_and_search(state: GraphState) -> dict[str, Any]:
    """Plan queries from conversation, run search, stash snippets."""
    settings = get_settings()
    messages = list(state.get("messages") or [])
    try:
        plan = extract_query_plan(settings=settings, messages=messages)
        snippets = run_searches(plan.search_queries, settings)
    except AppError as exc:
        logger.error(
            "plan_and_search failed",
            extra={"code": exc.code, "message": str(exc)},
        )
        return {
            "error_code": exc.code,
            "error_message": str(exc),
            "search_snippets": [],
            "location_focus": _latest_human_text(state) or "unspecified",
        }
    except Exception as exc:
        logger.exception("Unexpected error in plan_and_search")
        return {
            "error_code": "UNEXPECTED_ERROR",
            "error_message": str(exc),
            "search_snippets": [],
            "location_focus": "unspecified",
        }

    return {
        "search_snippets": snippets,
        "location_focus": plan.location_focus,
        "error_code": "",
        "error_message": "",
    }


def write_report(state: GraphState) -> dict[str, Any]:
    """Synthesize markdown report and append assistant message."""
    settings = get_settings()
    messages = list(state.get("messages") or [])
    snippets = state.get("search_snippets") or []
    location = (state.get("location_focus") or "").strip() or "unspecified"
    err_code = (state.get("error_code") or "").strip()
    err_msg = (state.get("error_message") or "").strip()

    if err_code and not snippets:
        body = (
            "I could not complete competitor research.\n\n"
            f"**Reason** ({err_code}): {err_msg}\n\n"
            "Try specifying a neighborhood and city, check your API keys, "
            "or add `TAVILY_API_KEY` for more reliable search."
        )
        return {"messages": [AIMessage(content=body)]}

    try:
        report = synthesize_report(
            settings=settings,
            conversation=messages,
            location_focus=location,
            search_snippets=snippets,
        )
    except AppError as exc:
        logger.error(
            "write_report failed",
            extra={"code": exc.code, "message": str(exc)},
        )
        body = (
            "Report generation failed.\n\n"
            f"**Reason** ({exc.code}): {exc}"
        )
        return {"messages": [AIMessage(content=body)]}
    except Exception as exc:
        logger.exception("Unexpected error in write_report")
        body = f"Unexpected error while generating the report: {exc}"
        return {"messages": [AIMessage(content=body)]}

    return {"messages": [AIMessage(content=report)]}
