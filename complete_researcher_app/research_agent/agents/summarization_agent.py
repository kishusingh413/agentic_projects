from __future__ import annotations

from typing import Optional

from langchain_core.messages import AIMessage

from research_agent.graph_state import ResearchState
from research_agent.logging_utils import log_exception
from research_agent.schemas import FinalResponse
from research_agent.services.summarization_service import summarize


def summarization_node(state: ResearchState) -> dict[str, object]:
    try:
        route = state.get("route")
        web_findings = state.get("web_findings") or []
        rag_findings = state.get("rag_findings") or []
        llm_draft = state.get("llm_draft")

        if route is None:
            # Default route for summarizer if router didn't set.
            from research_agent.types import RouteType

            route = RouteType.LLM

        final_response: FinalResponse = summarize(
            state["query"],
            route,
            web_findings=web_findings,
            rag_findings=rag_findings,
            llm_draft=llm_draft,
        )

        return {
            "final": final_response,
            "messages": [AIMessage(content=final_response.answer_markdown)],
        }
    except Exception as exc:  # noqa: BLE001
        log_exception("summarization_node", exc, extra={"trace_id": state.get("trace_id")})
        raise

