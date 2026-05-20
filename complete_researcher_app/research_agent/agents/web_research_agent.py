from __future__ import annotations

from research_agent.graph_state import ResearchState
from research_agent.logging_utils import log_exception
from research_agent.schemas import WebQueryInput
from research_agent.config import get_settings
from research_agent.services.web_service import web_search


def web_research_node(state: ResearchState) -> dict[str, object]:
    try:
        settings = get_settings()
        validated = WebQueryInput(
            query=state["query"],
            max_results=settings.web_max_results,
        )
        findings = web_search(validated.query)
        return {"web_findings": findings}
    except Exception as exc:  # noqa: BLE001
        log_exception("web_research_node", exc, extra={"query": state.get("query")})
        raise

