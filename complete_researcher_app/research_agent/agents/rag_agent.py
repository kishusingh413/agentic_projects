from __future__ import annotations

from research_agent.config import get_settings
from research_agent.graph_state import ResearchState
from research_agent.logging_utils import log_exception
from research_agent.schemas import RagQueryInput
from research_agent.services.rag_service import rag_retrieve


def rag_node(state: ResearchState) -> dict[str, object]:
    try:
        settings = get_settings()
        validated = RagQueryInput(query=state["query"], top_k=settings.rag_top_k)
        findings = rag_retrieve(validated.query, top_k=validated.top_k)
        return {"rag_findings": findings}
    except Exception as exc:  # noqa: BLE001
        log_exception("rag_node", exc, extra={"query": state.get("query")})
        raise

