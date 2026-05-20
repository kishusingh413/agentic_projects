from __future__ import annotations

from typing import Optional

from langchain_core.messages import AnyMessage

from research_agent.graph_state import ResearchState
from research_agent.logging_utils import log_exception
from research_agent.services.llm_reasoning_service import generate_llm_draft


def _last_user_hint(messages: list[AnyMessage]) -> Optional[str]:
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if isinstance(content, str) and content.strip():
            return content[:300]
    return None


def llm_reasoning_node(state: ResearchState) -> dict[str, object]:
    try:
        messages = state["messages"]
        hint = _last_user_hint(messages)
        draft = generate_llm_draft(state["query"], conversation_hint=hint)
        return {"llm_draft": draft}
    except Exception as exc:  # noqa: BLE001
        log_exception("llm_reasoning_node", exc, extra={"query": state.get("query")})
        raise

