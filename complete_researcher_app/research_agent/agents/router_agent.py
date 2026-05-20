from __future__ import annotations

import uuid
from typing import Optional

from langchain_core.messages import AnyMessage, HumanMessage

from research_agent.graph_state import ResearchState
from research_agent.logging_utils import log_exception
from research_agent.schemas import AppInput
from research_agent.schemas import RouterDecision
from research_agent.services.router_service import decide_route


def _last_user_hint(messages: list[AnyMessage]) -> Optional[str]:
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if isinstance(content, str) and content.strip():
            # We don't reliably know role type across message classes; include a tiny hint.
            return content[:300]
    return None


def ingest_user_message_node(state: ResearchState) -> dict[str, object]:
    trace_id = str(uuid.uuid4())
    validated = AppInput(query=state["query"], thread_id=state["thread_id"])
    # Always append the current user message to memory.
    return {
        "messages": [HumanMessage(content=validated.query)],
        "trace_id": trace_id,
    }


def router_node(state: ResearchState) -> dict[str, object]:
    try:
        trace_id = state.get("trace_id")
        messages = state["messages"]
        hint = _last_user_hint(messages)

        decision: RouterDecision = decide_route(
            state["query"],
            conversation_hint=hint,
        )

        return {
            "route": decision.route,
            "rationale": decision.rationale,
            "trace_id": trace_id,
        }
    except Exception as exc:  # noqa: BLE001
        log_exception("router_node", exc, extra={"trace_id": state.get("trace_id")})
        raise

