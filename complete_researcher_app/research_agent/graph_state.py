from __future__ import annotations

from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from research_agent.schemas import FinalResponse, RagFinding, WebFinding
from research_agent.types import RouteType


class ResearchState(TypedDict):
    # Persistent conversation memory.
    # LangGraph uses the `add_messages` annotation to automatically append messages.
    messages: Annotated[list[AnyMessage], add_messages]

    query: str
    thread_id: str

    # Router output
    route: Optional[RouteType]
    rationale: Optional[str]

    # Intermediate results
    web_findings: list[WebFinding]
    rag_findings: list[RagFinding]
    llm_draft: Optional[str]

    # Final result
    final: Optional[FinalResponse]

    # Trace / observability
    trace_id: Optional[str]

