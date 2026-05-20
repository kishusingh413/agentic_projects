from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from research_agent.types import RouteType


class RouterDecision(BaseModel):
    route: RouteType = Field(..., description="Where to answer from")
    rationale: str = Field(..., description="Short explanation for the routing choice")


class WebQueryInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    max_results: int = Field(ge=1, le=20)


class WebFinding(BaseModel):
    title: str
    url: str
    snippet: str
    extracted_text: Optional[str] = None


class RagQueryInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    top_k: int = Field(ge=1, le=10)


class RagFinding(BaseModel):
    source_path: str
    doc_id: str
    text: str
    score: float


class SummarizationInput(BaseModel):
    query: str
    router_route: RouteType
    web_findings: list[WebFinding] = []
    rag_findings: list[RagFinding] = []
    llm_draft: Optional[str] = None


class FinalResponse(BaseModel):
    answer_markdown: str
    sources: list[str] = []
    debug: Optional[dict[str, str]] = None


class AppInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    thread_id: str = Field(..., min_length=1, max_length=128)


class AppState(BaseModel):
    # The query currently being handled
    query: str
    # Conversation identity for memory
    thread_id: str
    # Router output
    route: Optional[RouteType] = None
    rationale: Optional[str] = None
    # Intermediate information
    web_findings: list[WebFinding] = []
    rag_findings: list[RagFinding] = []
    llm_draft: Optional[str] = None
    # Final output
    final: Optional[FinalResponse] = None
    # For debugging/traceability
    trace_id: Optional[str] = None

