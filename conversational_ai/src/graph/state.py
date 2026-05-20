"""Graph state schema."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict, total=False):
    """Mutable LangGraph state for one conversational turn."""

    messages: Annotated[list[AnyMessage], add_messages]
    search_snippets: list[str]
    location_focus: str
    error_code: str
    error_message: str
