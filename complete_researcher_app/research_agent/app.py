from __future__ import annotations

import uuid
from typing import Literal

from langchain_core.messages import AnyMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from research_agent.graph_state import ResearchState

from research_agent.agents.llm_agent import llm_reasoning_node
from research_agent.agents.rag_agent import rag_node
from research_agent.agents.router_agent import ingest_user_message_node, router_node
from research_agent.agents.summarization_agent import summarization_node
from research_agent.agents.web_research_agent import web_research_node
from research_agent.types import RouteType


def _route_selector(state: ResearchState) -> Literal["llm", "web", "rag"]:
    route = state.get("route")
    if route is None:
        return "llm"
    if route == RouteType.WEB:
        return "web"
    if route == RouteType.RAG:
        return "rag"
    return "llm"


def build_app():
    graph = StateGraph(ResearchState)

    graph.add_node("ingest_user_message", ingest_user_message_node)
    graph.add_node("router", router_node)
    graph.add_node("web_research", web_research_node)
    graph.add_node("rag_retrieve", rag_node)
    graph.add_node("llm_reasoning", llm_reasoning_node)
    graph.add_node("summarize", summarization_node)

    graph.add_edge(START, "ingest_user_message")
    graph.add_edge("ingest_user_message", "router")

    graph.add_conditional_edges(
        "router",
        _route_selector,
        {
            "web": "web_research",
            "rag": "rag_retrieve",
            "llm": "llm_reasoning",
        },
    )

    graph.add_edge("web_research", "summarize")
    graph.add_edge("rag_retrieve", "summarize")
    graph.add_edge("llm_reasoning", "summarize")
    graph.add_edge("summarize", END)

    # Enable conversational memory via checkpointing.
    # (Current LangGraph version provides in-memory saver here.)
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)
    return app

