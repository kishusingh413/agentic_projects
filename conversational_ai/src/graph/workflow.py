"""Compile the LangGraph workflow."""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.graph.nodes import plan_and_search, write_report
from src.graph.state import GraphState


def build_compiled_graph():
    """Build and compile the competitor-intelligence graph with in-memory checkpointing."""
    graph = StateGraph(GraphState)
    graph.add_node("research", plan_and_search)
    graph.add_node("report", write_report)
    graph.add_edge(START, "research")
    graph.add_edge("research", "report")
    graph.add_edge("report", END)
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)
