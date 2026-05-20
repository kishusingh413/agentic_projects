"""LangGraph construction: model node, tool node, and conditional routing."""

from __future__ import annotations

import logging
from typing import Annotated, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.math_tools import get_math_tools

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "For any question that involves arithmetic (addition, subtraction, multiplication, or division), "
    "you MUST call the appropriate tool: plus, subtract, multiply, or divide, with numeric arguments. "
    "Infer the numbers from the user's words. "
    "For all other questions, answer directly without using tools."
)


class AgentState(TypedDict):
    """Graph state: conversation messages."""

    messages: Annotated[list[BaseMessage], add_messages]


def build_agent_graph(llm: BaseChatModel) -> CompiledStateGraph:
    """
    Build and compile a LangGraph that routes between the LLM and math tools.

    Flow: START -> agent -> (tools | END); tools -> agent.
    """
    tools = get_math_tools()
    model_with_tools: Runnable = llm.bind_tools(tools)

    def call_model(state: AgentState, config: RunnableConfig) -> dict[str, list[AIMessage]]:
        try:
            response = model_with_tools.invoke(state["messages"], config=config)
        except Exception:
            logger.exception(
                "Model invocation failed",
                extra={"message_count": len(state.get("messages", []))},
            )
            raise
        if not isinstance(response, AIMessage):
            raise TypeError("Expected AIMessage from model")
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")

    return graph.compile()


def initial_messages(user_text: str) -> list[BaseMessage]:
    """Build the first turn with system instructions and the user question."""
    from langchain_core.messages import HumanMessage, SystemMessage

    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_text),
    ]
