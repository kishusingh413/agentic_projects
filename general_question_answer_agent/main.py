"""CLI entry: load config, build the LangGraph agent, run a query."""

from __future__ import annotations

import argparse
import logging
import sys

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage

from app.agent_graph import build_agent_graph, initial_messages
from app.config import get_settings
from app.errors import ConfigurationError
from app.services.llm_factory import build_chat_model

logger = logging.getLogger(__name__)


def _last_ai_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text")
                        if isinstance(text, str):
                            parts.append(text)
                if parts:
                    return "\n".join(parts)
    return ""


def run_query(user_text: str) -> str:
    """Execute one user turn and return the assistant's final text."""
    get_settings()
    llm = build_chat_model()
    graph = build_agent_graph(llm)
    result = graph.invoke({"messages": initial_messages(user_text)})
    messages = result.get("messages", [])
    if not isinstance(messages, list):
        return ""
    text = _last_ai_text(messages)
    return text if text else "(No assistant text returned.)"


def main() -> int:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="General Q&A agent with LangGraph math tools (Groq, Gemini, Ollama, or OpenAI).",
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Question to ask (if omitted, reads one line from stdin)",
    )
    args = parser.parse_args()

    if args.query:
        user_text = " ".join(args.query).strip()
    else:
        user_text = sys.stdin.readline().strip()

    if not user_text:
        logger.error("No query provided.")
        return 2

    try:
        answer = run_query(user_text)
    except ConfigurationError as exc:
        logger.error("Configuration error: %s", exc.message)
        return 1
    except Exception:
        logger.exception("Run failed")
        return 1

    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
