"""CLI entry: conversational competitor intelligence via LangGraph."""

from __future__ import annotations

import logging
import sys
import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from src.config import get_settings
from src.errors import ConfigurationError
from src.graph.workflow import build_compiled_graph

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def _print_welcome() -> None:
    print(
        "Clothing competitor intelligence (LangGraph)\n"
        "Ask for a report on nearby apparel competitors, footfall, and peak-hour signals.\n"
        "Commands: /quit — exit; /reset — new conversation thread.\n"
    )


def _extract_reply(messages: list[object]) -> str:
    if not messages:
        return "No response was produced."
    last = messages[-1]
    content = getattr(last, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts) if parts else str(content)
    return str(content)


def run() -> None:
    load_dotenv()
    setup_logging()
    try:
        get_settings()
    except ConfigurationError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)

    app = build_compiled_graph()
    thread_id = str(uuid.uuid4())
    config: dict[str, dict[str, str]] = {"configurable": {"thread_id": thread_id}}

    _print_welcome()
    while True:
        try:
            raw = input("You: ")
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break

        user = raw.strip()
        if not user:
            continue
        lowered = user.lower()
        if lowered in {"/quit", "/exit", "quit", "exit"}:
            print("Goodbye.")
            break
        if lowered == "/reset":
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            print("Started a new thread.\n")
            continue

        try:
            result = app.invoke(
                {"messages": [HumanMessage(content=user)]},
                config=config,
            )
        except Exception as exc:
            logger.exception(
                "Graph invocation failed",
                extra={"action": "invoke", "thread_id": thread_id},
            )
            print(f"\nAssistant: Something went wrong: {exc}\n")
            continue

        messages = result.get("messages")
        if not isinstance(messages, list):
            messages = []
        print(f"\nAssistant:\n{_extract_reply(messages)}\n")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
