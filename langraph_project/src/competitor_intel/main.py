import argparse
import asyncio
import logging
import sys
from typing import Sequence

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from competitor_intel.errors import ConfigurationError
from competitor_intel.graph.workflow import get_compiled_graph
from competitor_intel.logging_config import configure_logging

logger = logging.getLogger(__name__)


def _print_assistant_reply(messages: Sequence[BaseMessage]) -> None:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str):
                text = content
            else:
                text = str(content)
            print("\nAssistant:\n")
            print(text.strip())
            return
    print("\nAssistant:\n(no reply generated)\n")


def _run_turn(history: list[BaseMessage]) -> list[BaseMessage]:
    graph = get_compiled_graph()
    try:
        result = graph.invoke({"messages": list(history)})
    except Exception:
        logger.exception("Graph invocation failed")
        fallback = AIMessage(
            content=(
                "The request could not be completed due to an unexpected error. "
                "Check your API keys and network, then try again."
            )
        )
        return list(history) + [fallback]
    out_messages = result.get("messages")
    if isinstance(out_messages, list) and out_messages:
        return list(out_messages)
    return list(history)


async def _run_turn_async(history: list[BaseMessage]) -> list[BaseMessage]:
    graph = get_compiled_graph()
    try:
        result = await graph.ainvoke({"messages": list(history)})
    except Exception:
        logger.exception("Async graph invocation failed")
        fallback = AIMessage(
            content=(
                "The request could not be completed due to an unexpected error. "
                "Check your API keys and network, then try again."
            )
        )
        return list(history) + [fallback]
    out_messages = result.get("messages")
    if isinstance(out_messages, list) and out_messages:
        return list(out_messages)
    return list(history)


def run_cli_loop() -> None:
    print(
        "Competitor intelligence assistant (LangGraph). "
        "Ask about clothing competitors, footfall, and peak hours for a neighborhood. "
        "Commands: exit, quit.\n"
    )
    history: list[BaseMessage] = []
    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in {"exit", "quit"}:
            break
        history.append(HumanMessage(content=line))
        history = _run_turn(history)
        _print_assistant_reply(history)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Clothing competitor intelligence (LangGraph).")
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single-turn question (non-interactive).",
    )
    parser.add_argument(
        "--async-run",
        action="store_true",
        help="Use async graph invocation for the single query.",
    )
    args = parser.parse_args(argv)

    load_dotenv()
    configure_logging()

    try:
        from competitor_intel.config import get_settings

        get_settings()
    except ConfigurationError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc

    if args.query is not None:
        history: list[BaseMessage] = [HumanMessage(content=args.query.strip())]
        if args.async_run:
            history = asyncio.run(_run_turn_async(history))
        else:
            history = _run_turn(history)
        _print_assistant_reply(history)
        return

    run_cli_loop()


if __name__ == "__main__":
    main()
