from __future__ import annotations

import argparse
import json
import sys

from research_agent.app import build_app
from research_agent.config import get_settings
from research_agent.errors import AppError
from research_agent.logging_utils import configure_logging, log_exception


def _print_json(obj: object) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LangGraph multi-agent research and summarization system"
    )
    parser.add_argument("--thread-id", default="default", help="Conversation thread id")
    parser.add_argument(
        "--question",
        default=None,
        help="Ask a single question and exit (otherwise interactive REPL).",
    )
    args = parser.parse_args()

    try:
        configure_logging()
        _ = get_settings()
        app = build_app()

        if args.question:
            result = app.invoke(
                {"query": args.question, "thread_id": args.thread_id},
                config={"configurable": {"thread_id": args.thread_id}},
            )
            final = result["final"]
            _print_json(final.model_dump() if hasattr(final, "model_dump") else final)
            return 0

        print("Interactive mode. Type 'exit' to quit.\n")
        while True:
            q = input("You> ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                return 0
            result = app.invoke(
                {"query": q, "thread_id": args.thread_id},
                config={"configurable": {"thread_id": args.thread_id}},
            )
            final = result["final"]
            _print_json(final.model_dump() if hasattr(final, "model_dump") else final)
            print()

    except AppError as e:
        log_exception("app_error", e, extra={"kind": e.kind})
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 - top-level safety net
        log_exception("unhandled_exception", e)
        print("ERROR: Unhandled exception. See logs for details.", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
