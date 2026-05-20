from __future__ import annotations

import sys
from pathlib import Path
import json


def _serialize(obj: object) -> object:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(project_root))

    # Ensure configuration is validated at startup.
    from research_agent.app import build_app
    from research_agent.config import get_settings

    _ = get_settings()

    app = build_app()
    thread_id = "example-thread"

    queries: list[str] = [
        "Explain LangGraph and how it supports conditional routing.",
        "What is Retrieval-Augmented Generation (RAG)?",
        "What are the latest updates on LangGraph?",
        "What is 2+2?",
    ]

    for q in queries:
        result = app.invoke(
            {"query": q, "thread_id": thread_id},
            config={"configurable": {"thread_id": thread_id}},
        )
        final = result["final"]
        print("\n=== Query ===")
        print(q)
        print("\n=== Result ===")
        print(json.dumps(_serialize(final), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

