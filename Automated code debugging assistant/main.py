"""Entry point: automated Python debugging assistant (CrewAI)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from app.crew_pipeline import run_debugging_crew
from app.settings import load_settings

DEFAULT_SAMPLE = '''def fibonacci_iterative(n):

    if n < 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]

    fib_sequence = [0, 1]
    for i in range(2, n):
    next_fib = fib_sequence[-1] + fib_sequence[-2]
    fib_sequence.append(next_fib)
    return fib_sequence'''


def _read_code(path: Path | None) -> str:
    if path is None:
        return DEFAULT_SAMPLE
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"File is empty: {path}")
    return text


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Run CrewAI agents to analyze and correct Python code.",
    )
    parser.add_argument(
        "file",
        nargs="?",
        type=Path,
        help="Path to a .py file to debug (default: built-in Fibonacci example)",
    )
    args = parser.parse_args()

    try:
        settings = load_settings()
    except Exception as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        print(
            "Set OPENAI_API_KEY in the environment or a .env file. "
            "If Docker is unavailable, set CODE_INTERPRETER_UNSAFE=true for local trusted use only.",
            file=sys.stderr,
        )
        return 1

    try:
        code = _read_code(args.file)
    except (OSError, ValueError) as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 1

    try:
        result = run_debugging_crew(settings, code)
    except Exception as exc:
        print(f"Crew execution failed: {exc}", file=sys.stderr)
        return 1

    print("\n--- Crew raw output ---\n")
    print(result.raw)
    if getattr(result, "tasks_output", None):
        print("\n--- Per-task outputs ---\n")
        for i, task_out in enumerate(result.tasks_output, start=1):
            print(f"Task {i}: {task_out.raw}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
