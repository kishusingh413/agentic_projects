"""
Personalized Education Assistant — entrypoint.

Requires OPENAI_API_KEY and SERPER_API_KEY (see .env.example).
"""

from __future__ import annotations

import argparse
import logging
import sys

from education_assistant.config import load_config
from education_assistant.crew import run_education_pipeline
from education_assistant.models import ExpertiseLevel


def _parse_expertise(value: str) -> ExpertiseLevel:
    normalized = value.strip().lower()
    try:
        return ExpertiseLevel(normalized)
    except ValueError as exc:
        allowed = ", ".join(e.value for e in ExpertiseLevel)
        raise argparse.ArgumentTypeError(
            f"expertise must be one of: {allowed}"
        ) from exc


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run the Personalized Education Assistant (CrewAI sequential crew).",
    )
    parser.add_argument(
        "--topics",
        type=str,
        required=True,
        help="Comma-separated topics, e.g. 'Python async, REST APIs'",
    )
    parser.add_argument(
        "--expertise",
        type=_parse_expertise,
        required=True,
        help="beginner | intermediate | advanced",
    )
    args = parser.parse_args()

    if not args.topics.strip():
        logging.error("topics must be non-empty")
        return 2

    try:
        load_config()
    except RuntimeError as exc:
        logging.error("%s", exc)
        return 1

    try:
        report = run_education_pipeline(args.topics, args.expertise)
    except RuntimeError as exc:
        logging.error("%s", exc)
        return 1

    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
