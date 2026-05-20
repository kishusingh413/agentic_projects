"""
Entry point: sequential User Proxy → BMI tool (via BMI Agent in LLM mode) →
Diet Planner → Workout Scheduler.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from smart_health_assistant.config import Settings
from smart_health_assistant.pipeline import run_health_assistant_sync


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s %(message)s",
    )


def main() -> None:
    _configure_logging()
    log = logging.getLogger("smart_health.main")
    try:
        settings = Settings()
    except Exception as exc:
        log.exception("invalid_settings", extra={"action": "startup"})
        raise SystemExit(f"Configuration error: {exc}") from exc

    plan = run_health_assistant_sync(settings)
    print(plan.as_report())
    if plan.mode == "offline":
        print(
            "\nTip: set OPENAI_API_KEY (and optionally OPENAI_MODEL) to run the full "
            "AutoGen assistant chain with an LLM.\n"
        )


if __name__ == "__main__":
    main()
