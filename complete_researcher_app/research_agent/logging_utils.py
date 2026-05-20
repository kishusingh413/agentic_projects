from __future__ import annotations

import logging
import sys
import traceback
from typing import Any, Optional

from research_agent.errors import AppError


def configure_logging(level: int = logging.INFO) -> None:
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(level)


def log_exception(
    action: str,
    exc: BaseException,
    *,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    logger = logging.getLogger("research_agent")
    payload: dict[str, Any] = extra or {}

    if isinstance(exc, AppError):
        payload["kind"] = exc.kind
        payload["details"] = exc.details

    payload["action"] = action
    payload["exception_type"] = type(exc).__name__
    payload["exception_message"] = str(exc)

    logger.error(
        "Error occurred: %s",
        payload,
        exc_info=exc,
    )


def log_info(action: str, message: str, *, extra: Optional[dict[str, Any]] = None) -> None:
    logger = logging.getLogger("research_agent")
    payload: dict[str, Any] = {"action": action}
    if extra:
        payload.update(extra)
    logger.info("%s | %s", message, payload)


def format_traceback(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

