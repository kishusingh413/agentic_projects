from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping


logger = logging.getLogger("logistics_crew")


@dataclass(frozen=True)
class ErrorContext:
    action: str
    payload: Mapping[str, object] | None = None


class AppError(Exception):
    """Base class for predictable application errors."""

    def __init__(self, message: str, *, context: ErrorContext | None = None) -> None:
        super().__init__(message)
        self.context = context

    def log(self) -> None:
        if self.context is None:
            logger.error(str(self))
            return
        logger.error(
            "%s | action=%s payload=%s",
            str(self),
            self.context.action,
            dict(self.context.payload or {}),
        )


class ProductsInputError(AppError):
    """Raised when product input can't be loaded or validated."""


class ProductsFileFormatError(ProductsInputError):
    """Raised when product file format is unsupported or ambiguous."""


class ProductsFileValidationError(ProductsInputError):
    """Raised when product file contents fail schema validation."""


class MissingEnvironmentError(AppError):
    """Raised when required environment variables are missing or invalid."""

