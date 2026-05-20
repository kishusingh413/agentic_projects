from __future__ import annotations

from typing import Any, Optional


class AppError(Exception):
    def __init__(
        self,
        message: str,
        *,
        kind: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.kind: str = kind
        self.details: dict[str, Any] = details or {}


class ConfigurationError(AppError):
    def __init__(self, message: str, *, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message, kind="configuration_error", details=details)


class ExternalAPIError(AppError):
    def __init__(
        self, message: str, *, details: Optional[dict[str, Any]] = None
    ) -> None:
        super().__init__(message, kind="external_api_error", details=details)


class RoutingError(AppError):
    def __init__(
        self, message: str, *, details: Optional[dict[str, Any]] = None
    ) -> None:
        super().__init__(message, kind="routing_error", details=details)


class KnowledgeBaseError(AppError):
    def __init__(
        self, message: str, *, details: Optional[dict[str, Any]] = None
    ) -> None:
        super().__init__(message, kind="knowledge_base_error", details=details)

