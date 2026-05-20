"""Application-specific errors for centralized handling."""


class AppError(Exception):
    """Base error with a stable code for logging and UI."""

    code: str

    def __init__(self, message: str, code: str = "APP_ERROR") -> None:
        super().__init__(message)
        self.code = code


class ConfigurationError(AppError):
    """Invalid or missing configuration at startup."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="CONFIG_ERROR")


class SearchError(AppError):
    """Search provider failure."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="SEARCH_ERROR")


class ReportGenerationError(AppError):
    """LLM report synthesis failure."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="REPORT_ERROR")


class PlanningError(AppError):
    """LLM query planning failure."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="PLANNING_ERROR")
