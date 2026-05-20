"""Application-specific errors for the portfolio workflow."""


class PortfolioWorkflowError(Exception):
    """Raised when the multi-agent portfolio workflow cannot proceed safely."""

    def __init__(self, message: str, *, context: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.context: dict[str, str] = context or {}
