"""Application-specific errors for centralized handling."""


class BillManagementError(Exception):
    """Base error for the bill management agent."""


class ConfigurationError(BillManagementError):
    """Raised when environment configuration is invalid or missing."""


class BillProcessingError(BillManagementError):
    """Raised when bill image processing fails."""


class SummarizationError(BillManagementError):
    """Raised when expense summarization fails."""
