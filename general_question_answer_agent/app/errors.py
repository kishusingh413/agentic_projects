"""Application-specific errors."""


class ConfigurationError(Exception):
    """Raised when environment or settings are invalid."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
