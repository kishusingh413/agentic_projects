class CompetitorIntelError(Exception):
    """Base error for the competitor intelligence pipeline."""


class ConfigurationError(CompetitorIntelError):
    """Raised when required configuration is missing or invalid."""


class SearchError(CompetitorIntelError):
    """Raised when the search backend fails."""
