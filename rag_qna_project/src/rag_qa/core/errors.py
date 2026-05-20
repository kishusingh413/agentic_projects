class RAGBaseError(Exception):
    """Base class for RAG-related errors."""


class ConfigurationError(RAGBaseError):
    """Raised when environment/configuration is invalid."""


class InputValidationError(RAGBaseError):
    """Raised when external input validation fails."""


class DocumentProcessingError(RAGBaseError):
    """Raised when document parsing/chunking fails."""


class RetrievalError(RAGBaseError):
    """Raised when retrieval cannot be completed."""


class GenerationError(RAGBaseError):
    """Raised when answer generation fails."""
