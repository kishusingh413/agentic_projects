/**
 * Centralized error handling for the RAG QA system.
 */

export class RAGError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly context?: Record<string, unknown>
  ) {
    super(message);
    this.name = 'RAGError';
    Object.setPrototypeOf(this, RAGError.prototype);
  }

  toLogString(): string {
    const contextStr = this.context
      ? ` | Context: ${JSON.stringify(this.context)}`
      : '';
    return `[${this.code}] ${this.message}${contextStr}`;
  }
}

export class ConfigError extends RAGError {
  constructor(message: string, context?: Record<string, unknown>) {
    super(message, 'CONFIG_ERROR', context);
  }
}

export class DocumentError extends RAGError {
  constructor(message: string, context?: Record<string, unknown>) {
    super(message, 'DOCUMENT_ERROR', context);
  }
}

export class EmbeddingError extends RAGError {
  constructor(message: string, context?: Record<string, unknown>) {
    super(message, 'EMBEDDING_ERROR', context);
  }
}

export class RetrievalError extends RAGError {
  constructor(message: string, context?: Record<string, unknown>) {
    super(message, 'RETRIEVAL_ERROR', context);
  }
}

export class GenerationError extends RAGError {
  constructor(message: string, context?: Record<string, unknown>) {
    super(message, 'GENERATION_ERROR', context);
  }
}
