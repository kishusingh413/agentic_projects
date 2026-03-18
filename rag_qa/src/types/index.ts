/**
 * Core types for the RAG QA system.
 */

export interface DocumentChunk {
  id: string;
  content: string;
  embedding: number[];
  source: DocumentSource;
  chunkIndex: number;
}

export interface DocumentSource {
  fileName: string;
  pageNumber?: number;
  startChar?: number;
  endChar?: number;
}

export interface RetrievedChunk {
  chunk: DocumentChunk;
  score: number;
}

export interface RAGResponse {
  answer: string;
  sources: SourceCitation[];
}

export interface SourceCitation {
  fileName: string;
  pageNumber?: number;
  excerpt: string;
  score: number;
}

export interface IngestResult {
  documentsProcessed: number;
  totalChunks: number;
  errors: string[];
}
