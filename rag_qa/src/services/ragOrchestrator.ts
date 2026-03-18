/**
 * RAG orchestrator: coordinates preprocessing, retrieval, and answer generation with source attribution.
 */

import { readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { loadConfig } from '../config/env.js';
import { DocumentError } from '../errors/index.js';
import type { DocumentChunk, RAGResponse, RetrievedChunk, SourceCitation } from '../types/index.js';
import { AnswerGenerationService } from './answerGenerationService.js';
import { DocumentPreprocessor } from './documentPreprocessor.js';
import { RetrievalService } from './retrievalService.js';

export class RAGOrchestrator {
  private readonly preprocessor: DocumentPreprocessor;
  private readonly retrievalService: RetrievalService;
  private readonly answerService: AnswerGenerationService;
  private readonly config: ReturnType<typeof loadConfig>;

  private chunks: DocumentChunk[] = [];

  constructor() {
    this.config = loadConfig();
    this.preprocessor = new DocumentPreprocessor();
    this.retrievalService = new RetrievalService();
    this.answerService = new AnswerGenerationService();
  }

  async ingest(): Promise<{ documentsProcessed: number; totalChunks: number; errors: string[] }> {
    const documentsDir = join(process.cwd(), this.config.RAG_DOCUMENTS_DIR);
    const { chunks, result } = await this.preprocessor.processDocuments(documentsDir);
    this.chunks = chunks;
    await this.persistVectorStore();
    return result;
  }

  private async persistVectorStore(): Promise<void> {
    const path = join(process.cwd(), this.config.RAG_VECTOR_STORE_PATH);
    const payload = JSON.stringify({
      chunks: this.chunks.map((c) => ({
        ...c,
        embedding: c.embedding,
      })),
    });
    await writeFile(path, payload, 'utf-8');
  }

  private async loadVectorStore(): Promise<void> {
    const path = join(process.cwd(), this.config.RAG_VECTOR_STORE_PATH);
    if (!existsSync(path)) {
      throw new DocumentError(
        'Vector store not found. Run ingest first to process documents.',
        { path }
      );
    }
    const data = await readFile(path, 'utf-8');
    const parsed = JSON.parse(data) as { chunks: DocumentChunk[] };
    const chunks = parsed?.chunks;
    if (!Array.isArray(chunks)) {
      throw new DocumentError('Invalid vector store format', { path });
    }
    this.chunks = chunks;
  }

  private buildSourceCitations(retrieved: RetrievedChunk[]): SourceCitation[] {
    return retrieved.map((rc) => ({
      fileName: rc.chunk.source.fileName,
      pageNumber: rc.chunk.source.pageNumber,
      excerpt: rc.chunk.content.slice(0, 200) + (rc.chunk.content.length > 200 ? '...' : ''),
      score: Math.round(rc.score * 1000) / 1000,
    }));
  }

  async ask(question: string): Promise<RAGResponse> {
    if (this.chunks.length === 0) {
      await this.loadVectorStore();
    }

    const retrieved = await this.retrievalService.retrieveRelevantChunks(
      question,
      this.chunks
    );
    const answer = await this.answerService.generateAnswer(question, retrieved);
    const sources = this.buildSourceCitations(retrieved);

    return { answer, sources };
  }

  getChunkCount(): number {
    return this.chunks.length;
  }
}
