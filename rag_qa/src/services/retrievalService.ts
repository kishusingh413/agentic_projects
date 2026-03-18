/**
 * Retrieval system: fetch relevant document chunks based on user queries.
 */

import OpenAI from 'openai';
import { loadConfig } from '../config/env.js';
import { EmbeddingError, RetrievalError } from '../errors/index.js';
import type { DocumentChunk, RetrievedChunk } from '../types/index.js';
import { cosineSimilarity } from '../utils/vectorUtils.js';

export class RetrievalService {
  private readonly openai: OpenAI;
  private readonly config: ReturnType<typeof loadConfig>;

  constructor() {
    this.config = loadConfig();
    this.openai = new OpenAI({
      apiKey: this.config.OPENAI_API_KEY,
    });
  }

  async embedQuery(query: string): Promise<number[]> {
    try {
      const response = await this.openai.embeddings.create({
        model: this.config.OPENAI_EMBEDDING_MODEL,
        input: query,
      });

      const data = response?.data?.[0];
      if (!data?.embedding || !Array.isArray(data.embedding)) {
        throw new EmbeddingError('Invalid query embedding response', {
          query: query.slice(0, 50),
        });
      }

      return data.embedding;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      throw new EmbeddingError(`Query embedding failed: ${message}`, { query });
    }
  }

  retrieve(
    queryEmbedding: number[],
    chunks: DocumentChunk[],
    topK?: number
  ): RetrievedChunk[] {
    const k = topK ?? this.config.RAG_TOP_K;

    if (chunks.length === 0) {
      throw new RetrievalError('No document chunks available for retrieval', {
        topK: k,
      });
    }

    const scored = chunks
      .map((chunk) => ({
        chunk,
        score: cosineSimilarity(queryEmbedding, chunk.embedding),
      }))
      .filter((item): item is RetrievedChunk => item.score > 0)
      .sort((a, b) => b.score - a.score);

    return scored.slice(0, k);
  }

  async retrieveRelevantChunks(
    query: string,
    chunks: DocumentChunk[],
    topK?: number
  ): Promise<RetrievedChunk[]> {
    const queryEmbedding = await this.embedQuery(query);
    return this.retrieve(queryEmbedding, chunks, topK);
  }
}
