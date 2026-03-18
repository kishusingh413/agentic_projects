/**
 * Document preprocessing: load PDFs, split into chunks, generate embeddings.
 */

import { createRequire } from 'node:module';
import { readdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';

const require = createRequire(import.meta.url);
const pdfParse = require('pdf-parse') as (buffer: Buffer) => Promise<{ text: string }>;
import OpenAI from 'openai';
import { loadConfig } from '../config/env.js';
import { DocumentError, EmbeddingError } from '../errors/index.js';
import type { DocumentChunk, DocumentSource, IngestResult } from '../types/index.js';
import { randomUUID } from 'node:crypto';
import { splitTextIntoChunks } from '../utils/textSplitter.js';

function generateId(): string {
  return `chunk_${randomUUID()}`;
}

export class DocumentPreprocessor {
  private readonly openai: OpenAI;
  private readonly config: ReturnType<typeof loadConfig>;

  constructor() {
    this.config = loadConfig();
    this.openai = new OpenAI({
      apiKey: this.config.OPENAI_API_KEY,
    });
  }

  async loadDocuments(documentsDir: string): Promise<Map<string, string>> {
    const documents = new Map<string, string>();

    try {
      const entries = await readdir(documentsDir, { withFileTypes: true });
      const pdfFiles = entries.filter(
        (e) => e.isFile() && e.name.toLowerCase().endsWith('.pdf')
      );

      if (pdfFiles.length === 0) {
        throw new DocumentError(
          `No PDF files found in ${documentsDir}. Add PDF files to the documents folder.`,
          { path: documentsDir }
        );
      }

      for (const file of pdfFiles) {
        const filePath = join(documentsDir, file.name);
        try {
          const buffer = await readFile(filePath);
          const data = await pdfParse(buffer);
          const text = (data?.text ?? '').trim();
          if (text.length > 0) {
            documents.set(file.name, text.trim());
          }
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err);
          throw new DocumentError(
            `Failed to parse PDF: ${file.name}`,
            { error: message, fileName: file.name }
          );
        }
      }
    } catch (err) {
      if (err instanceof DocumentError) {
        throw err;
      }
      const message = err instanceof Error ? err.message : String(err);
      throw new DocumentError(
        `Failed to read documents directory: ${documentsDir}`,
        { error: message }
      );
    }

    return documents;
  }

  async createEmbeddings(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) {
      return [];
    }

    try {
      const response = await this.openai.embeddings.create({
        model: this.config.OPENAI_EMBEDDING_MODEL,
        input: texts,
      });

      const data = response?.data;
      if (!data || !Array.isArray(data)) {
        throw new EmbeddingError('Invalid response from embeddings API', {
          response: JSON.stringify(response),
        });
      }

      const embeddings = data
        .sort((a, b) => (a.index ?? 0) - (b.index ?? 0))
        .map((item) => item.embedding)
        .filter((e): e is number[] => Array.isArray(e));

      return embeddings;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      throw new EmbeddingError(`Embedding generation failed: ${message}`, {
        textCount: texts.length,
      });
    }
  }

  async processDocuments(documentsDir: string): Promise<{
    chunks: DocumentChunk[];
    result: IngestResult;
  }> {
    const documents = await this.loadDocuments(documentsDir);
    const errors: string[] = [];
    const allChunks: DocumentChunk[] = [];

    for (const [fileName, text] of documents) {
      const source: DocumentSource = { fileName };
      const textChunks = splitTextIntoChunks(
        text,
        this.config.RAG_CHUNK_SIZE,
        this.config.RAG_CHUNK_OVERLAP,
        source
      );

      if (textChunks.length === 0) {
        continue;
      }

      const contents = textChunks.map((c) => c.content);
      const embeddings = await this.createEmbeddings(contents);

      for (let i = 0; i < textChunks.length; i++) {
        const tc = textChunks[i];
        const embedding = embeddings[i];
        if (!embedding) {
          errors.push(`Missing embedding for chunk ${i} in ${fileName}`);
          continue;
        }
        allChunks.push({
          id: generateId(),
          content: tc.content,
          embedding,
          source: tc.source,
          chunkIndex: tc.chunkIndex,
        });
      }
    }

    return {
      chunks: allChunks,
      result: {
        documentsProcessed: documents.size,
        totalChunks: allChunks.length,
        errors,
      },
    };
  }
}
