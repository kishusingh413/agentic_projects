/**
 * Text splitting utilities for chunking documents.
 */

import type { DocumentSource } from '../types/index.js';

export interface TextChunk {
  content: string;
  source: DocumentSource;
  chunkIndex: number;
}

export function splitTextIntoChunks(
  text: string,
  chunkSize: number,
  overlap: number,
  source: DocumentSource
): TextChunk[] {
  if (chunkSize <= 0) {
    return [];
  }

  const chunks: TextChunk[] = [];
  let start = 0;
  let chunkIndex = 0;

  while (start < text.length) {
    const end = Math.min(start + chunkSize, text.length);
    let chunkEnd = end;

    if (end < text.length) {
      const lastSpace = text.lastIndexOf(' ', end);
      if (lastSpace > start) {
        chunkEnd = lastSpace;
      }
    }

    const content = text.slice(start, chunkEnd).trim();
    if (content.length > 0) {
      chunks.push({
        content,
        source: {
          ...source,
          startChar: start,
          endChar: chunkEnd,
        },
        chunkIndex,
      });
      chunkIndex += 1;
    }

    start = chunkEnd - (chunkEnd - start > overlap ? overlap : 0);
    if (start >= text.length) break;
  }

  return chunks;
}
