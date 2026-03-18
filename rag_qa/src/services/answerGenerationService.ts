/**
 * Answer generation: synthesize retrieved context into well-formulated answers using LLM.
 */

import OpenAI from 'openai';
import { loadConfig } from '../config/env.js';
import { GenerationError } from '../errors/index.js';
import type { RetrievedChunk } from '../types/index.js';

const SYSTEM_PROMPT = `You are a helpful assistant that answers questions based ONLY on the provided context from research papers.

Rules:
1. Answer the question using ONLY the information in the provided context.
2. If the context doesn't contain relevant information, say so clearly.
3. Be precise and cite the source documents when making claims.
4. Use the format [Source: filename] when referencing specific documents.
5. Structure your answer clearly with paragraphs or bullet points when appropriate.
6. Do not make up information. Stay factual.`;

const USER_PROMPT_TEMPLATE = `Context from research papers:

{context}

---

Question: {question}

Provide a well-formulated answer based on the context above. Include source attribution for key claims.`;

function buildContext(retrievedChunks: RetrievedChunk[]): string {
  return retrievedChunks
    .map((rc) => {
      const source = rc.chunk.source.fileName;
      const page = rc.chunk.source.pageNumber;
      const header = page !== undefined
        ? `[Source: ${source}, p.${page}]`
        : `[Source: ${source}]`;
      return `${header}\n${rc.chunk.content}`;
    })
    .join('\n\n---\n\n');
}

export class AnswerGenerationService {
  private readonly openai: OpenAI;
  private readonly config: ReturnType<typeof loadConfig>;

  constructor() {
    this.config = loadConfig();
    this.openai = new OpenAI({
      apiKey: this.config.OPENAI_API_KEY,
    });
  }

  async generateAnswer(
    question: string,
    retrievedChunks: RetrievedChunk[]
  ): Promise<string> {
    if (retrievedChunks.length === 0) {
      return "I don't have enough relevant context from the documents to answer this question. Please ensure documents have been ingested and try a different phrasing.";
    }

    const context = buildContext(retrievedChunks);
    const userPrompt = USER_PROMPT_TEMPLATE
      .replace('{context}', context)
      .replace('{question}', question);

    try {
      const response = await this.openai.chat.completions.create({
        model: this.config.OPENAI_CHAT_MODEL,
        messages: [
          { role: 'system', content: SYSTEM_PROMPT },
          { role: 'user', content: userPrompt },
        ],
        max_tokens: this.config.OPENAI_MAX_TOKENS,
        temperature: this.config.OPENAI_TEMPERATURE,
      });

      const choice = response?.choices?.[0];
      const content = choice?.message?.content?.trim();

      if (!content) {
        throw new GenerationError('Empty response from LLM', {
          question: question.slice(0, 50),
        });
      }

      return content;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      throw new GenerationError(`Answer generation failed: ${message}`, {
        question: question.slice(0, 50),
      });
    }
  }
}
