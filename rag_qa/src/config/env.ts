/**
 * Validated environment configuration.
 * Never trust process.env blindly — parse and validate at startup.
 */

import { z } from 'zod';
import { ConfigError } from '../errors/index.js';

const envSchema = z.object({
  OPENAI_API_KEY: z.string().min(1, 'OPENAI_API_KEY is required'),
  OPENAI_CHAT_MODEL: z.string().default('gpt-4o-mini'),
  OPENAI_EMBEDDING_MODEL: z.string().default('text-embedding-3-small'),
  OPENAI_MAX_TOKENS: z.coerce.number().min(1).max(128000).default(2048),
  OPENAI_TEMPERATURE: z.coerce.number().min(0).max(2).default(0.3),
  RAG_TOP_K: z.coerce.number().min(1).max(20).default(5),
  RAG_CHUNK_SIZE: z.coerce.number().min(100).max(4000).default(1000),
  RAG_CHUNK_OVERLAP: z.coerce.number().min(0).max(1000).default(200),
  RAG_DOCUMENTS_DIR: z.string().default('./documents'),
  RAG_VECTOR_STORE_PATH: z.string().default('./vector-store.json'),
});

export type EnvConfig = z.infer<typeof envSchema>;

let cachedConfig: EnvConfig | null = null;

export function loadConfig(): EnvConfig {
  if (cachedConfig !== null) {
    return cachedConfig;
  }

  const result = envSchema.safeParse({
    OPENAI_API_KEY: process.env.OPENAI_API_KEY,
    OPENAI_CHAT_MODEL: process.env.OPENAI_CHAT_MODEL,
    OPENAI_EMBEDDING_MODEL: process.env.OPENAI_EMBEDDING_MODEL,
    OPENAI_MAX_TOKENS: process.env.OPENAI_MAX_TOKENS,
    OPENAI_TEMPERATURE: process.env.OPENAI_TEMPERATURE,
    RAG_TOP_K: process.env.RAG_TOP_K,
    RAG_CHUNK_SIZE: process.env.RAG_CHUNK_SIZE,
    RAG_CHUNK_OVERLAP: process.env.RAG_CHUNK_OVERLAP,
    RAG_DOCUMENTS_DIR: process.env.RAG_DOCUMENTS_DIR,
    RAG_VECTOR_STORE_PATH: process.env.RAG_VECTOR_STORE_PATH,
  });

  if (!result.success) {
    const firstError = result.error.errors[0];
    const msg = firstError
      ? `${firstError.path.join('.')}: ${firstError.message}`
      : 'Invalid environment configuration';
    throw new ConfigError(msg, { issues: result.error.errors });
  }

  cachedConfig = result.data;
  return cachedConfig;
}
