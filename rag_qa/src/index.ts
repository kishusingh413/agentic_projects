#!/usr/bin/env node
/**
 * RAG QA CLI: ingest documents and answer questions.
 */

import 'dotenv/config';
import { loadConfig } from './config/env.js';
import { RAGError } from './errors/index.js';
import { logError, logInfo } from './utils/logger.js';
import { RAGOrchestrator } from './services/ragOrchestrator.js';

const getQuestionArg = (): string =>
  process.argv.slice(3).join(' ').trim();

async function main(): Promise<void> {
  try {
    loadConfig();
  } catch (err) {
    if (err instanceof RAGError) {
      logError(err.toLogString(), err.context);
    } else {
      logError(err instanceof Error ? err.message : String(err));
    }
    process.exit(1);
  }

  const command = process.argv[2]?.toLowerCase();
  const orchestrator = new RAGOrchestrator();

  if (command === 'ingest') {
    await runIngest(orchestrator);
    return;
  }

  if (command === 'ask') {
    const question = getQuestionArg();
    if (!question) {
      logError('Usage: npm run ask -- "Your question here"');
      process.exit(1);
    }
    await runAsk(orchestrator, question);
    return;
  }

  printUsage();
}

async function runIngest(orchestrator: RAGOrchestrator): Promise<void> {
  try {
    logInfo('Starting document ingestion...');
    const result = await orchestrator.ingest();
    logInfo('Ingestion complete', {
      documentsProcessed: result.documentsProcessed,
      totalChunks: result.totalChunks,
    });
    if (result.errors.length > 0) {
      result.errors.forEach((e) => logError(e));
    }
  } catch (err) {
    handleError(err);
  }
}

async function runAsk(orchestrator: RAGOrchestrator, question: string): Promise<void> {
  try {
    logInfo('Processing question...');
    const response = await orchestrator.ask(question);

    console.log('\n--- Answer ---\n');
    console.log(response.answer);
    console.log('\n--- Sources ---\n');
    response.sources.forEach((s, i) => {
      console.log(`${i + 1}. ${s.fileName}${s.pageNumber != null ? ` (p.${s.pageNumber})` : ''} [score: ${s.score}]`);
      console.log(`   "${s.excerpt}..."`);
    });
  } catch (err) {
    handleError(err);
  }
}

function handleError(err: unknown): void {
  if (err instanceof RAGError) {
    logError(err.toLogString(), err.context);
  } else {
    logError(err instanceof Error ? err.message : String(err));
  }
  process.exit(1);
}

function printUsage(): void {
  console.log(`
RAG QA - Question Answering from Documents

Usage:
  npm run ingest              Process PDFs in ./documents and build vector store
  npm run ask -- "question"   Ask a question based on ingested documents

Setup:
  1. Copy .env.example to .env and set OPENAI_API_KEY
  2. Add PDF files to ./documents
  3. Run: npm run ingest
  4. Run: npm run ask -- "Your question"
`);
}

main().catch((err) => {
  logError(err instanceof Error ? err.message : String(err));
  process.exit(1);
});
