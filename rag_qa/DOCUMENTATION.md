# RAG QA – Technical Documentation

Technical documentation for the RAG-based Question Answering system.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture & Flow](#architecture--flow)
3. [Component Details](#component-details)
4. [Configuration](#configuration)
5. [Usage](#usage)

---

## Project Overview

### Goal

Enable users to ask questions and receive accurate, context-aware answers derived from provided documents (e.g. AI research papers), with source attribution.

### Deliverables

1. **Document Preprocessing** – Load PDFs, split into chunks, generate vector embeddings  
2. **Retrieval System** – Retrieve relevant document segments based on user queries  
3. **Answer Generation** – Use an LLM to synthesize retrieved information into answers  
4. **Source Attribution** – Answers include references to the specific sections used  

---

## Architecture & Flow

### End-to-End Flow

```
┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│  PDF Documents  │────▶│  Document           │────▶│  Vector Store       │
│  (documents/)   │     │  Preprocessor       │     │  (JSON file)        │
└─────────────────┘     │  (load, chunk, embed)│     └──────────┬──────────┘
                       └─────────────────────┘                │
                                                              │
┌─────────────────┐     ┌─────────────────────┐               │
│  User Question  │────▶│  Retrieval Service  │◀──────────────┘
└─────────────────┘     │  (embed query,      │
                        │   cosine similarity)│
                        └──────────┬─────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐     ┌─────────────────────┐
                        │  Answer Generation │────▶│  Response           │
                        │  Service (LLM)     │     │  (answer + sources) │
                        └─────────────────────┘     └─────────────────────┘
```

### Data Flow

1. **Ingest**: PDFs → text extraction → chunking → embeddings → persisted vector store  
2. **Ask**: Question → query embedding → similarity search → top-k chunks → LLM prompt → answer + citations  

---

## Component Details

### Document Preprocessor (`documentPreprocessor.ts`)

- Loads all PDFs from `documents/`
- Uses `pdf-parse` for text extraction
- Splits text with configurable chunk size and overlap
- Generates embeddings via OpenAI `text-embedding-3-small` (configurable)
- Produces `DocumentChunk[]` with id, content, embedding, source metadata

### Retrieval Service (`retrievalService.ts`)

- Embeds user query with same model as documents
- Uses cosine similarity for vector search
- Returns top-k most relevant chunks with scores

### Answer Generation Service (`answerGenerationService.ts`)

- Builds context from retrieved chunks with source labels
- Calls OpenAI Chat Completions (e.g. `gpt-4o-mini`)
- Instructs model to use only context and cite sources

### RAG Orchestrator (`ragOrchestrator.ts`)

- Coordinates ingest and ask flows
- Persists/loads vector store to/from JSON
- Builds source citations from retrieved chunks

---

## Configuration

All config is validated at startup with Zod. See `.env.example` for options.

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *required* | OpenAI API key |
| `OPENAI_CHAT_MODEL` | gpt-4o-mini | Model for answers |
| `OPENAI_EMBEDDING_MODEL` | text-embedding-3-small | Model for embeddings |
| `RAG_TOP_K` | 5 | Chunks to retrieve per query |
| `RAG_CHUNK_SIZE` | 1000 | Chunk size in characters |
| `RAG_CHUNK_OVERLAP` | 200 | Overlap between chunks |

---

## Usage

```bash
# 1. Setup
cp .env.example .env
# Edit .env and set OPENAI_API_KEY

# 2. Add PDFs to documents/

# 3. Ingest
npm run ingest

# 4. Ask
npm run ask -- "What are the main components of a RAG model?"
```

---

## Sample Questions (from Project Brief)

1. What are the main components of a RAG model, and how do they interact?
2. What are the two sub-layers in each encoder layer of the Transformer model?
3. Explain how positional encoding is implemented in Transformers and why it is necessary.
4. Describe the concept of multi-head attention in the Transformer architecture. Why is it beneficial?
5. What is few-shot learning, and how does GPT-3 implement it during inference?
