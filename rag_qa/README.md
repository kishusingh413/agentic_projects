# RAG QA – Retrieval-Augmented Generation for Question Answering

A TypeScript RAG system that answers questions from a collection of AI research papers (or any PDF documents).

## Project Goal

Enable users to ask questions and receive accurate, context-aware answers derived from the provided documents, with source attribution.

## Key Components

1. **Document Preprocessing** – Load PDFs, split into chunks, generate vector embeddings
2. **Retrieval System** – Retrieve relevant document segments based on user queries
3. **Answer Generation** – Use an LLM to synthesize retrieved information into answers
4. **Source Attribution** – Answers include references to the specific sections used

## Setup

1. **Install dependencies**

   ```bash
   cd rag_qa
   npm install
   ```

2. **Configure environment**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and set your `OPENAI_API_KEY`.

3. **Add documents**

   Place your PDF files (e.g. AI research papers) in the `documents/` folder.

## Usage

### 1. Ingest documents

Process PDFs and build the vector store:

```bash
npm run ingest
```

### 2. Ask questions

```bash
npm run ask -- "What are the main components of a RAG model, and how do they interact?"
```

### Sample questions (from the project brief)

- What are the main components of a RAG model, and how do they interact?
- What are the two sub-layers in each encoder layer of the Transformer model?
- Explain how positional encoding is implemented in Transformers and why it is necessary.
- Describe the concept of multi-head attention in the Transformer architecture. Why is it beneficial?
- What is few-shot learning, and how does GPT-3 implement it during inference?

## Configuration (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *required* | OpenAI API key |
| `OPENAI_CHAT_MODEL` | gpt-4o-mini | Model for answer generation |
| `OPENAI_EMBEDDING_MODEL` | text-embedding-3-small | Model for embeddings |
| `RAG_TOP_K` | 5 | Number of chunks to retrieve per query |
| `RAG_CHUNK_SIZE` | 1000 | Chunk size in characters |
| `RAG_CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `RAG_DOCUMENTS_DIR` | ./documents | Folder containing PDFs |
| `RAG_VECTOR_STORE_PATH` | ./vector-store.json | Path to persist embeddings |

## Project structure

```
rag_qa/
├── documents/          # Place your PDFs here
├── src/
│   ├── config/        # Env validation
│   ├── errors/        # Custom error classes
│   ├── services/      # Business logic
│   │   ├── documentPreprocessor.ts
│   │   ├── retrievalService.ts
│   │   ├── answerGenerationService.ts
│   │   └── ragOrchestrator.ts
│   ├── types/
│   ├── utils/
│   └── index.ts       # CLI entry point
├── .env.example
├── package.json
└── README.md
```

## Implementation notes

- **Clean architecture** – Business logic in services, not controllers
- **Validation** – Zod for env and inputs
- **Error handling** – Custom error classes and centralized logging
- **TypeScript strict** – Explicit types, no `any`
- **Safety** – Optional chaining, guards, no non-null assertions
