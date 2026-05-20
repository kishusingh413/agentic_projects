# RAG QA Project (Python)

This project implements a Retrieval-Augmented Generation (RAG) system for question answering over local research documents.

## Features aligned to the 4 required steps

1. **Document Processing**  
   Loads `.txt`, `.md`, and `.pdf` files, then chunks text into overlapping segments.
2. **Retrieval System**  
   Uses TF-IDF + cosine similarity to retrieve the most relevant chunks for a question.
3. **Answer Generation**  
   Supports:
   - `extractive` mode (no API key required), and
   - `openai` mode (`OPENAI_API_KEY` required).
4. **Source Attribution**  
   Every answer includes source citations (filename, page if PDF, retrieval score).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration (optional)

Create `.env` in the project root:

```env
RAG_LLM_MODE=extractive
RAG_MODEL_NAME=gpt-4o-mini
RAG_CHUNK_SIZE=900
RAG_CHUNK_OVERLAP=120
OPENAI_API_KEY=your_key_here
```

Notes:
- Use `RAG_LLM_MODE=openai` to enable LLM generation.
- If `RAG_LLM_MODE=openai`, `OPENAI_API_KEY` is mandatory and validated at startup.

## Run (entry point with context path)

Pass file or folder path via `--context-path`:

```bash
python src/main.py --context-path "C:\path\to\papers" --top-k 4
```

You can then ask questions interactively:

```text
Q> What are the two sub-layers in each Transformer encoder layer?
```

Type `exit` to quit.
