# LangGraph Multi-Agent Research & Summarization System

## Goal
Build a multi-agent system (using **LangGraph** in **Python**) that answers user questions by dynamically selecting one of three knowledge sources:
1. **LLM reasoning** (general questions that do not need web or RAG)
2. **Web research** (latest/current information)
3. **RAG** (retrieval from a local knowledge base + vector DB)

After collecting evidence, a **Summarization Agent** synthesizes a structured final response.

## Architecture (Agents + Services)

### Router Agent
**Responsibility:** Decide which path to take for each user query: `web`, `rag`, or `llm`.

- When `OPENAI_API_KEY` is available, the router uses an LLM classifier with structured output (`RouterDecision`).
- When the key is missing (offline fallback), the router uses keyword heuristics (so the system still runs).

### Web Research Agent
**Responsibility:** Fetch live web information.

- Uses `ddgs` (preferred) or falls back to `duckduckgo_search`.
- Returns a list of `WebFinding` objects (title/url/snippet).
- Best-effort page fetching is attempted to extract additional text (optional).

### RAG Agent
**Responsibility:** Retrieve relevant passages from a local knowledge base.

- Uses **Chroma** as a persistent vector database.
- Indexes local documents from `data/knowledge_base/*.txt` using OpenAI embeddings.
- When `OPENAI_API_KEY` is missing, an **offline keyword retrieval fallback** is used so RAG queries still return grounded evidence from the files.

### Summarization Agent
**Responsibility:** Combine the gathered evidence into a final markdown response.

- With an OpenAI key, it uses an LLM with structured output (`FinalResponse`).
- Without the key, it produces a deterministic offline markdown summary including available sources.

## LangGraph Workflow (State Graph)

The system is implemented as a LangGraph `StateGraph` with these nodes:

`START`
`ingest_user_message`  -> adds the user query to `messages` for conversational memory
`router`                -> sets `route` (`llm` / `web` / `rag`)
conditional edge from `router`:
- `web` -> `web_research`
- `rag` -> `rag_retrieve`
- `llm` -> `llm_reasoning`
then all paths go to:
`summarize` -> `END`

### Memory / Conversational Context
The graph includes LangGraph `messages` state and uses a checkpointing mechanism (`MemorySaver`) so repeated calls using the same `thread_id` keep conversation context inside the running process.

## Decision Rules (Routing)
When the OpenAI key is present: LLM-based classification.
When offline: keyword-based routing:
- `web` if query includes terms like `latest`, `today`, `current`, `news`, `price`, `weather`, etc.
- `rag` if query includes terms like `langgraph`, `rag`, `vector`, `knowledge base`, `chroma`, `openai api`
- `llm` otherwise

## How to Run

1. Install dependencies:
   - `pip install -r requirements.txt`
2. (Optional) set `OPENAI_API_KEY` in your environment or a `.env` file.
3. Run the CLI:
   - `python run.py --question "Explain LangGraph"`
4. Run example queries:
   - `python tests/example_queries.py`

## Example Test Queries (Offline Results)

These results were produced with `OPENAI_API_KEY` **not set** (offline fallback mode).

### Query 1
Command:
`python run.py --thread-id test1 --question "Explain LangGraph"`

Result (structured JSON):
```json
{
  "answer_markdown": "# Answer (offline fallback)\n\n## Query\nExplain LangGraph\n\n## Routed via\nrag\n\n## RAG evidence\n- **langgraph_overview** (data\\knowledge_base\\langgraph_overview.txt)\n  - LangGraph overview\n\nLangGraph is a framework ...\n",
  "sources": [
    "data\\knowledge_base\\langgraph_overview.txt"
  ],
  "debug": {
    "mode": "offline_fallback"
  }
}
```

### Query 2
Command:
`python run.py --thread-id test2 --question "What is 2+2?"`

Result (structured JSON):
```json
{
  "answer_markdown": "# Answer (offline fallback)\n\n## Query\nWhat is 2+2?\n\n## Routed via\nllm\n\n## Draft\nDraft answer for: `What is 2+2?` ...\n",
  "sources": [],
  "debug": {
    "mode": "offline_fallback"
  }
}
```

### Query 3
Command:
`python run.py --thread-id test3 --question "What are the latest updates on LangGraph?"`

Result (structured JSON):
```json
{
  "answer_markdown": "# Answer (offline fallback)\n\n## Query\nWhat are the latest updates on LangGraph?\n\n## Routed via\nweb\n\n## Summary\nThis is a deterministic offline summary because `OPENAI_API_KEY` is not set. ...",
  "sources": [],
  "debug": {
    "mode": "offline_fallback"
  }
}
```

Note: if the environment cannot reach the internet, `web_research` may return no findings. The router + workflow still behave correctly.

