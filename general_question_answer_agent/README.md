# General Question & Answer Agent (LangGraph)

A LangGraph agent that answers general questions with an LLM and routes arithmetic to four tools: `plus`, `subtract`, `multiply`, and `divide` (with safe handling for division by zero).

## Setup

1. Create a virtual environment (recommended) and install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and set API keys as needed.

- **`LLM_PROVIDER=auto` (default)** — uses the first backend that has credentials, in order: **OpenAI** → **Groq** → **Gemini** → **Ollama** (local). You can leave `LLM_PROVIDER` unset or set it to `auto` and only add the keys you use (for example only `OPENAI_API_KEY`).
- **Explicit provider** — set `LLM_PROVIDER` to `groq`, `gemini`, `ollama`, or `openai`. If that provider’s credentials are missing but another backend is configured, the app **falls back** to the same auto order and logs a warning.

Per-provider variables:

- **Groq**: `GROQ_API_KEY`, optional `GROQ_MODEL`
- **Gemini**: `GOOGLE_API_KEY`, optional `GEMINI_MODEL`
- **Ollama**: optional `OLLAMA_BASE_URL` and `OLLAMA_MODEL`. Start Ollama and use a model that supports tool calling (for example `ollama pull llama3.2`).
- **OpenAI**: `OPENAI_API_KEY`, optional `OPENAI_MODEL` and `OPENAI_BASE_URL` (custom API base only).

3. Run:

```bash
python main.py "What is 5 plus 3?"
python main.py "Explain what a black hole is in one sentence."
```

Or pipe a question:

```bash
echo "How much is 8 divided by 2?" | python main.py
```

## Project layout

- `app/math_tools.py` — tool definitions (`plus`, `subtract`, `multiply`, `divide`)
- `app/agent_graph.py` — LangGraph `StateGraph` with `agent` → `tools_condition` → `tools` / end
- `app/services/llm_factory.py` — builds the chat model from validated settings
- `app/config.py` — parses and validates environment variables at startup
- `main.py` — CLI entry

If your assignment referenced a PDF with extra steps, add that file to this folder or share the download URL so those instructions can be applied.
