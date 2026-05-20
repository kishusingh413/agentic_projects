# Intelligent Travel Assistant — Report

## 1. How the LLM is used for reasoning

This assistant uses an **OpenAI chat model** (default `gpt-4o-mini`, configurable via `OPENAI_MODEL`) as the **reasoning and orchestration** layer. The model does not guess weather or attractions from memory for factual data; instead it **decides when and how to call tools**, interprets raw tool output, and **writes the final natural-language answer** for the user.

Concretely, the LLM:

1. Reads the user’s goal (e.g. “plan a trip to Lisbon”).
2. Chooses tool calls such as `get_destination_weather` and `search_top_tourist_attractions` with appropriate arguments (the place name).
3. Receives **observations** (strings) from those tools and may reason over them in further internal steps (the “agent scratchpad”).
4. Produces a structured reply with **Weather** and **Things to see** sections, staying faithful to tool output.

That separation—**tools for facts**, **LLM for planning and presentation**—is the core pattern of a **tool-calling agent**.

## 2. How the reasoning step works (tool-calling loop)

The program uses LangChain’s **`create_tool_calling_agent`** together with **`AgentExecutor`**:

1. The **prompt** defines the system behavior and includes a **`MessagesPlaceholder("agent_scratchpad")`** so intermediate tool calls and results can be appended to the conversation state LangChain manages.
2. The **LLM** is invoked with **tool definitions** bound to it (native tool/function calling). The model may return **tool calls** (function name + arguments) instead of a final answer.
3. **`AgentExecutor`** runs a loop: if the model requested tools, each tool runs (HTTP weather request, web search for attractions via DuckDuckGo or Tavily), results are fed back, and the model is called again until it returns a normal assistant message.
4. **`verbose=True`** prints the intermediate steps in the terminal, which helps demonstrate the assignment’s “reasoning trace” during demos.

So the “reasoning step” is not a separate algorithm in our code; it is **the model’s repeated generation** over an expanding message list that includes tool results—standard **ReAct-style** agent execution as implemented by `AgentExecutor`.

## 3. Code and program flow

### Configuration (`config.py`)

Environment variables are validated at startup with **Pydantic Settings**: `OPENAI_API_KEY`, `WEATHER_API_KEY`, optional `OPENAI_MODEL`, optional `ATTRACTION_SEARCH_PROVIDER` (`duckduckgo` or `tavily`), and `TAVILY_API_KEY` when Tavily is selected. Empty required keys are rejected so the app fails fast with a clear error.

### Services (business logic, no routing)

- **`services/weather.py`** — Calls **WeatherAPI.com** `current.json` with `requests`, checks HTTP status, parses JSON, handles API error objects, and returns a short text summary (location resolution, °C, condition, humidity, wind). Errors are logged with context (`destination`, action name).
- **`services/attractions_search.py`** — Searches for “top tourist attractions …” using either **`DuckDuckGoSearchRun`** (default, no extra API key) or **`TavilySearchAPIWrapper`** when `ATTRACTION_SEARCH_PROVIDER=tavily` and `TAVILY_API_KEY` is set (validated in `config.py`). Tavily responses are formatted as a short summary plus titled snippets and URLs. Failures are caught, logged, and returned as user-visible strings so the agent can still respond.

### Tools (`travel_assistant.py`)

Two LangChain tools are defined with **`@tool`**:

- `get_destination_weather` → `fetch_current_weather(...)`
- `search_top_tourist_attractions` → `search_top_attractions(...)`

Docstrings describe when the model should use each tool; the LLM uses these descriptions for tool selection.

### Agent assembly (`travel_assistant.py`)

1. **`ChatOpenAI`** is constructed with the validated API key and model name.
2. **`create_tool_calling_agent(llm, tools, prompt)`** returns an agent **runnable**.
3. **`AgentExecutor(agent=..., tools=..., verbose=True, handle_parsing_errors=True)`** wraps it for **`invoke({"input": ...})`**.
4. **`run_interactive_cli`** loops on stdin, builds a clear user message naming the destination, prints the `output` field of the invoke result.

### Optional notebook

**`travel_assistant.ipynb`** demonstrates the same pipeline in a few cells (imports, settings, one sample `invoke`) for coursework submission alongside the script.

## 4. Setup summary

1. Copy `.env.example` to `.env` and set keys (OpenAI + [WeatherAPI.com](https://www.weatherapi.com/)).
2. Optionally set `ATTRACTION_SEARCH_PROVIDER=tavily` and add a [Tavily](https://tavily.com) API key as `TAVILY_API_KEY`.
3. `pip install -r requirements.txt`
4. `python travel_assistant.py`

### Tavily vs DuckDuckGo

The assignment allows **DuckDuckGo or Tavily**. The project supports **both**: default is **DuckDuckGo** (no search API key). Set **`ATTRACTION_SEARCH_PROVIDER=tavily`** and **`TAVILY_API_KEY`** to use **Tavily** via `TavilySearchAPIWrapper` inside `search_top_attractions`. Startup validation ensures the Tavily key is present when that provider is selected. The agent wiring (`create_tool_calling_agent` + `AgentExecutor`) is unchanged either way.
