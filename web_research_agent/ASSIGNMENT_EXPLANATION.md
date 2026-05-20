# ReAct Web Research Agent - Explanation

## What this project does
This project implements a ReAct-style web research agent in Python.  
Given a topic, the agent:
1. Uses an LLM to **reason and plan** (generate 5-6 guiding research questions).
2. Uses Tavily to **act** (search the web for each question).
3. Produces a structured markdown report with sources and synthesized key points.

## How LLM reasoning works (Planning phase)
The reasoning step is implemented in `generate_research_questions()`:
- The agent sends a prompt to the selected LLM provider (`openai`, `groq`, or `ollama`).
- The prompt asks for 5-6 non-overlapping questions that cover:
  - background/definition
  - trends/data
  - challenges
  - solutions/policies
  - future outlook
- The LLM must return strict JSON:
  ```json
  {
    "questions": ["...", "..."]
  }
  ```
- The agent validates and parses this JSON before moving to search.

This step is the "Reasoning" part of ReAct: the agent decides *what* to investigate before using external tools.

## How tool use works (Acting phase)
The acting step is implemented in `search_web()`:
- For each planned question, the agent queries Tavily.
- It extracts the `title`, `url`, and `content` of each result.
- Results are validated and stored as typed `SearchResult` objects.

This step is the "Acting" part of ReAct: the agent interacts with an external tool (web search) to gather evidence.

## Report generation
The report is built in `compile_report()` and includes:
- Title
- Introduction
- A section per research question
- Source-backed key points for each section
- Conclusion

Output is written to a markdown file (default: `report.md`).

## Program flow
1. Parse CLI arguments (`--topic`, `--output`).
2. Validate environment configuration (`load_and_validate_config()`).
3. Initialize `ReActResearchAgent`.
4. Planning:
   - Generate research questions with LLM.
5. Acting:
   - Search Tavily for each question.
6. Synthesis:
   - Compile all findings into a markdown report.
7. Save final report to disk.

## Run instructions
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set environment variables (copy from `.env.example`).
3. Run:
   ```bash
   python react_research_agent.py --topic "Climate Change" --output report.md
   ```

## Notes about robustness
- Environment variables are validated at startup.
- Provider values are restricted using an enum.
- LLM output is parsed and validated as JSON.
- Search result fields are type-checked before use.
- Errors include context (provider/model/query) to help debugging.
