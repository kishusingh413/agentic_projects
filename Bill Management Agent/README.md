# Bill Management Agent

Python agent that reads a **receipt image** (photo or scan), categorizes line items, totals spending by category, and surfaces simple insights. Workflow: **User Proxy** -> **Bill Processing** -> **Expense Summarization**, coordinated by a **Group Chat Manager**.

## Setup

```bash
cd "Bill Management Agent"
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

Copy `.env.example` to `.env` and set **`OPENAI_API_KEY`**. You only need to pass a path to your receipt image; no JSON or extra files are required for normal use.

## Run

```bash
python -m bill_management_agent.main path\to\receipt.jpg
```

Optional: `python -m bill_management_agent.main --demo` uses the bundled sample image in `data/`.

## Layout

- `src/bill_management_agent/` — application package
- `data/` — sample assets for `--demo` / developer testing
