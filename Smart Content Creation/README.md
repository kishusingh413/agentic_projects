# Smart Content Creation (AutoGen + Python)

This project simulates a two-agent reflection workflow on **Agentic AI**:

- **Content Creator Agent** drafts and revises content.
- **Content Critic Agent** evaluates language clarity and technical correctness.
- The loop runs for **3 to 5 turns**.
- A reflection summary returns the **final refined draft in markdown**.

## Files

- `smart_content_creation_autogen.py` - main script
- `requirements.txt` - Python dependency list

## Setup

1. Create and activate a local virtual environment (recommended):

   ```bash
   # PowerShell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies inside the virtual environment:

   ```bash
   pip install -r requirements.txt --upgrade --force-reinstall
   ```

3. Set environment variables:

   ```bash
   # PowerShell
   $env:OPENAI_API_KEY="your_api_key_here"
   $env:OPENAI_MODEL="gpt-4o-mini"
   ```

   `OPENAI_MODEL` is optional and defaults to `gpt-4o-mini`.

   You can also use a `.env` file in the project root:

   ```env
   OPENAI_API_KEY=your_api_key_here
   OPENAI_MODEL=gpt-4o-mini
   ```

## Run

```bash
python smart_content_creation_autogen.py
```

If you open a new terminal, reactivate the environment first:

```bash
.\.venv\Scripts\Activate.ps1
```

## Expected Output

The script prints:

- A title: `Final Refined Draft (Markdown)`
- The final polished content on **Agentic AI** in markdown format.
