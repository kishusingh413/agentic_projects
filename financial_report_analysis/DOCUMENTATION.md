# Financial Report Analysis – Technical Documentation

This document describes the complete flow, components, libraries, and design decisions for the AI-powered Financial Report Analysis project.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture & Flow](#architecture--flow)
3. [Component Details](#component-details)
4. [Libraries & Dependencies](#libraries--dependencies)
5. [Configuration](#configuration)
6. [Usage Examples](#usage-examples)
7. [Output Guidelines](#output-guidelines)
8. [Extending the Project](#extending-the-project)

---

## Project Overview

### Problem Statement

Financial professionals, investors, and decision-makers struggle to quickly extract relevant insights from lengthy financial documents (annual reports, balance sheets, earnings reports). These documents contain dense numerical data and jargon, making manual review time-consuming and prone to oversight. Existing AI tools often lack context-aware, actionable insights tailored to the user's role.

### Solution

An AI-powered system that:

- Accepts financial document images (from URLs or local paths)
- Uses a vision-capable LLM to analyze charts, tables, and text
- Produces structured, role-tailored summaries with key metrics, trends, and risks

---

## Architecture & Flow

### End-to-End Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────────┐
│  User Input     │────▶│  Image Loader   │────▶│  Document Analyzer  │
│  (prompt +      │     │  (PIL, URLs)    │     │  (OpenAI Vision)    │
│   image paths)  │     └─────────────────┘     └──────────┬─────────┘
└─────────────────┘                                         │
                                                             ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────────┐
│  Summary Output │◀────│  Role-Specific  │◀────│  Structured         │
│  (stdout/file)  │     │  Prompting      │     │  AI Response        │
└─────────────────┘     └─────────────────┘     └─────────────────────┘
```

### Step-by-Step Flow

1. **Input**
   - User provides:
     - One or more image paths (local) or URLs
     - Optional prompt (default: comprehensive summary)
     - Optional role (investor, analyst, auditor, general)

2. **Image Loading**
   - Each source is classified as URL or local path
   - PIL loads images; URLs are fetched via `requests`
   - Images are converted to RGB for consistency

3. **Analysis**
   - Images are Base64-encoded and sent to the OpenAI Vision API
   - System prompt defines structure (metrics, balance sheet, credit quality, etc.)
   - User prompt is augmented with role-specific focus
   - Model returns a formatted summary

4. **Output**
   - Summary printed to stdout or written to a file

---

## Component Details

### 1. Image Loader (`src/image_loader.py`)

**Purpose:** Load financial document images from URLs or local paths using PIL.

**Functions:**

| Function | Description |
|----------|-------------|
| `load_image(source)` | Load a single image from a URL or path |
| `load_images(sources)` | Load multiple images; raises on first failure |

**Logic:**

- **URL detection:** Source starts with `http://` or `https://`
- **URL loading:** `requests.get()` → `BytesIO` → `PIL.Image.open()` → `convert("RGB")`
- **Local loading:** `Path` resolution → format check → `PIL.Image.open()` → `convert("RGB")`

**Supported formats:** PNG, JPG, JPEG, WEBP, BMP, GIF, TIFF

---

### 2. Document Analyzer (`src/document_analyzer.py`)

**Purpose:** Send images and prompts to a vision-capable LLM and return the analysis.

**Functions:**

| Function | Description |
|----------|-------------|
| `analyze_financial_documents(images, user_prompt, role, model)` | Main analysis entry point |

**Logic:**

- Converts each PIL image to Base64
- Builds a content array: `[text, image1, image2, ...]`
- Calls `client.chat.completions.create()` with:
  - System prompt (financial analyst persona + output structure)
  - User message containing prompt + images
- Returns the assistant’s text response

**Error handling:** Propagates API errors as `RuntimeError` with context.

---

### 3. Prompts (`src/prompts.py`)

**Purpose:** Define system and user prompts for consistent, structured output.

**Components:**

| Component | Description |
|-----------|-------------|
| `SYSTEM_PROMPT` | Defines the AI as a financial analyst; lists output sections and guidelines |
| `get_user_prompt(prompt, role)` | Appends role-specific focus to the user prompt |

**Output sections (when data available):**

1. Key Financial Metrics  
2. Income and Expenses  
3. Balance Sheet Highlights  
4. Credit Quality  
5. Strategic and Operational Updates  
6. Market Conditions and Outlook  
7. Shareholder Information  

**Role-specific suffixes:**

- **Investor:** returns, growth, risks, shareholder value  
- **Analyst:** ratios, trends, peer comparison, forecasting  
- **Auditor:** consistency, material items, compliance  
- **General:** no extra focus  

---

### 4. Configuration (`config/settings.py`)

**Purpose:** Load and validate settings from environment variables at startup.

**Variables:**

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key |
| `OPENAI_VISION_MODEL` | No | `gpt-4o` | Vision model name |
| `OPENAI_MAX_TOKENS` | No | `4096` | Max tokens (512–16000) |
| `OPENAI_TEMPERATURE` | No | `0.2` | Temperature (0–2) |
| `DEBUG` | No | `false` | Debug flag |

**Validation:** API key required; model must be in allowlist; numeric values clamped.

---

### 5. Main Entry Point (`main.py`)

**Purpose:** CLI wrapper for running the analysis.

**Usage:**

```bash
python main.py --images img1.png img2.png --prompt "Summarize key metrics" --role investor
python main.py -i https://example.com/report.png -p "Analyze balance sheet" -o summary.txt
```

**Arguments:**

| Argument | Short | Required | Description |
|----------|-------|----------|-------------|
| `--images` | `-i` | Yes | Image paths or URLs |
| `--prompt` | `-p` | No | User prompt (has default) |
| `--role` | `-r` | No | `investor` \| `analyst` \| `auditor` \| `general` |
| `--output` | `-o` | No | Write output to file |

---

## Libraries & Dependencies

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| **openai** | ≥1.12.0 | OpenAI Chat Completions API (incl. vision) |
| **Pillow (PIL)** | ≥10.0.0 | Image loading, format conversion, Base64 encoding |
| **requests** | ≥2.28.0 | HTTP requests for loading images from URLs |

### Why These Libraries?

- **openai:** Official client for the Vision API; supports multimodal content and Base64 images.
- **Pillow:** Standard for image I/O in Python; supports many formats and conversions.
- **requests:** Simple, robust HTTP client for fetching remote images.

### Python Version

- Python 3.10+ (for type hints and modern syntax).

---

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and set:

```
OPENAI_API_KEY=sk-...
OPENAI_VISION_MODEL=gpt-4o
OPENAI_MAX_TOKENS=4096
OPENAI_TEMPERATURE=0.2
```

For local development, use `python-dotenv` if you want automatic loading from `.env`:

```bash
pip install python-dotenv
```

Then in your entry script:

```python
from dotenv import load_dotenv
load_dotenv()
```

---

## Usage Examples

### Basic Run

```bash
cd financial_report_analysis
python main.py --images ./chart.png ./balance_sheet.png
```

### Role-Based Analysis

```bash
python main.py -i report.png -p "Summarize for investment decision" -r investor
```

### With Custom Output File

```bash
python main.py -i report1.png report2.png -o analysis_summary.txt
```

### Using Sample Assets

```bash
python run_example.py
# Or with explicit paths:
python run_example.py --images ../assets/image1.png ../assets/image2.png
```

---

## Output Guidelines

The system prompt instructs the model to produce summaries that include (when present in the documents):

- **Key metrics:** revenue, net income, EPS, RoE, cost/income ratio, growth
- **Balance sheet:** assets, liabilities, equity, loans, deposits
- **Income/expenses:** NII, fees, operating expenses
- **Credit quality:** cost of risk, NPLs, forbearance
- **Strategic updates:** acquisitions, digital, sustainability
- **Market & outlook:** risks, regulatory, management commentary
- **Shareholder info:** dividends, stock performance

Output uses bullet points, explicit units (EUR million, %, bps), and comparisons to prior periods and targets.

---

## Extending the Project

### Adding New Roles

1. Update `UserRole` in `src/prompts.py`
2. Add the role entry to `role_context` in `get_user_prompt()`
3. Extend the `--role` choices in `main.py` and `run_example.py`

### Switching to Another Vision Provider

1. Implement a new analyzer module (e.g. `src/document_analyzer_gemini.py`)
2. Use the same interface: `analyze_financial_documents(images, prompt, role) -> str`
3. Update `main.py` to call the new module (or use a config flag)

### Adding PDF Support

1. Use `pdf2image` or `PyMuPDF` to convert PDF pages to images
2. Feed the resulting images into the existing pipeline

### Adding a Web API

1. Create a FastAPI/Flask app
2. Accept multipart file upload or image URLs
3. Use `load_images()` and `analyze_financial_documents()` in the route handler


python main.py --images "..\assets\WhatsApp Image 2026-03-14 at 2.38.54 PM.jpeg" "..\assets\WhatsApp Image 2026-03-14 at 2.39.28 PM.jpeg" -r investor -o summary.txt