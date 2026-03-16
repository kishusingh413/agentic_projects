# AI-Powered Financial Report Analysis

Extract and summarize key insights from financial document images (balance sheets, charts, dashboards) using a vision-capable AI model.

## Features

- **Multi-source input:** Load images from local paths or URLs
- **Vision analysis:** Interprets tables, charts, graphs, and text
- **Role-tailored output:** Summaries optimized for investors, analysts, or auditors
- **Structured summaries:** Key metrics, balance sheet highlights, trends, risks, and targets

## Quick Start

### 1. Prerequisites

- Python 3.10+
- OpenAI API key (with Vision API access)

### 2. Install

```bash
cd financial_report_analysis
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-your-key-here
```

For quick testing, you can also set the variable in your shell:

```bash
export OPENAI_API_KEY=sk-your-key-here   # Linux/macOS
set OPENAI_API_KEY=sk-your-key-here      # Windows CMD
$env:OPENAI_API_KEY="sk-your-key-here"   # Windows PowerShell
```

### 4. Run

**With your own images:**

```bash
python main.py --images path/to/chart.png path/to/balance_sheet.png
```

**With sample images (from `../assets`):**

```bash
python run_example.py
```

**Custom prompt and role:**

```bash
python main.py -i report.png -p "Summarize key metrics for investors" -r investor -o summary.txt
```

## Usage

| Option | Short | Description |
|--------|-------|-------------|
| `--images` | `-i` | Image paths or URLs (required) |
| `--prompt` | `-p` | Analysis prompt (default: comprehensive summary) |
| `--role` | `-r` | Target: `investor`, `analyst`, `auditor`, `general` |
| `--output` | `-o` | Write output to file |

## Output Example

The AI returns structured summaries with sections such as:

- **Key Financial Metrics** – Net profit, EPS, RoE, cost/income ratio, etc.
- **Income and Expenses** – NII, fees, operating expenses
- **Balance Sheet Highlights** – Total assets, loans, deposits, equity
- **Credit Quality** – Cost of risk, NPLs, forbearance
- **Strategic and Operational Updates** – Digital, sustainability, partnerships
- **Market Conditions and Outlook** – Risks, regulatory, management commentary

## Documentation

See [DOCUMENTATION.md](./DOCUMENTATION.md) for:

- Architecture and data flow
- Component and library details
- Configuration options
- Extension points

## License

MIT
