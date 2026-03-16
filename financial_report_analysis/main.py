#!/usr/bin/env python3
"""
CLI entry point for AI-powered Financial Report Analysis.

Usage:
    python main.py --images img1.png img2.png --prompt "Summarize the key metrics"
    python main.py --images https://example.com/report.png --role investor
"""

from dotenv import load_dotenv

load_dotenv()

import argparse
import sys
from pathlib import Path

from src.document_analyzer import analyze_financial_documents
from src.image_loader import load_images
from src.prompts import UserRole


def _resolve_images(sources: list[str]) -> list[Path | str]:
    """Resolve image paths; support both local paths and URLs."""
    resolved: list[Path | str] = []
    for s in sources:
        s = s.strip()
        if not s:
            continue
        if s.lower().startswith(("http://", "https://")):
            resolved.append(s)
        else:
            resolved.append(Path(s).resolve())
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AI-powered Financial Report Analysis from document images",
    )
    parser.add_argument(
        "--images",
        "-i",
        nargs="+",
        required=True,
        help="Image paths or URLs (e.g. ./chart.png https://example.com/report.png)",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default="Analyze these financial document images and provide a comprehensive summary following the standard output format.",
        help="User prompt or analysis question",
    )
    parser.add_argument(
        "--role",
        "-r",
        type=str,
        choices=["investor", "analyst", "auditor", "general"],
        default="general",
        help="Target audience for insights",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Write output to file instead of stdout",
    )

    args = parser.parse_args()
    sources = _resolve_images(args.images)

    try:
        images = load_images(sources)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error loading images: {e}", file=sys.stderr)
        return 1

    try:
        summary = analyze_financial_documents(
            images=images,
            user_prompt=args.prompt,
            role=args.role,
        )
    except Exception as e:
        print(f"Analysis failed: {e}", file=sys.stderr)
        return 1

    if args.output:
        Path(args.output).write_text(summary, encoding="utf-8")
        print(f"Summary written to {args.output}", file=sys.stderr)
    else:
        print(summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
