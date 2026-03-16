#!/usr/bin/env python3
"""
Example script to run financial report analysis with sample images.

Run from project root:
    python run_example.py

Or specify custom image paths:
    python run_example.py --images path/to/image1.png path/to/image2.png
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.document_analyzer import analyze_financial_documents
from src.image_loader import load_images


def main() -> int:
    parser = argparse.ArgumentParser(description="Run financial report analysis with sample images")
    parser.add_argument(
        "--images",
        "-i",
        nargs="+",
        default=None,
        help="Image paths (default: assets in workspace)",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default="Analyze these financial document images and provide a comprehensive summary following the standard output format. Include key financial metrics, balance sheet highlights, trends, and any targets or footnotes.",
        help="Analysis prompt",
    )
    parser.add_argument(
        "--role",
        "-r",
        type=str,
        choices=["investor", "analyst", "auditor", "general"],
        default="investor",
        help="Target audience",
    )
    args = parser.parse_args()

    # Default to workspace assets if not specified
    if args.images:
        sources = [Path(p).resolve() for p in args.images]
    else:
        assets_dir = _PROJECT_ROOT.parent / "assets"
        if not assets_dir.exists():
            print(
                f"No default assets folder at {assets_dir}. "
                "Please provide images with --images.",
                file=sys.stderr,
            )
            return 1
        # Use any PNG/JPG in assets
        sources = list(assets_dir.glob("*.png")) + list(assets_dir.glob("*.jpg"))
        sources = sorted(sources)[:5]  # Limit to 5
        if not sources:
            print(
                f"No images found in {assets_dir}. Use --images to specify paths.",
                file=sys.stderr,
            )
            return 1

    try:
        images = load_images([str(p) for p in sources])
    except (ValueError, FileNotFoundError) as e:
        print(f"Error loading images: {e}", file=sys.stderr)
        return 1

    print(f"Analyzing {len(images)} image(s)...", file=sys.stderr)
    print("-" * 60, file=sys.stderr)

    try:
        summary = analyze_financial_documents(
            images=images,
            user_prompt=args.prompt,
            role=args.role,
        )
        print(summary)
        return 0
    except Exception as e:
        print(f"Analysis failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
