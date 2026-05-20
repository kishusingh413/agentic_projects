"""CLI entry: Bill Management Agent with group-style collaboration."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from bill_management_agent.config import get_settings
from bill_management_agent.errors import ConfigurationError
from bill_management_agent.group_chat import run_sequential_agent_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def _ensure_sample_image(project_root: Path) -> Path:
    """Create a minimal PNG next to sample_bill.json if missing (for demos)."""
    png = project_root / "data" / "sample_bill.png"
    if png.is_file():
        return png
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required to create the sample image.") from exc

    png.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (520, 280), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    lines = [
        "SAMPLE MART",
        "Milk 12.50",
        "Snacks 5.92",
        "Total 18.42",
    ]
    y = 20
    for line in lines:
        draw.text((20, y), line, fill=(0, 0, 0))
        y += 36
    img.save(png, format="PNG")
    logger.info("Created sample bill image at %s", png)
    return png


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bill Management Agent - process bill images with grouped agent roles.",
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Path to your receipt image (png, jpg, jpeg, webp). Required unless --demo.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use the bundled sample receipt image in data/ (for quick tests).",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "sequential", "groupchat"),
        default="auto",
        help="All modes run the same native group chat (User Proxy -> Bill Processing -> "
        "Expense Summarization). 'groupchat' is an explicit alias for the collaborative run.",
    )
    args = parser.parse_args(argv)

    try:
        get_settings()
    except ConfigurationError as exc:
        logger.error("%s", exc)
        return 1

    project_root = Path(__file__).resolve().parents[2]
    if args.demo:
        _ensure_sample_image(project_root)
        path_obj = (project_root / "data" / "sample_bill.png").resolve()
    elif args.image:
        path_obj = Path(args.image).resolve()
    else:
        parser.error("Provide the path to your receipt image, or use --demo for the sample.")
    if not path_obj.is_file():
        logger.error("Receipt image not found: %s", path_obj)
        return 1

    logger.info("run mode=%s image=%s", args.mode, path_obj)

    try:
        print(
            run_sequential_agent_pipeline(
                str(path_obj),
                allow_packaged_demo=bool(args.demo),
            )
        )
    except Exception as exc:
        logger.exception("Run failed")
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
