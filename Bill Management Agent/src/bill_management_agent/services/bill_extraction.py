"""Extract and categorize expenses from bill images."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

from openai import OpenAI

from bill_management_agent.config import get_settings
from bill_management_agent.errors import BillProcessingError
from bill_management_agent.models import CategorizedExpenses, ExpenseCategory, LineItem

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM = """You are a bill processing assistant. Given a receipt or bill image, extract line items.
For each line: description, amount (number), and category.
Categories must be exactly one of: groceries, dining, utilities, shopping, entertainment, other.
Respond with ONLY valid JSON matching this shape (no markdown):
{
  "currency": "USD",
  "line_items": [
    {"description": "string", "amount": 0.0, "category": "groceries"}
  ],
  "notes": "optional string"
}
"""


def _load_sidecar_json(image_path: Path) -> dict[str, Any] | None:
    """Load companion JSON (same stem) for offline/demo runs."""
    sidecar = image_path.with_suffix(".json")
    if not sidecar.is_file():
        stem_json = image_path.parent / f"{image_path.stem}_bill.json"
        if stem_json.is_file():
            sidecar = stem_json
        else:
            return None
    try:
        raw = sidecar.read_text(encoding="utf-8")
        return json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "bill_extraction.sidecar_failed",
            extra={"path": str(sidecar), "error": str(exc)},
        )
        return None


def _parse_categorized_payload(
    payload: dict[str, Any],
    source_label: str,
) -> CategorizedExpenses:
    """Validate external JSON into CategorizedExpenses."""
    items_raw = payload.get("line_items")
    if not isinstance(items_raw, list):
        raise BillProcessingError("Invalid payload: line_items must be a list")
    line_items: list[LineItem] = []
    for entry in items_raw:
        if not isinstance(entry, dict):
            continue
        cat_raw = entry.get("category", "other")
        try:
            category = ExpenseCategory(str(cat_raw).lower())
        except ValueError:
            category = ExpenseCategory.OTHER
        desc = str(entry.get("description", "")).strip()
        if not desc:
            continue
        line_items.append(
            LineItem(
                description=desc,
                amount=entry["amount"],
                category=category,
            )
        )
    currency = str(payload.get("currency", "USD")).strip() or "USD"
    notes = payload.get("notes")
    notes_str = str(notes) if notes is not None else None
    return CategorizedExpenses(
        source_label=source_label,
        currency=currency,
        line_items=line_items,
        notes=notes_str,
    )


def _mime_for_image(path: Path) -> str:
    ext = path.suffix.lower()
    mapping: dict[str, str] = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    return mapping.get(ext, "image/jpeg")


def _extract_via_openai_vision(image_path: Path) -> CategorizedExpenses:
    settings = get_settings()
    if not settings.openai_api_key:
        raise BillProcessingError(
            "OPENAI_API_KEY is not set. Add it to your environment or .env file to process receipt "
            "images. You only need to upload a photo of the receipt; no JSON files are required."
        )
    raw_bytes = image_path.read_bytes()
    b64 = base64.standard_b64encode(raw_bytes).decode("ascii")
    mime = _mime_for_image(image_path)
    client = OpenAI(api_key=settings.openai_api_key)
    try:
        completion = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all line items from this bill image.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{b64}",
                            },
                        },
                    ],
                },
            ],
            temperature=0.2,
        )
    except Exception as exc:
        logger.exception(
            "bill_extraction.openai_failed",
            extra={"path": str(image_path), "action": "vision_extract"},
        )
        raise BillProcessingError(f"Vision API request failed: {exc}") from exc
    choice = completion.choices[0]
    content = choice.message.content
    if not content:
        raise BillProcessingError("Empty response from vision model")
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BillProcessingError(f"Model did not return valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise BillProcessingError("Model JSON must be an object")
    return _parse_categorized_payload(payload, source_label=str(image_path))


def extract_categorized_expenses(
    image_path: str | Path,
    *,
    allow_packaged_demo: bool = False,
) -> CategorizedExpenses:
    """
    Extract categorized expenses from a receipt image path.

    Normal use (receipt photo only):
    - Set OPENAI_API_KEY; the image is read and analyzed with vision (no JSON files needed).

    Optional overrides (development):
    - BILL_USE_SIDECAR_JSON=1: load same-stem .json next to the image if present.
    - BILL_AGENT_DEMO=1: use packaged data/sample_bill.json when no API key (demo only).
    """
    path = Path(image_path).resolve()
    if not path.is_file():
        raise BillProcessingError(f"Image not found: {path}")

    settings = get_settings()

    # Primary path: vision from the receipt image only
    if settings.openai_api_key:
        logger.info(
            "bill_extraction.vision",
            extra={"path": str(path), "action": "openai_vision"},
        )
        return _extract_via_openai_vision(path)

    if settings.bill_use_sidecar_json:
        sidecar = _load_sidecar_json(path)
        if sidecar is not None:
            logger.info(
                "bill_extraction.used_sidecar",
                extra={"path": str(path), "action": "sidecar_json"},
            )
            return _parse_categorized_payload(sidecar, source_label=str(path))

    if settings.bill_agent_demo or allow_packaged_demo:
        demo = Path(__file__).resolve().parents[3] / "data" / "sample_bill.json"
        if demo.is_file():
            logger.info(
                "bill_extraction.packaged_demo",
                extra={"path": str(path), "action": "sample_bill_json"},
            )
            payload = json.loads(demo.read_text(encoding="utf-8"))
            return _parse_categorized_payload(payload, source_label=str(path))

    raise BillProcessingError(
        "OPENAI_API_KEY is not set. To process a receipt, add your API key to .env (see .env.example) "
        "and run again with only your receipt image path. "
        "For offline tests, set BILL_USE_SIDECAR_JSON=1 or BILL_AGENT_DEMO=1."
    )
