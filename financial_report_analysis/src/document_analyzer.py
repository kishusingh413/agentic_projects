"""AI-powered document analyzer using vision-capable LLM."""

import base64
import io
from typing import List, Optional

from openai import OpenAI
from PIL import Image

from config.settings import get_settings
from src.prompts import SYSTEM_PROMPT, UserRole, get_user_prompt


def _image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Convert a PIL Image to base64-encoded string for API submission."""
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _build_image_content(image: Image.Image) -> dict:
    """Build API content block for a single image."""
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{_image_to_base64(image)}",
        },
    }


def analyze_financial_documents(
    images: List[Image.Image],
    user_prompt: str,
    role: UserRole = "general",
    model: Optional[str] = None,
) -> str:
    """
    Analyze financial document images and return a structured summary.

    Args:
        images: List of PIL Image objects (financial documents)
        user_prompt: User's analysis request or question
        role: Target audience (investor, analyst, auditor, general)
        model: Optional override for the vision model

    Returns:
        Structured summary text from the AI
    """
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    resolved_model = model or settings.vision_model

    if not images:
        return "No images provided for analysis."

    # Build message content: [text, img1, img2, ...]
    content: List[dict] = [
        {"type": "text", "text": get_user_prompt(user_prompt, role)},
    ]
    for img in images:
        content.append(_build_image_content(img))

    try:
        response = client.chat.completions.create(
            model=resolved_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
        )
        choice = response.choices[0]
        if choice.finish_reason == "length":
            return (choice.message.content or "") + "\n\n[Output truncated due to length limit.]"
        return choice.message.content or ""
    except Exception as e:
        raise RuntimeError(f"AI analysis failed: {e}") from e
