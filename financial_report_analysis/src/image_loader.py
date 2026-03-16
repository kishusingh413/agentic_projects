"""Load financial document images from URLs or local paths using PIL."""

import io
from pathlib import Path
from typing import List, Union

import requests
from PIL import Image

# Supported image formats for financial documents
SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"}


def _is_url(source: str) -> bool:
    """Check if the source string is a URL."""
    return source.strip().lower().startswith(("http://", "https://"))


def _load_from_url(url: str) -> Image.Image:
    """Load an image from a URL."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        return image.convert("RGB")
    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch image from URL '{url}': {e}") from e
    except Exception as e:
        raise ValueError(f"Failed to decode image from URL '{url}': {e}") from e


def _load_from_path(path: Union[str, Path]) -> Image.Image:
    """Load an image from a local file path."""
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    if resolved.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported image format '{resolved.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )
    try:
        image = Image.open(resolved)
        return image.convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to open image at '{path}': {e}") from e


def load_image(source: Union[str, Path]) -> Image.Image:
    """Load a single image from a URL or local path."""
    source_str = str(source).strip()
    if not source_str:
        raise ValueError("Image source cannot be empty")
    if _is_url(source_str):
        return _load_from_url(source_str)
    return _load_from_path(source_str)


def load_images(sources: List[Union[str, Path]]) -> List[Image.Image]:
    """
    Load multiple images from a list of URLs or local paths.
    Returns a list of PIL Image objects in RGB mode.
    """
    images: List[Image.Image] = []
    for i, source in enumerate(sources):
        try:
            img = load_image(source)
            images.append(img)
        except Exception as e:
            raise ValueError(f"Error loading image at index {i} (source: {source}): {e}") from e
    return images
