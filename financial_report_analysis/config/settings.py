"""Application settings with environment variable validation."""

import os
from functools import lru_cache


def _parse_env_bool(value: str | None, default: bool = False) -> bool:
    """Parse a string into a boolean, validating known values."""
    if value is None or value.strip() == "":
        return default
    normalized = value.strip().lower()
    if normalized in ("true", "1", "yes", "on"):
        return True
    if normalized in ("false", "0", "no", "off"):
        return False
    return default


def _parse_env_str(value: str | None, default: str = "") -> str:
    """Return trimmed string or default."""
    if value is None:
        return default
    return value.strip() or default


@lru_cache(maxsize=1)
def get_settings() -> "AppSettings":
    """Load and validate settings from environment. Cached for performance."""
    return AppSettings()


class AppSettings:
    """Validated application settings loaded from environment variables."""

    def __init__(self) -> None:
        self._openai_api_key = _parse_env_str(os.getenv("OPENAI_API_KEY"))
        self._model = _parse_env_str(
            os.getenv("OPENAI_VISION_MODEL", "gpt-4o"),
        )
        self._debug = _parse_env_bool(os.getenv("DEBUG", "false"))

        self._validate()

    def _validate(self) -> None:
        """Validate required settings at startup."""
        if not self._openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required. Set it in .env or your environment."
            )

        valid_models: tuple[str, ...] = (
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4-turbo-preview",
            "gpt-4-vision-preview",
        )
        if self._model not in valid_models:
            raise ValueError(
                f"OPENAI_VISION_MODEL must be one of {valid_models}. "
                f"Got: {self._model}"
            )

    @property
    def openai_api_key(self) -> str:
        return self._openai_api_key

    @property
    def model(self) -> str:
        return self._model

    @property
    def debug(self) -> bool:
        return self._debug

    @property
    def vision_model(self) -> str:
        return self._model

    @property
    def max_tokens(self) -> int:
        raw = os.getenv("OPENAI_MAX_TOKENS", "4096")
        try:
            val = int(raw)
            return max(512, min(16000, val))
        except ValueError:
            return 4096

    @property
    def temperature(self) -> float:
        raw = os.getenv("OPENAI_TEMPERATURE", "0.2")
        try:
            val = float(raw)
            return max(0.0, min(2.0, val))
        except ValueError:
            return 0.2
