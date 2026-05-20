"""Load and validate environment variables at startup."""

import logging
import os
from typing import Final

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

logger: Final[logging.Logger] = logging.getLogger(__name__)


class AppConfig(BaseModel):
    """Validated API configuration (never trust raw os.environ)."""

    openai_api_key: str = Field(
        ...,
        description="OpenAI API key for the LLM used by CrewAI",
    )
    serper_api_key: str = Field(
        ...,
        description="Serper API key for Google search via Serper",
    )

    @field_validator("openai_api_key", "serper_api_key")
    @classmethod
    def strip_nonempty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("API key must be a non-empty string")
        return cleaned


def load_config() -> AppConfig:
    """
    Load .env then validate required keys.
    Sets process env so CrewAI and SerperDevTool pick up credentials.
    """
    load_dotenv()
    raw_openai = os.getenv("OPENAI_API_KEY", "")
    raw_serper = os.getenv("SERPER_API_KEY", "")
    try:
        config = AppConfig(
            openai_api_key=raw_openai,
            serper_api_key=raw_serper,
        )
    except Exception as exc:
        logger.error(
            "Invalid environment configuration",
            extra={
                "has_openai": bool(raw_openai.strip()),
                "has_serper": bool(raw_serper.strip()),
            },
        )
        raise RuntimeError(
            "Missing or invalid API keys. Set OPENAI_API_KEY and SERPER_API_KEY "
            "in a .env file (see .env.example)."
        ) from exc

    os.environ["OPENAI_API_KEY"] = config.openai_api_key
    os.environ["SERPER_API_KEY"] = config.serper_api_key
    return config
