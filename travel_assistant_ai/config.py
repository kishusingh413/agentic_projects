"""Load and validate environment configuration at startup."""

from enum import Enum
from typing import Self

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AttractionSearchProvider(str, Enum):
    """Backend for `search_top_attractions` (DuckDuckGo or Tavily)."""

    DUCKDUCKGO = "duckduckgo"
    TAVILY = "tavily"


class Settings(BaseSettings):
    """Application settings sourced from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(..., description="OpenAI API key for the chat model")
    weather_api_key: str = Field(..., description="WeatherAPI.com API key")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model id")
    attraction_search_provider: AttractionSearchProvider = Field(
        default=AttractionSearchProvider.DUCKDUCKGO,
        description="duckduckgo (no extra key) or tavily (requires TAVILY_API_KEY)",
    )
    tavily_api_key: str | None = Field(
        default=None,
        description="Tavily API key; required when attraction_search_provider is tavily",
    )

    @field_validator("openai_api_key", "weather_api_key")
    @classmethod
    def strip_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must be a non-empty string")
        return stripped

    @field_validator("tavily_api_key", mode="before")
    @classmethod
    def empty_tavily_key_to_none(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("TAVILY_API_KEY must be a string when set")
        stripped = value.strip()
        return stripped if stripped else None

    @model_validator(mode="after")
    def require_tavily_key_when_tavily(self) -> Self:
        if self.attraction_search_provider == AttractionSearchProvider.TAVILY:
            if not self.tavily_api_key:
                raise ValueError(
                    "TAVILY_API_KEY is required when ATTRACTION_SEARCH_PROVIDER=tavily"
                )
        return self
