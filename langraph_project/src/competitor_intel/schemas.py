from enum import Enum

from pydantic import BaseModel, Field


class BeneficiaryAudience(str, Enum):
    OWNER = "business_owner"
    MARKETING = "marketing_strategy"
    REAL_ESTATE = "real_estate_location"
    INVESTOR = "investor_analyst"


class ParsedQuery(BaseModel):
    """Structured extraction from the latest user message."""

    location: str | None = Field(
        default=None,
        description="Neighborhood, city, or address anchor for local search.",
    )
    store_focus: str = Field(
        default="clothing and apparel retail",
        description="Retail category to emphasize.",
    )
    audience: BeneficiaryAudience | None = Field(
        default=None,
        description="Optional audience to tailor recommendations.",
    )
    needs_clarification: bool = Field(
        default=False,
        description="True if location or intent is too vague to search.",
    )
    clarification_question: str | None = Field(
        default=None,
        description="Question to ask the user when clarification is required.",
    )


class SearchHit(BaseModel):
    title: str
    snippet: str
    url: str | None = None
