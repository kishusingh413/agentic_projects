"""Validated user inputs for portfolio analysis."""

from decimal import Decimal
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator


class CurrencyCode(str, Enum):
    """Supported base currencies for display (extend as needed)."""

    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    INR = "INR"


class UserPortfolioInput(BaseModel):
    """Structured portfolio data collected before the agent workflow runs."""

    salary_annual: Annotated[Decimal, Field(gt=0, description="Gross annual salary.")]
    currency: CurrencyCode = CurrencyCode.USD
    fixed_deposits: Annotated[Decimal, Field(ge=0)] = Decimal("0")
    sip_monthly: Annotated[Decimal, Field(ge=0)] = Decimal("0")
    real_estate_value: Annotated[Decimal, Field(ge=0)] = Decimal("0")
    equities_value: Annotated[Decimal, Field(ge=0)] = Decimal("0")
    other_assets_note: str = ""

    @field_validator("other_assets_note")
    @classmethod
    def strip_note(cls, value: str) -> str:
        return value.strip()

    def to_prompt_block(self) -> str:
        """Format validated inputs for the user proxy and downstream agents."""
        lines = [
            f"Annual salary ({self.currency.value}): {self.salary_annual}",
            f"Fixed deposits / cash-like ({self.currency.value}): {self.fixed_deposits}",
            f"Monthly SIP total ({self.currency.value}): {self.sip_monthly}",
            f"Real estate estimate ({self.currency.value}): {self.real_estate_value}",
            f"Equities / funds ({self.currency.value}): {self.equities_value}",
        ]
        if self.other_assets_note:
            lines.append(f"Other assets / notes: {self.other_assets_note}")
        return "\n".join(lines)
