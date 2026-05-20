"""Domain models and enums for expenses."""

from __future__ import annotations

from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class ExpenseCategory(str, Enum):
    """Fixed set of bill categories."""

    GROCERIES = "groceries"
    DINING = "dining"
    UTILITIES = "utilities"
    SHOPPING = "shopping"
    ENTERTAINMENT = "entertainment"
    OTHER = "other"


class LineItem(BaseModel):
    """Single line on a receipt or bill."""

    description: str = Field(min_length=1)
    amount: Decimal = Field(ge=0)
    category: ExpenseCategory

    @field_validator("amount", mode="before")
    @classmethod
    def coerce_decimal(cls, value: object) -> Decimal:
        if isinstance(value, Decimal):
            return value
        if isinstance(value, (int, float, str)):
            return Decimal(str(value))
        raise TypeError("amount must be convertible to Decimal")


class CategorizedExpenses(BaseModel):
    """Structured output from bill processing."""

    source_label: str = Field(min_length=1)
    currency: str = Field(default="USD", min_length=1)
    line_items: list[LineItem] = Field(default_factory=list)
    notes: str | None = None

    def total(self) -> Decimal:
        return sum((li.amount for li in self.line_items), Decimal("0"))


class CategoryTotal(BaseModel):
    """Aggregate spend for one category."""

    category: ExpenseCategory
    total: Decimal = Field(ge=0)


class ExpenseSummary(BaseModel):
    """Output from expense summarization."""

    category_totals: list[CategoryTotal]
    grand_total: Decimal
    highest_category: ExpenseCategory
    highest_amount: Decimal
    insights: list[str] = Field(default_factory=list)
