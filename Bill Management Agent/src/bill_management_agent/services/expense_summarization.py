"""Summarize categorized expenses and surface spending insights."""

from __future__ import annotations

import logging
from collections import defaultdict
from decimal import Decimal

from bill_management_agent.errors import SummarizationError
from bill_management_agent.models import (
    CategorizedExpenses,
    CategoryTotal,
    ExpenseCategory,
    ExpenseSummary,
)

logger = logging.getLogger(__name__)


def summarize_expenses(data: CategorizedExpenses) -> ExpenseSummary:
    """Compute per-category totals, highest category, and simple insights."""
    try:
        totals: dict[ExpenseCategory, Decimal] = defaultdict(lambda: Decimal("0"))
        for item in data.line_items:
            totals[item.category] += item.amount

        category_totals = [
            CategoryTotal(category=cat, total=amt)
            for cat, amt in sorted(totals.items(), key=lambda x: x[0].value)
        ]
        grand_total = sum((ct.total for ct in category_totals), Decimal("0"))

        if not category_totals or grand_total == 0:
            return ExpenseSummary(
                category_totals=category_totals,
                grand_total=grand_total,
                highest_category=ExpenseCategory.OTHER,
                highest_amount=Decimal("0"),
                insights=[
                    "No line items with amounts were found; add clearer receipt data or check extraction.",
                ],
            )

        highest = max(category_totals, key=lambda c: c.total)
        mean = grand_total / Decimal(len(data.line_items)) if data.line_items else Decimal("0")
        insights: list[str] = []
        insights.append(
            f"Largest share of spending: {highest.category.value} "
            f"({highest.total} {data.currency})."
        )
        if data.line_items:
            insights.append(
                f"Average line item: {mean.quantize(Decimal('0.01'))} {data.currency} "
                f"across {len(data.line_items)} items."
            )
        for ct in category_totals:
            if grand_total > 0 and ct.total / grand_total >= Decimal("0.4"):
                insights.append(
                    f"High concentration in {ct.category.value}: "
                    f"{(ct.total / grand_total * 100).quantize(Decimal('0.1'))}% of total."
                )
        return ExpenseSummary(
            category_totals=category_totals,
            grand_total=grand_total,
            highest_category=highest.category,
            highest_amount=highest.total,
            insights=insights,
        )
    except Exception as exc:
        logger.exception(
            "expense_summarization.failed",
            extra={"source": data.source_label, "action": "summarize_expenses"},
        )
        raise SummarizationError(str(exc)) from exc
