"""Entry point: group chat workflow with StateFlow routing and advisor report output."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from decimal import Decimal, InvalidOperation
from typing import NoReturn

from autogen_agentchat.base import TaskResult

from src.config import get_settings
from src.errors import PortfolioWorkflowError
from src.models import CurrencyCode, UserPortfolioInput
from src.stateflow import INVESTMENT_ADVISOR
from src.workflow_service import run_portfolio_workflow

logger = logging.getLogger(__name__)


def _parse_decimal(prompt: str) -> Decimal:
    while True:
        raw = input(f"{prompt}: ").strip()
        try:
            return Decimal(raw)
        except InvalidOperation:
            print("Invalid number. Try again (e.g. 75000 or 5000.50).")


def _collect_portfolio_interactive() -> UserPortfolioInput:
    print("\n--- Financial Portfolio Manager — enter your profile ---\n")
    salary = _parse_decimal("Annual gross salary")
    print("Currency code (USD, EUR, GBP, INR) [USD]: ", end="")
    cur_raw = input().strip().upper() or "USD"
    try:
        currency = CurrencyCode(cur_raw)
    except ValueError:
        print("Unknown currency, defaulting to USD.")
        currency = CurrencyCode.USD

    fd = _parse_decimal("Fixed deposits / cash-like balance (total)")
    sip = _parse_decimal("Total monthly SIP / recurring investment amount")
    re_val = _parse_decimal("Estimated real estate value (0 if none)")
    eq = _parse_decimal("Equities / mutual funds holdings (total)")
    print("Other assets or notes (optional, single line): ", end="")
    note = input().strip()

    return UserPortfolioInput(
        salary_annual=salary,
        currency=currency,
        fixed_deposits=fd,
        sip_monthly=sip,
        real_estate_value=re_val,
        equities_value=eq,
        other_assets_note=note,
    )


def _demo_portfolio() -> UserPortfolioInput:
    return UserPortfolioInput(
        salary_annual=Decimal("95000"),
        currency=CurrencyCode.USD,
        fixed_deposits=Decimal("20000"),
        sip_monthly=Decimal("1500"),
        real_estate_value=Decimal("120000"),
        equities_value=Decimal("35000"),
        other_assets_note="401(k) match only; no crypto.",
    )


def _extract_advisor_report(task_result: TaskResult) -> str:
    for message in reversed(task_result.messages):
        if message.source == INVESTMENT_ADVISOR:
            return message.to_text()
    return ""


def _fail(message: str, *, code: int = 1) -> NoReturn:
    logger.error(message)
    print(message, file=sys.stderr)
    raise SystemExit(code)


async def _run_workflow(portfolio: UserPortfolioInput) -> None:
    try:
        settings = get_settings()
    except Exception as exc:
        _fail(
            "Configuration error: set OPENAI_API_KEY in the environment or a .env file. "
            f"Details: {exc}"
        )

    task_result = await run_portfolio_workflow(portfolio, settings=settings)

    report = _extract_advisor_report(task_result)
    print("\n" + "=" * 72)
    print("INVESTMENT ADVISOR AGENT — PERSONALIZED FINANCIAL REPORT")
    print("=" * 72 + "\n")
    if report:
        print(report)
    else:
        print("No advisor message was produced. Check logs for earlier agent outputs.")
    print("\n" + "=" * 72)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("autogen_core", "autogen_agentchat", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Financial Portfolio Manager (multi-agent).")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use built-in sample portfolio data instead of interactive prompts.",
    )
    args = parser.parse_args()

    portfolio = _demo_portfolio() if args.demo else _collect_portfolio_interactive()

    try:
        asyncio.run(_run_workflow(portfolio))
    except PortfolioWorkflowError as exc:
        _fail(str(exc), code=2)
    except SystemExit:
        raise
    except Exception as exc:
        logger.exception("Fatal error", extra={"action": "main"})
        _fail(f"Fatal error: {exc}", code=3)


if __name__ == "__main__":
    main()
