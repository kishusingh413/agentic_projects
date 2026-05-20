"""
CrewAI multi-agent system for logistics analysis and optimization strategy.

Requires OPENAI_API_KEY (or provider-specific keys per CrewAI docs) in the environment.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Sequence

from crewai import Agent, Crew, LLM, Process, Task
from dotenv import load_dotenv

from services.errors import AppError, MissingEnvironmentError, ProductsInputError
from services.product_loader import load_products_from_file


def format_products_block(products: Sequence[str]) -> str:
    """Human-readable block listing products/SKUs for task prompts."""
    items = [p.strip() for p in products if p and str(p).strip()]
    if not items:
        return (
            "No specific product list was supplied. Analyze general logistics operations "
            "and industry-typical route efficiency and inventory turnover patterns."
        )
    lines = "\n".join(f"  - {p}" for p in items)
    return f"Products / SKUs to prioritize for this analysis:\n{lines}"


def create_logistics_analyst(llm: str | LLM | None = None) -> Agent:
    base = dict(
        role="Logistics Analyst",
        goal=(
            "Produce a concise, evidence-grounded assessment of current logistics "
            "operations with emphasis on route efficiency and inventory turnover trends."
        ),
        backstory=(
            "You are a senior logistics analyst with years in distribution and last-mile "
            "operations. You synthesize operational KPIs, constraints, and trade-offs. "
            "You flag bottlenecks, variability, and data gaps without inventing numbers—"
            "when data is missing, you state assumptions clearly."
        ),
        verbose=True,
        allow_delegation=False,
    )
    if llm is None:
        return Agent(**base)
    return Agent(**base, llm=llm)


def create_optimization_strategist(llm: str | LLM | None = None) -> Agent:
    base = dict(
        role="Optimization Strategist",
        goal=(
            "Turn operational insights into actionable optimization strategies with clear "
            "priorities, risks, and measurable next steps for the given product scope."
        ),
        backstory=(
            "You are an operations research and supply-chain strategist. You design "
            "practical improvements (routing, slotting, safety stock, replenishment) "
            "that respect real-world constraints. You align recommendations with the "
            "analyst's findings and call out dependencies and implementation order."
        ),
        verbose=True,
        allow_delegation=False,
    )
    if llm is None:
        return Agent(**base)
    return Agent(**base, llm=llm)


def build_tasks(
    products: Sequence[str],
    analyst: Agent,
    strategist: Agent,
) -> tuple[Task, Task]:
    """Create parametrized tasks; `products` drives the scope of both tasks."""
    product_block = format_products_block(products)

    research_task = Task(
        description=(
            "Research and summarize the current state of logistics operations relevant to "
            "optimization. Cover BOTH of the following angles where applicable:\n"
            "1) Route efficiency: fleet utilization, stop density, time windows, "
            "backhauls, and factors hurting on-time performance.\n"
            "2) Inventory turnover: stock levels, aging, seasonality, and SKU-level "
            "movement patterns.\n\n"
            f"{product_block}\n\n"
            "Deliver structured insights suitable as input for downstream strategy work."
        ),
        expected_output=(
            "A structured brief (sections with bullet points) including: key findings, "
            "assumed or stated constraints, risks, and open questions. Explicitly relate "
            "findings to the listed products when a list was provided."
        ),
        agent=analyst,
    )

    strategy_task = Task(
        description=(
            "Using ONLY the insights and context from the Logistics Analyst's output, "
            "design an optimization strategy for the same scope.\n\n"
            f"{product_block}\n\n"
            "Include: prioritized initiatives, rationale tied to the analyst's findings, "
            "quick wins vs longer-term bets, metrics to track, and major risks or "
            "dependencies. If products were listed, tailor actions per product or product "
            "group where it adds value."
        ),
        expected_output=(
            "An optimization strategy document with clear headings: objectives, "
            "recommended initiatives (prioritized), KPIs, rollout sequence, and risks."
        ),
        agent=strategist,
        context=[research_task],
    )

    return research_task, strategy_task


def build_crew(products: Sequence[str], llm: str | LLM | None = None) -> Crew:
    analyst = create_logistics_analyst(llm=llm)
    strategist = create_optimization_strategist(llm=llm)
    research_task, strategy_task = build_tasks(products, analyst, strategist)

    return Crew(
        agents=[analyst, strategist],
        tasks=[research_task, strategy_task],
        process=Process.sequential,
        verbose=True,
    )


def parse_products_arg(values: list[str] | None) -> list[str]:
    """Support repeated `--product` flags or comma-separated `--products`."""
    if not values:
        return []
    out: list[str] = []
    for raw in values:
        for part in raw.split(","):
            s = part.strip()
            if s:
                out.append(s)
    return out


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run Logistics Analyst + Optimization Strategist crew."
    )
    parser.add_argument(
        "--product",
        action="append",
        dest="product_items",
        metavar="NAME",
        default=None,
        help="Product or SKU to include (repeatable).",
    )
    parser.add_argument(
        "--products",
        type=str,
        default=None,
        dest="products_csv",
        help="Comma-separated list of products/SKUs (alternative to repeated --product).",
    )
    parser.add_argument(
        "--products-file",
        type=str,
        default=None,
        dest="products_file",
        help="Path to a products file (CSV, Excel .xlsx/.xls, or JSON).",
    )
    parser.add_argument(
        "--products-file-format",
        type=str,
        default=None,
        dest="products_file_format",
        help="Optional override: csv | excel | json. If omitted, inferred from extension.",
    )
    parser.add_argument(
        "--sku-column",
        type=str,
        default=None,
        dest="sku_column",
        help="Optional column name for the SKU field in CSV/Excel (e.g., 'sku', 'product_id').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Load and validate products, then print the derived product prompt text. Does not call the LLM.",
    )
    args = parser.parse_args()

    product_list: list[str] = []
    if args.product_items:
        product_list.extend(parse_products_arg(args.product_items))
    if args.products_csv:
        product_list.extend(parse_products_arg([args.products_csv]))
    if args.products_file:
        try:
            file_products = load_products_from_file(
                args.products_file,
                file_format=args.products_file_format,
                sku_column=args.sku_column,
            )
            product_list.extend(file_products)
        except ProductsInputError as e:
            e.log()
            sys.exit(1)

    if args.dry_run:
        if not product_list:
            print("No products loaded (product_list is empty).", file=sys.stderr)
            sys.exit(0)
        print("Loaded products for prompt scope:\n")
        print("\n".join(f"- {p}" for p in product_list))
        sys.exit(0)

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None or not api_key.strip():
        err = MissingEnvironmentError(
            "OPENAI_API_KEY is not set or is empty. Add it to your environment or a .env file "
            "in this directory (see env.example).",
        )
        err.log()
        sys.exit(1)

    try:
        crew = build_crew(product_list)
        result = crew.kickoff()
        print("\n--- Crew result ---\n")
        print(result)
    except AppError as e:
        e.log()
        sys.exit(1)


if __name__ == "__main__":
    main()
