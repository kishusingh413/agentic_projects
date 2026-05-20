"""
Intelligent Travel Assistant: LangChain tool-calling agent with weather + web search.

Run:  python travel_assistant.py
Requires: .env with OPENAI_API_KEY and WEATHER_API_KEY (see .env.example).
Optional: ATTRACTION_SEARCH_PROVIDER=tavily and TAVILY_API_KEY for Tavily instead of DuckDuckGo.
"""

from __future__ import annotations

import logging
import sys
from typing import Sequence

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI

from config import Settings
from services.attractions_search import search_top_attractions
from services.weather import fetch_current_weather

logger = logging.getLogger(__name__)


def build_travel_tools(settings: Settings) -> Sequence[BaseTool]:
    """Create LangChain tools that delegate to service functions."""

    @tool
    def get_destination_weather(destination: str) -> str:
        """Current weather (temperature, sky conditions, humidity, wind) for a place.

        Use for cities, towns, or landmarks. Pass only the place name, e.g. "Lisbon" or "Bali".
        """
        return fetch_current_weather(destination, settings.weather_api_key)

    @tool
    def search_top_tourist_attractions(city_or_destination: str) -> str:
        """Search the web for top tourist attractions and notable sights.

        Pass the city or region the traveler asked about.
        """
        return search_top_attractions(
            city_or_destination,
            provider=settings.attraction_search_provider,
            tavily_api_key=settings.tavily_api_key,
        )

    return [get_destination_weather, search_top_tourist_attractions]


def build_agent_executor(settings: Settings, *, verbose: bool = True) -> AgentExecutor:
    """Bind the LLM, tools, and prompt; wrap with AgentExecutor."""
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )
    tools = list(build_travel_tools(settings))

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an intelligent travel assistant. When the user names or implies a "
                "destination, use the tools to fetch (1) current weather and (2) top tourist "
                "attractions. Call both tools with the same place name when appropriate. "
                "Then answer in two clear sections: **Weather** and **Things to see**. "
                "Synthesize briefly; do not invent tool facts. If a tool returns an error, "
                "state that and continue with whatever information you have.",
            ),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        handle_parsing_errors=True,
    )


def run_interactive_cli(executor: AgentExecutor) -> None:
    """Read destinations from stdin until the user quits."""
    print("Travel Assistant — type a destination (or 'quit' to exit).")
    while True:
        try:
            line = input("Destination> ").strip()
        except EOFError:
            print()
            break

        if not line:
            continue
        lowered = line.lower()
        if lowered in {"quit", "exit", "q"}:
            break

        user_message = (
            f"I'm planning a trip to {line}. "
            "Give me the current weather there and top tourist attractions to visit."
        )

        try:
            result = executor.invoke({"input": user_message})
        except Exception as exc:
            logger.exception(
                "Agent invocation failed",
                extra={"action": "run_interactive_cli", "destination": line},
            )
            print(f"Something went wrong: {exc}")
            continue

        output = result.get("output")
        if isinstance(output, str) and output.strip():
            print(output.strip())
        else:
            print("No assistant reply was produced.")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        settings = Settings()
    except Exception as exc:
        logger.error("Invalid configuration: %s", exc)
        sys.exit(1)

    logger.info(
        "Attraction search provider: %s",
        settings.attraction_search_provider.value,
    )

    try:
        executor = build_agent_executor(settings, verbose=True)
    except Exception as exc:
        logger.error("Failed to build agent: %s", exc)
        sys.exit(1)

    run_interactive_cli(executor)


if __name__ == "__main__":
    main()
