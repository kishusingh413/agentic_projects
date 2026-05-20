"""Web search for tourist attractions via DuckDuckGo or Tavily (configurable)."""

from __future__ import annotations

import logging
from typing import Any

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

from config import AttractionSearchProvider

logger = logging.getLogger(__name__)

_ddg_runner: DuckDuckGoSearchRun | None = None


def _get_ddg() -> DuckDuckGoSearchRun:
    global _ddg_runner
    if _ddg_runner is None:
        _ddg_runner = DuckDuckGoSearchRun()
    return _ddg_runner


def _format_tavily_results(raw: dict[str, Any]) -> str:
    """Turn Tavily JSON into a compact string for the LLM."""
    lines: list[str] = []
    answer = raw.get("answer")
    if isinstance(answer, str) and answer.strip():
        lines.append(f"Summary: {answer.strip()}")

    results = raw.get("results")
    if not isinstance(results, list) or not results:
        if lines:
            return "\n".join(lines)
        return "Tavily returned no result entries."

    for i, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        url = item.get("url")
        content = item.get("content")
        title_s = title.strip() if isinstance(title, str) else ""
        url_s = url.strip() if isinstance(url, str) else ""
        content_s = content.strip() if isinstance(content, str) else ""
        block = f"{i}. {title_s or 'Untitled'}\n   {content_s}"
        if url_s:
            block += f"\n   Source: {url_s}"
        lines.append(block)

    return "\n".join(lines) if lines else "Tavily returned no usable snippets."


def _search_tavily(query: str, api_key: str) -> str:
    wrapper = TavilySearchAPIWrapper(tavily_api_key=api_key)
    try:
        raw = wrapper.raw_results(
            query,
            max_results=5,
            search_depth="advanced",
            include_answer=True,
        )
    except Exception as exc:
        logger.exception(
            "Tavily search failed",
            extra={"action": "search_top_attractions", "query": query},
        )
        return f"Attraction search (Tavily) failed: {exc}"

    if not isinstance(raw, dict):
        return "Tavily returned an unexpected response shape."

    return _format_tavily_results(raw)


def _search_duckduckgo(query: str) -> str:
    try:
        raw = _get_ddg().invoke(query)
    except Exception as exc:
        logger.exception(
            "DuckDuckGo search failed",
            extra={"action": "search_top_attractions", "query": query},
        )
        return f"Attraction search failed: {exc}"

    text = raw.strip() if isinstance(raw, str) else ""
    if not text:
        return "No search results were returned; try a more specific city name."

    return text


def search_top_attractions(
    city_or_destination: str,
    *,
    provider: AttractionSearchProvider,
    tavily_api_key: str | None,
) -> str:
    """
    Search the web for top attractions in *city_or_destination* using the configured provider.
    """
    place = city_or_destination.strip()
    if not place:
        return "Error: no destination was provided for the attractions search."

    query = f"top tourist attractions and must-see places in {place}"

    if provider == AttractionSearchProvider.TAVILY:
        if not tavily_api_key:
            return "Configuration error: Tavily is selected but no API key is available."
        return _search_tavily(query, tavily_api_key)

    return _search_duckduckgo(query)
