"""Web search for competitor and footfall-related signals."""

from __future__ import annotations

import logging
from typing import Any

from src.config import Settings
from src.errors import SearchError

logger = logging.getLogger(__name__)

MAX_SNIPPETS_PER_QUERY = 6
MAX_QUERY_CHARS = 240


def _normalize_snippet(title: str, body: str, url: str) -> str:
    title = title.strip() or "Untitled"
    body = (body or "").strip()
    url = url.strip()
    if url:
        return f"{title}\n{body}\nSource: {url}"
    return f"{title}\n{body}"


def _search_tavily(queries: list[str], settings: Settings) -> list[str]:
    key = settings.tavily_api_key
    if key is None:
        return []
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=key.get_secret_value())
    except Exception as exc:
        logger.exception("Failed to initialize Tavily client")
        raise SearchError(f"Tavily initialization failed: {exc}") from exc

    snippets: list[str] = []
    for raw_query in queries:
        query = raw_query.strip()[:MAX_QUERY_CHARS]
        if not query:
            continue
        try:
            response: dict[str, Any] = client.search(
                query=query,
                search_depth="advanced",
                max_results=5,
            )
        except Exception as exc:
            logger.warning(
                "Tavily search failed for query",
                extra={"query": query, "error": str(exc)},
            )
            continue
        results = response.get("results")
        if not isinstance(results, list):
            continue
        for item in results[:MAX_SNIPPETS_PER_QUERY]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", ""))
            content = str(item.get("content", ""))
            url = str(item.get("url", ""))
            snippets.append(_normalize_snippet(title, content, url))
    return snippets


def _search_duckduckgo(queries: list[str]) -> list[str]:
    try:
        from duckduckgo_search import DDGS
    except ImportError as exc:
        raise SearchError("duckduckgo-search is not installed") from exc

    snippets: list[str] = []
    for raw_query in queries:
        query = raw_query.strip()[:MAX_QUERY_CHARS]
        if not query:
            continue
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=5)
        except Exception as exc:
            logger.warning(
                "DuckDuckGo search failed for query",
                extra={"query": query, "error": str(exc)},
            )
            continue
        if not results:
            continue
        for item in results[:MAX_SNIPPETS_PER_QUERY]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", ""))
            body = str(item.get("body", ""))
            href = str(item.get("href", ""))
            snippets.append(_normalize_snippet(title, body, href))
    return snippets


def run_searches(queries: list[str], settings: Settings) -> list[str]:
    """
    Execute search queries and return flattened text snippets.

    Prefers Tavily when TAVILY_API_KEY is set; otherwise uses DuckDuckGo.
    """
    cleaned = [q.strip() for q in queries if q and q.strip()]
    if not cleaned:
        raise SearchError("No valid search queries were provided")

    if settings.tavily_api_key is not None:
        snippets = _search_tavily(cleaned, settings)
        if snippets:
            logger.info(
                "Search completed via Tavily",
                extra={"queries": len(cleaned), "snippets": len(snippets)},
            )
            return snippets
        logger.warning("Tavily returned no results; falling back to DuckDuckGo")

    snippets = _search_duckduckgo(cleaned)
    if not snippets:
        raise SearchError(
            "All search attempts returned no usable results. "
            "Check network access or set TAVILY_API_KEY for more reliable results."
        )
    logger.info(
        "Search completed via DuckDuckGo",
        extra={"queries": len(cleaned), "snippets": len(snippets)},
    )
    return snippets
