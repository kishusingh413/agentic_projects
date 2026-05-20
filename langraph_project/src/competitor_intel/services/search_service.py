import logging
from typing import Sequence

from competitor_intel.config import get_settings
from competitor_intel.errors import SearchError
from competitor_intel.schemas import SearchHit

logger = logging.getLogger(__name__)


def _run_tavily(
    queries: Sequence[str],
    *,
    api_key: str,
    max_results_per_query: int,
) -> list[SearchHit]:
    try:
        from tavily import TavilyClient
    except ImportError as exc:
        raise SearchError("tavily-python is not installed.") from exc

    client = TavilyClient(api_key=api_key)
    hits: list[SearchHit] = []
    for query in queries:
        stripped = query.strip()
        if not stripped:
            continue
        try:
            response = client.search(query=stripped, max_results=max_results_per_query)
        except Exception as exc:
            logger.warning(
                "Tavily query failed",
                extra={"query": stripped, "error": str(exc)},
            )
            continue
        results = response.get("results") if isinstance(response, dict) else None
        if not isinstance(results, list):
            continue
        for row in results:
            if not isinstance(row, dict):
                continue
            title = str(row.get("title") or "").strip()
            body = str(row.get("content") or row.get("snippet") or "").strip()
            url = row.get("url")
            url_str = str(url).strip() if url else None
            if not title and not body:
                continue
            hits.append(
                SearchHit(
                    title=title or "(no title)",
                    snippet=body,
                    url=url_str,
                )
            )
    return hits


def _run_duckduckgo(queries: Sequence[str], max_results_per_query: int) -> list[SearchHit]:
    try:
        from duckduckgo_search import DDGS
    except ImportError as exc:
        raise SearchError("duckduckgo-search is not installed.") from exc

    hits: list[SearchHit] = []
    ddgs = DDGS()
    try:
        for query in queries:
            stripped = query.strip()
            if not stripped:
                continue
            try:
                raw = ddgs.text(stripped, max_results=max_results_per_query)
            except Exception as exc:
                logger.warning(
                    "DuckDuckGo query failed",
                    extra={"query": stripped, "error": str(exc)},
                )
                continue
            for item in raw or []:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or "").strip()
                body = str(item.get("body") or item.get("snippet") or "").strip()
                url = item.get("href") or item.get("url")
                url_str = str(url).strip() if url else None
                if not title and not body:
                    continue
                hits.append(
                    SearchHit(
                        title=title or "(no title)",
                        snippet=body,
                        url=url_str,
                    )
                )
    finally:
        close = getattr(ddgs, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                logger.debug("DDGS close failed: %s", exc)

    return hits


def run_text_searches(queries: Sequence[str], max_results_per_query: int = 5) -> list[SearchHit]:
    if not queries:
        return []
    settings = get_settings()
    tavily_key = settings.tavily_api_key
    try:
        if tavily_key is not None:
            return _run_tavily(
                queries,
                api_key=tavily_key,
                max_results_per_query=max_results_per_query,
            )
        return _run_duckduckgo(queries, max_results_per_query)
    except SearchError:
        raise
    except Exception as exc:
        raise SearchError("Search backend failed.") from exc


def format_hits_for_prompt(hits: Sequence[SearchHit], limit: int = 25) -> str:
    lines: list[str] = []
    for idx, hit in enumerate(hits[:limit], start=1):
        url_part = f" — {hit.url}" if hit.url else ""
        lines.append(f"{idx}. {hit.title}{url_part}\n   {hit.snippet}")
    return "\n\n".join(lines) if lines else "No search results were returned."
