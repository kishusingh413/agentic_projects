from __future__ import annotations

import re
from typing import Optional

import requests
try:
    from ddgs import DDGS  # type: ignore
except ImportError:  # pragma: no cover
    from duckduckgo_search import DDGS

from research_agent.config import get_settings
from research_agent.errors import ExternalAPIError
from research_agent.logging_utils import log_exception
from research_agent.schemas import WebFinding


_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html_to_text(html: str) -> str:
    text = _TAG_RE.sub(" ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def web_search(query: str) -> list[WebFinding]:
    """
    Perform a live web search and return findings.
    """
    settings = get_settings()

    if not query.strip():
        raise ExternalAPIError("Empty query passed to web_search")

    try:
        # Prefer Tavily if API key is available.
        if settings.tavily_api_key is not None and settings.tavily_api_key.strip():
            try:
                from tavily import TavilyClient

                with TavilyClient(api_key=settings.tavily_api_key.strip()) as tavily:
                    resp = tavily.search(
                        query,
                        max_results=settings.web_max_results,
                        include_answer=False,
                        include_raw_content=False,
                        timeout=20,
                    )

                results = resp.get("results", []) if isinstance(resp, dict) else []
                findings: list[WebFinding] = []
                for res in results:
                    if not isinstance(res, dict):
                        continue
                    title = str(res.get("title", "")).strip()
                    url = str(res.get("url", "")).strip()
                    snippet = str(res.get("content", res.get("snippet", ""))).strip()
                    if not title or not url or not snippet:
                        continue
                    snippet = snippet[: settings.web_max_chars_per_snippet]
                    findings.append(
                        WebFinding(
                            title=title,
                            url=url,
                            snippet=snippet,
                            extracted_text=None,
                        )
                    )
                if findings:
                    return findings
            except Exception as tav_exc:  # noqa: BLE001
                # If Tavily fails for any reason, fall back to DDGS.
                log_exception("web_search_tavily_fallback", tav_exc, extra={"query": query})

        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=settings.web_max_results)
            findings: list[WebFinding] = []
            for idx, res in enumerate(results):
                title = str(res.get("title", "")).strip()
                url = str(res.get("href", "")).strip()
                snippet = str(res.get("body", "")).strip()

                if not title or not url or not snippet:
                    continue

                snippet = snippet[: settings.web_max_chars_per_snippet]
                extracted_text: Optional[str] = None

                # Best-effort fetch. If it fails, we still return snippet.
                if idx < 2:
                    try:
                        resp = requests.get(
                            url,
                            timeout=10,
                            headers={"User-Agent": "Mozilla/5.0"},
                        )
                        if resp.status_code == 200:
                            extracted = _strip_html_to_text(resp.text)
                            extracted_text = extracted[: settings.web_max_chars_per_snippet]
                    except Exception:
                        extracted_text = None

                findings.append(
                    WebFinding(
                        title=title,
                        url=url,
                        snippet=snippet,
                        extracted_text=extracted_text,
                    )
                )
            return findings
    except ExternalAPIError:
        raise
    except Exception as exc:  # noqa: BLE001
        log_exception("web_search", exc, extra={"query": query})
        raise ExternalAPIError("Web search failed") from exc

