from __future__ import annotations

from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser

from research_agent.config import get_settings
from research_agent.errors import ExternalAPIError
from research_agent.logging_utils import log_exception
from research_agent.schemas import FinalResponse, RagFinding, WebFinding
from research_agent.services.llm_service import get_chat_model
from research_agent.types import RouteType


SUMMARIZER_SYSTEM_PROMPT: str = (
    "You are a research summarization agent. "
    "Given optional web findings, optional RAG findings, and an optional LLM draft, "
    "produce a final, well-structured answer. "
    "Use markdown. Include a concise 'Sources' section with the provided source URLs/paths. "
    "If evidence is missing, say so and provide a best-effort answer with clear uncertainty."
)


def summarize(
    query: str,
    router_route: RouteType,
    *,
    web_findings: list[WebFinding],
    rag_findings: list[RagFinding],
    llm_draft: Optional[str],
) -> FinalResponse:
    settings = get_settings()

    # Offline fallback if OpenAI key is missing.
    if settings.openai_api_key is None:
        return _summarize_offline(query, router_route, web_findings, rag_findings, llm_draft)

    parser = PydanticOutputParser(pydantic_object=FinalResponse)
    format_instructions = parser.get_format_instructions()
    model = get_chat_model()

    web_text = ""
    if web_findings:
        web_text = "WEB FINDINGS:\n" + "\n".join(
            f"TITLE: {f.title}\nURL: {f.url}\nSNIPPET: {f.snippet}\n" for f in web_findings
        )

    rag_text = ""
    if rag_findings:
        rag_text = "RAG FINDINGS:\n" + "\n".join(
            f"DOC_ID: {r.doc_id}\nSOURCE: {r.source_path}\nEXCERPT: {r.text}\n" for r in rag_findings
        )

    draft_text = ""
    if llm_draft:
        draft_text = f"LLM DRAFT:\n{llm_draft}\n"

    human = HumanMessage(
        content=(
            f"User query:\n{query}\n\n"
            f"Selected route: {router_route.value}\n\n"
            f"{draft_text}\n"
            f"{web_text}\n\n"
            f"{rag_text}\n\n"
            f"{format_instructions}"
        )
    )
    system = SystemMessage(content=SUMMARIZER_SYSTEM_PROMPT)

    try:
        response = model.invoke([system, human])
        parsed: FinalResponse = parser.parse(response.content)

        sources: list[str] = []
        for w in web_findings:
            if isinstance(w.url, str) and w.url.strip():
                sources.append(w.url.strip())
        for r in rag_findings:
            if isinstance(r.source_path, str) and r.source_path.strip():
                sources.append(r.source_path.strip())
        parsed.sources = parsed.sources or sources
        return parsed
    except ExternalAPIError:
        raise
    except Exception as exc:  # noqa: BLE001
        log_exception(
            "summarize",
            exc,
            extra={"query": query, "route": router_route.value},
        )
        # If key disappeared at runtime, still return deterministic output.
        if settings.openai_api_key is None:
            return _summarize_offline(query, router_route, web_findings, rag_findings, llm_draft)
        raise ExternalAPIError("Summarization failed") from exc


def _summarize_offline(
    query: str,
    router_route: RouteType,
    web_findings: list[WebFinding],
    rag_findings: list[RagFinding],
    llm_draft: Optional[str],
) -> FinalResponse:
    sources: list[str] = []
    lines: list[str] = []

    lines.append(f"# Answer (offline fallback)")
    lines.append("")
    lines.append(f"## Query")
    lines.append(query)
    lines.append("")
    lines.append(f"## Routed via")
    lines.append(router_route.value)
    lines.append("")

    if llm_draft:
        lines.append("## Draft")
        lines.append(llm_draft)
        lines.append("")

    if web_findings:
        lines.append("## Web evidence")
        for w in web_findings:
            sources.append(w.url)
            lines.append(f"- **{w.title}** ({w.url})")
            lines.append(f"  - {w.snippet}")
        lines.append("")

    if rag_findings:
        lines.append("## RAG evidence")
        for r in rag_findings:
            sources.append(r.source_path)
            lines.append(f"- **{r.doc_id}** ({r.source_path})")
            lines.append(f"  - {r.text[:400]}...")
        lines.append("")

    lines.append("## Summary")
    lines.append(
        "This is a deterministic offline summary because `OPENAI_API_KEY` is not set. "
        "With a key, the summarization agent would synthesize a fully formatted LLM response."
    )

    # De-duplicate sources while preserving order
    seen: set[str] = set()
    unique_sources: list[str] = []
    for s in sources:
        if s not in seen and s.strip():
            seen.add(s)
            unique_sources.append(s)

    return FinalResponse(
        answer_markdown="\n".join(lines).strip(),
        sources=unique_sources,
        debug={"mode": "offline_fallback"},
    )

