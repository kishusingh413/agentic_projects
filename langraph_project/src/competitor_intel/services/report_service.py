from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from competitor_intel.schemas import BeneficiaryAudience


def _audience_label(audience: BeneficiaryAudience | None) -> str:
    if audience is None:
        return "general retail stakeholder"
    mapping: dict[BeneficiaryAudience, str] = {
        BeneficiaryAudience.OWNER: "store owner or manager",
        BeneficiaryAudience.MARKETING: "marketing and strategy team",
        BeneficiaryAudience.REAL_ESTATE: "real estate or location analyst",
        BeneficiaryAudience.INVESTOR: "investor or market analyst",
    }
    return mapping[audience]


def generate_competitor_report(
    llm: ChatOpenAI,
    *,
    location: str,
    store_focus: str,
    audience: BeneficiaryAudience | None,
    search_context: str,
    user_request: str,
) -> str:
    reader = _audience_label(audience)
    system = SystemMessage(
        content=(
            "You are a retail location analyst. Using ONLY the provided search snippets as "
            "evidence, draft a practical competitor briefing. "
            "Clearly state that footfall and busy-hour claims are inferred from public web "
            "sources and should be verified (e.g. on-site visits, mall analytics, paid data). "
            "Structure the report with markdown headings: "
            "## Executive summary\n"
            "## Competitor landscape\n"
            "## Footfall and peak timing (evidence-based)\n"
            "## Risks and unknowns\n"
            "## Recommended actions\n"
            "Tailor tone and emphasis for the reader role. "
            "When snippets name specific stores or areas, reference them; do not invent venues."
        )
    )
    human = HumanMessage(
        content=(
            f"Reader: {reader}\n"
            f"Location focus: {location}\n"
            f"Store category focus: {store_focus}\n"
            f"Original user request: {user_request}\n\n"
            f"Search evidence:\n{search_context}"
        )
    )
    response = llm.invoke([system, human])
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "\n".join(parts).strip()
    return str(content)
