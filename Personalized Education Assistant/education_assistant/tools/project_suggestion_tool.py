"""Custom tool: seed project ideas by expertise level and topics."""

import json
import logging
from typing import Literal, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from education_assistant.models import ExpertiseLevel

logger = logging.getLogger(__name__)


class ProjectSuggestionToolArgs(BaseModel):
    """Inputs validated before _run."""

    expertise_level: Literal["beginner", "intermediate", "advanced"] = Field(
        description="Learner level: beginner, intermediate, or advanced"
    )
    topics: str = Field(
        description="Comma-separated topics of interest, e.g. 'Python, APIs, testing'"
    )


class ProjectSuggestionTool(BaseTool):
    """
    Generates structured project idea seeds tailored to expertise and topics.
    Agents should refine and align these with web context and user goals.
    """

    name: str = "project_suggestion_generator"
    description: str = (
        "Produces a JSON list of practical project idea seeds for the given "
        "expertise_level (beginner, intermediate, advanced) and topics string. "
        "Call this when you need grounded starting points before finalizing "
        "structured project recommendations."
    )
    args_schema: Type[BaseModel] = ProjectSuggestionToolArgs

    def _run(self, expertise_level: str, topics: str) -> str:
        try:
            level = ExpertiseLevel(expertise_level.lower().strip())
        except ValueError:
            logger.warning(
                "Unknown expertise_level in tool; defaulting to beginner",
                extra={"expertise_level": expertise_level},
            )
            level = ExpertiseLevel.BEGINNER

        topic_parts = [t.strip() for t in topics.split(",") if t.strip()]
        if not topic_parts:
            topic_parts = ["the learner's topics"]

        primary = topic_parts[0]
        rest = ", ".join(topic_parts[1:]) if len(topic_parts) > 1 else ""

        ideas: list[dict[str, object]] = []

        if level is ExpertiseLevel.BEGINNER:
            ideas = [
                {
                    "title": f"{primary.title()} glossary and flashcards",
                    "description": (
                        f"Build a small CLI or web app that stores terms for {primary}"
                        + (f" related to {rest}" if rest else "")
                        + ", with quiz mode to reinforce vocabulary."
                    ),
                    "skills_practiced": ["basic syntax", "data structures", "user input"],
                    "estimated_duration": "4–10 hours",
                },
                {
                    "title": f"Personal notes dashboard for {primary}",
                    "description": (
                        "Create a simple page or script to save daily notes, "
                        "tagged by subtopic, with search."
                    ),
                    "skills_practiced": ["CRUD basics", "file or simple DB I/O"],
                    "estimated_duration": "1–2 days",
                },
                {
                    "title": f"{primary.title()} concept explainer",
                    "description": (
                        "Pick one core concept and build a step-by-step tutorial "
                        "site or notebook with examples and a mini exercise."
                    ),
                    "skills_practiced": ["technical writing", "examples", "testing ideas"],
                    "estimated_duration": "1–3 days",
                },
            ]
        elif level is ExpertiseLevel.INTERMEDIATE:
            ideas = [
                {
                    "title": f"REST API service for {primary} resources",
                    "description": (
                        f"Design CRUD endpoints, validation, and basic auth or API keys; "
                        f"include integration tests. Extend with {rest or 'related subdomains'}."
                    ),
                    "skills_practiced": ["REST design", "validation", "automated tests"],
                    "estimated_duration": "3–7 days",
                },
                {
                    "title": f"Data pipeline for {primary} datasets",
                    "description": (
                        "Ingest CSV/API data, clean, aggregate, and expose summaries "
                        "via a small dashboard or scheduled report."
                    ),
                    "skills_practiced": ["ETL basics", "scheduling", "visualization"],
                    "estimated_duration": "4–10 days",
                },
                {
                    "title": f"Collaboration tool mini-feature in {primary}",
                    "description": (
                        "Implement one real feature: comments, shared lists, or "
                        "notifications— with persistence and error handling."
                    ),
                    "skills_practiced": ["state management", "API integration", "UX"],
                    "estimated_duration": "1–2 weeks part-time",
                },
            ]
        else:
            ideas = [
                {
                    "title": f"Production-style platform module for {primary}",
                    "description": (
                        "Design for scale: caching, observability, feature flags, "
                        "and deployment notes; include load or chaos notes where relevant."
                    ),
                    "skills_practiced": ["architecture", "SRE practices", "deployment"],
                    "estimated_duration": "2–4 weeks",
                },
                {
                    "title": f"Advanced automation around {primary}",
                    "description": (
                        "End-to-end automation (CI, infra-as-code snippet, or bots) "
                        "with rollback strategy and security review checklist."
                    ),
                    "skills_practiced": ["CI/CD", "security mindset", "automation"],
                    "estimated_duration": "2–3 weeks",
                },
                {
                    "title": f"Research-grade prototype in {primary}",
                    "description": (
                        "Benchmark alternatives, document tradeoffs, ship a minimal "
                        "vertical slice with metrics and a written ADR."
                    ),
                    "skills_practiced": ["benchmarking", "technical decision records"],
                    "estimated_duration": "2–5 weeks",
                },
            ]

        payload = {"expertise_level": level.value, "topics": topic_parts, "ideas": ideas}
        return json.dumps(payload, indent=2)
