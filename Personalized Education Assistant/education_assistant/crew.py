"""Three-agent sequential crew: materials → quiz → project ideas."""

import json
import logging
from typing import Any, Final

from crewai import Agent, Crew, Process, Task
from crewai_tools import SerperDevTool

from education_assistant.models import (
    ExpertiseLevel,
    LearningMaterialsOutput,
    ProjectIdeasOutput,
    QuizOutput,
)
from education_assistant.tools.project_suggestion_tool import ProjectSuggestionTool

logger: Final[logging.Logger] = logging.getLogger(__name__)


def build_education_crew() -> Crew:
    """Construct agents, tasks, and a sequential Crew (call after load_config)."""
    serper_tool = SerperDevTool()
    project_tool = ProjectSuggestionTool()

    learning_agent = Agent(
        role="Learning Material Curator",
        goal=(
            "Curate reputable videos, articles, and hands-on exercises that match "
            "the learner's topics, using live web search."
        ),
        backstory=(
            "You are a careful instructional designer. You favor primary sources, "
            "well-known educators, and official documentation. You label resource "
            "types accurately and write concise summaries."
        ),
        tools=[serper_tool],
        verbose=True,
        allow_delegation=False,
    )

    quiz_agent = Agent(
        role="Quiz Creator",
        goal=(
            "Design fair multiple-choice quizzes that assess understanding of the "
            "curated learning materials from the prior step."
        ),
        backstory=(
            "You write clear assessment items with plausible distractors and short "
            "explanations. You never invent facts that contradict the provided materials."
        ),
        tools=[],
        verbose=True,
        allow_delegation=False,
    )

    project_agent = Agent(
        role="Project Idea Strategist",
        goal=(
            "Recommend practical projects suited to the learner's expertise level "
            "and topics, grounded in tool output and prior context."
        ),
        backstory=(
            "You connect classroom concepts to portfolio-worthy work. You call the "
            "project suggestion tool for structured seeds, then refine ideas so they "
            "are specific, feasible, and aligned with the user's level."
        ),
        tools=[project_tool],
        verbose=True,
        allow_delegation=False,
    )

    task_learning = Task(
        description=(
            "Topics of interest: {topics}\n\n"
            "Use the search tool to find up-to-date learning resources. Build a "
            "structured bundle with separate lists for videos, articles, and exercises. "
            "Prefer working URLs and name the publisher or channel when possible."
        ),
        expected_output=(
            "A single object matching LearningMaterialsOutput: topics_summary plus "
            "non-empty lists of videos, articles, and exercises where feasible."
        ),
        agent=learning_agent,
        output_pydantic=LearningMaterialsOutput,
    )

    task_quiz = Task(
        description=(
            "Topics: {topics}\n\n"
            "Using the learning materials from the previous task, create one quiz "
            "with at least five four-option multiple-choice questions. Each question "
            "must test concepts that appear in those materials. Include a brief "
            "explanation for each correct answer."
        ),
        expected_output="An object matching QuizOutput with title and questions[].",
        agent=quiz_agent,
        context=[task_learning],
        output_pydantic=QuizOutput,
    )

    task_projects = Task(
        description=(
            "Learner expertise level: {expertise_level}\n"
            "Topics: {topics}\n\n"
            "First call the tool `project_suggestion_generator` with "
            "expertise_level=\"{expertise_level}\" and topics=\"{topics}\" "
            "to obtain JSON seeds. Then refine and expand into at least three "
            "distinct ProjectIdea entries that fit the learner's level and tie "
            "to the earlier learning materials and quiz themes where helpful."
        ),
        expected_output=(
            "An object matching ProjectIdeasOutput with at least three ProjectIdea "
            "items (title, description, skills_practiced, estimated_duration)."
        ),
        agent=project_agent,
        context=[task_learning, task_quiz],
        output_pydantic=ProjectIdeasOutput,
    )

    return Crew(
        agents=[learning_agent, quiz_agent, project_agent],
        tasks=[task_learning, task_quiz, task_projects],
        process=Process.sequential,
        verbose=True,
    )


def _serialize_task_output(task_output: Any) -> str:
    """Best-effort JSON string for one task result (varies by CrewAI version)."""
    for attr in ("pydantic", "output_pydantic", "structured_output"):
        candidate = getattr(task_output, attr, None)
        if candidate is not None and hasattr(candidate, "model_dump_json"):
            return candidate.model_dump_json(indent=2)
    json_dict = getattr(task_output, "json_dict", None)
    if isinstance(json_dict, dict):
        return json.dumps(json_dict, indent=2, default=str)
    for attr in ("raw", "exported_output", "output"):
        raw = getattr(task_output, attr, None)
        if raw is not None:
            return str(raw)
    return json.dumps({"repr": repr(task_output)}, indent=2)


def run_education_pipeline(topics: str, expertise: ExpertiseLevel) -> str:
    """
    Run the sequential crew and return a human-readable summary of all structured outputs.
    """
    crew = build_education_crew()
    try:
        result = crew.kickoff(
            inputs={
                "topics": topics.strip(),
                "expertise_level": expertise.value,
            }
        )
    except Exception as exc:
        logger.exception(
            "Crew execution failed",
            extra={"topics": topics, "expertise_level": expertise.value},
        )
        raise RuntimeError("Education crew failed; see logs for context.") from exc

    sections: list[str] = []
    tasks_output = getattr(result, "tasks_output", None)
    if tasks_output:
        labels = ("Learning materials", "Quiz", "Project ideas")
        for idx, task_output in enumerate(tasks_output):
            label = labels[idx] if idx < len(labels) else f"Task {idx + 1}"
            sections.append(f"=== {label} ===\n{_serialize_task_output(task_output)}")
    else:
        sections.append(getattr(result, "raw", str(result)))

    return "\n\n".join(sections)
