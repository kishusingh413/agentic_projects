"""Pydantic models for structured crew outputs."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator


class ExpertiseLevel(StrEnum):
    """Fixed expertise tiers for project suggestions and prompts."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class ResourceType(StrEnum):
    """Kind of learning resource."""

    VIDEO = "video"
    ARTICLE = "article"
    EXERCISE = "exercise"


class LearningResource(BaseModel):
    """A single curated learning item."""

    title: str = Field(description="Clear title of the resource")
    resource_type: ResourceType = Field(description="video, article, or exercise")
    url: str | None = Field(
        default=None,
        description="Direct link when available from search; null if not known",
    )
    summary: str = Field(description="One or two sentences on what the learner gains")
    source_name: str | None = Field(
        default=None, description="Publisher, channel, or site name"
    )


class LearningMaterialsOutput(BaseModel):
    """Curated learning materials for the user's topics."""

    topics_summary: str = Field(
        description="Brief restatement of the topics covered by this bundle"
    )
    videos: list[LearningResource] = Field(
        default_factory=list,
        description="Video tutorials or lectures (aim for at least 2 when possible)",
    )
    articles: list[LearningResource] = Field(
        default_factory=list,
        description="Articles, docs, or blog posts",
    )
    exercises: list[LearningResource] = Field(
        default_factory=list,
        description="Hands-on exercises, labs, or practice problem sets",
    )


class QuizQuestion(BaseModel):
    """One multiple-choice question."""

    question: str = Field(description="The question text")
    options: list[str] = Field(
        description="Exactly four plausible options; one must be correct"
    )
    correct_answer_index: int = Field(
        ge=0,
        description="Zero-based index into options for the correct answer",
    )
    explanation: str = Field(
        description="Short explanation of why the correct answer is right"
    )

    @field_validator("options")
    @classmethod
    def four_options(cls, v: list[str]) -> list[str]:
        if len(v) != 4:
            raise ValueError("Quiz must have exactly four options per question")
        return v

    @model_validator(mode="after")
    def correct_index_in_range(self) -> QuizQuestion:
        if self.correct_answer_index >= len(self.options):
            raise ValueError("correct_answer_index must be within options length")
        return self


class QuizOutput(BaseModel):
    """Personalized quiz aligned to the learning materials."""

    title: str = Field(description="Short quiz title")
    questions: list[QuizQuestion] = Field(
        description="At least 5 questions testing concepts from the materials"
    )

    @field_validator("questions")
    @classmethod
    def min_questions(cls, v: list[QuizQuestion]) -> list[QuizQuestion]:
        if len(v) < 5:
            raise ValueError("Quiz must contain at least 5 questions")
        return v


class ProjectIdea(BaseModel):
    """One practical project suggestion."""

    title: str = Field(description="Project name")
    description: str = Field(
        description="What to build, main features, and how it ties to the topics"
    )
    skills_practiced: list[str] = Field(
        description="Concrete skills the learner will practice"
    )
    estimated_duration: str = Field(
        description="Rough time estimate, e.g. '4–8 hours' or '1–2 weeks part-time'"
    )


class ProjectIdeasOutput(BaseModel):
    """Set of project ideas for the user's level and interests."""

    ideas: list[ProjectIdea] = Field(
        description="At least 3 distinct, actionable project ideas"
    )

    @field_validator("ideas")
    @classmethod
    def min_ideas(cls, v: list[ProjectIdea]) -> list[ProjectIdea]:
        if len(v) < 3:
            raise ValueError("Provide at least 3 project ideas")
        return v
