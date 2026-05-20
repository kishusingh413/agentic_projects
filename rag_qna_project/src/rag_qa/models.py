from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class SupportedExtension(str, Enum):
    TXT = ".txt"
    MD = ".md"
    PDF = ".pdf"


class Chunk(BaseModel):
    chunk_id: str = Field(min_length=1)
    file_path: Path
    source_name: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)
    text: str = Field(min_length=1)


class RetrievedChunk(BaseModel):
    chunk: Chunk
    similarity_score: float = Field(ge=0.0)


class QAQuery(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        normalized_value: str = value.strip()
        if not normalized_value:
            raise ValueError("Question cannot be empty.")
        return normalized_value


class QAResponse(BaseModel):
    answer: str = Field(min_length=1)
    citations: list[str]
