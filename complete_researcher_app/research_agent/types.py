from __future__ import annotations

from enum import Enum
from typing import Literal


class RouteType(str, Enum):
    LLM = "llm"
    WEB = "web"
    RAG = "rag"


class SourceType(str, Enum):
    WEB = "web"
    RAG = "rag"


RouterMode = Literal["llm", "web", "rag"]

