from __future__ import annotations

import logging
from pathlib import Path

from rag_qa.core.errors import InputValidationError
from rag_qa.models import QAQuery, QAResponse, RetrievedChunk
from rag_qa.services.answer_generator import AnswerGenerator
from rag_qa.services.document_processor import DocumentProcessor
from rag_qa.services.retriever import Retriever

logger = logging.getLogger(__name__)


class QAOrchestrator:
    def __init__(
        self,
        processor: DocumentProcessor,
        retriever: Retriever,
        generator: AnswerGenerator,
    ) -> None:
        self.processor = processor
        self.retriever = retriever
        self.generator = generator

    def build_context_index(self, context_path: Path) -> int:
        chunks = self.processor.load_and_chunk_documents(context_path)
        self.retriever.build_index(chunks)
        logger.info(
            "Context index built | path=%s | chunks=%s",
            str(context_path),
            len(chunks),
        )
        return len(chunks)

    def answer_question(self, query: QAQuery) -> QAResponse:
        if not query.question.strip():
            raise InputValidationError("Question cannot be empty.")

        retrieved: list[RetrievedChunk] = self.retriever.retrieve(
            question=query.question,
            top_k=query.top_k,
        )
        answer: str = self.generator.generate(query.question, retrieved)
        citations: list[str] = self._citations_from_retrieved(retrieved)
        return QAResponse(answer=answer, citations=citations)

    def _citations_from_retrieved(self, retrieved: list[RetrievedChunk]) -> list[str]:
        citations: list[str] = []
        seen_citations: set[str] = set()
        for item in retrieved:
            page_part: str = (
                f"page {item.chunk.page_number}" if item.chunk.page_number else "page n/a"
            )
            citation: str = (
                f"{item.chunk.source_name} ({page_part}, score={item.similarity_score:.3f})"
            )
            if citation in seen_citations:
                continue
            citations.append(citation)
            seen_citations.add(citation)
        return citations
