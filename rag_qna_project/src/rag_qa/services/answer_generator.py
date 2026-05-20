from __future__ import annotations

import logging

from openai import OpenAI

from rag_qa.core.errors import GenerationError
from rag_qa.models import RetrievedChunk

logger = logging.getLogger(__name__)


class AnswerGenerator:
    def __init__(
        self,
        llm_mode: str,
        model_name: str,
        openai_api_key: str | None,
    ) -> None:
        self.llm_mode = llm_mode
        self.model_name = model_name
        self.openai_client: OpenAI | None = None
        if llm_mode == "openai" and openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)

    def generate(self, question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
        if not retrieved_chunks:
            return "I could not find relevant context for this question."

        if self.llm_mode == "openai":
            return self._generate_with_openai(question, retrieved_chunks)
        return self._generate_extractive(question, retrieved_chunks)

    def _generate_with_openai(
        self, question: str, retrieved_chunks: list[RetrievedChunk]
    ) -> str:
        if self.openai_client is None:
            raise GenerationError("OpenAI client is not configured.")

        context_block: str = self._format_context(retrieved_chunks)
        prompt: str = (
            "Answer the user question using ONLY the provided context.\n"
            "If information is missing, clearly state uncertainty.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context_block}\n\n"
            "Answer:"
        )
        try:
            response = self.openai_client.responses.create(
                model=self.model_name,
                input=prompt,
                temperature=0.2,
            )
            answer_text: str = response.output_text.strip()
            if not answer_text:
                raise GenerationError("The model returned an empty answer.")
            return answer_text
        except Exception as exc:
            logger.exception(
                "LLM generation failed | model=%s | action=generate_openai",
                self.model_name,
            )
            raise GenerationError(f"OpenAI answer generation failed: {exc}") from exc

    def _generate_extractive(
        self, question: str, retrieved_chunks: list[RetrievedChunk]
    ) -> str:
        top_chunk: RetrievedChunk = retrieved_chunks[0]
        first_text: str = top_chunk.chunk.text.strip()
        preview: str = first_text[:700].strip()
        return (
            f"Based on the most relevant source, here is a context-grounded answer:\n{preview}"
        )

    def _format_context(self, retrieved_chunks: list[RetrievedChunk]) -> str:
        lines: list[str] = []
        for index, item in enumerate(retrieved_chunks, start=1):
            page_text: str = (
                f"page {item.chunk.page_number}" if item.chunk.page_number else "page n/a"
            )
            lines.append(
                f"[{index}] source={item.chunk.source_name}, {page_text}, "
                f"score={item.similarity_score:.3f}\n{item.chunk.text}\n"
            )
        return "\n".join(lines)
