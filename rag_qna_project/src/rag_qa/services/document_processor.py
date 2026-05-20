from __future__ import annotations

import logging
from pathlib import Path

from pypdf import PdfReader

from rag_qa.core.errors import DocumentProcessingError, InputValidationError
from rag_qa.models import Chunk, SupportedExtension

logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_and_chunk_documents(self, input_path: Path) -> list[Chunk]:
        if not input_path.exists():
            raise InputValidationError(f"Input path does not exist: {input_path}")

        files: list[Path] = self._resolve_supported_files(input_path)
        if not files:
            raise InputValidationError(
                "No supported files found. Provide .txt, .md, or .pdf files."
            )

        all_chunks: list[Chunk] = []
        for file_path in files:
            try:
                if file_path.suffix.lower() == SupportedExtension.PDF.value:
                    all_chunks.extend(self._chunk_pdf(file_path))
                else:
                    all_chunks.extend(self._chunk_text_file(file_path))
            except Exception as exc:
                logger.exception(
                    "Document processing failed | file=%s | action=load_and_chunk",
                    str(file_path),
                )
                raise DocumentProcessingError(
                    f"Failed processing file '{file_path.name}': {exc}"
                ) from exc
        return all_chunks

    def _resolve_supported_files(self, input_path: Path) -> list[Path]:
        if input_path.is_file():
            return [input_path] if self._is_supported(input_path) else []
        return [
            path
            for path in sorted(input_path.rglob("*"))
            if path.is_file() and self._is_supported(path)
        ]

    def _is_supported(self, file_path: Path) -> bool:
        suffix: str = file_path.suffix.lower()
        return suffix in {item.value for item in SupportedExtension}

    def _chunk_text_file(self, file_path: Path) -> list[Chunk]:
        full_text: str = file_path.read_text(encoding="utf-8", errors="ignore")
        return self._slice_text_into_chunks(full_text, file_path=file_path, page_number=None)

    def _chunk_pdf(self, file_path: Path) -> list[Chunk]:
        reader: PdfReader = PdfReader(str(file_path))
        chunks: list[Chunk] = []
        for page_index, page in enumerate(reader.pages, start=1):
            page_text: str = (page.extract_text() or "").strip()
            if not page_text:
                continue
            chunks.extend(
                self._slice_text_into_chunks(
                    page_text,
                    file_path=file_path,
                    page_number=page_index,
                )
            )
        return chunks

    def _slice_text_into_chunks(
        self,
        text: str,
        file_path: Path,
        page_number: int | None,
    ) -> list[Chunk]:
        normalized_text: str = text.strip()
        if not normalized_text:
            return []

        chunks: list[Chunk] = []
        start: int = 0
        chunk_counter: int = 1

        while start < len(normalized_text):
            end: int = min(start + self.chunk_size, len(normalized_text))
            chunk_text: str = normalized_text[start:end].strip()
            if chunk_text:
                chunk_id: str = f"{file_path.name}-{page_number or 0}-{chunk_counter}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        file_path=file_path.resolve(),
                        source_name=file_path.name,
                        page_number=page_number,
                        text=chunk_text,
                    )
                )
                chunk_counter += 1
            if end == len(normalized_text):
                break
            start = max(0, end - self.chunk_overlap)

        return chunks
