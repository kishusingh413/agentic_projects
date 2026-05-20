from __future__ import annotations

import logging

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag_qa.core.errors import RetrievalError
from rag_qa.models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self) -> None:
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix: np.ndarray | None = None
        self._chunks: list[Chunk] = []

    def build_index(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise RetrievalError("Cannot build index with zero chunks.")

        try:
            texts: list[str] = [chunk.text for chunk in chunks]
            vectorizer: TfidfVectorizer = TfidfVectorizer(stop_words="english")
            matrix = vectorizer.fit_transform(texts)
            self._vectorizer = vectorizer
            self._matrix = matrix
            self._chunks = chunks
            logger.info("Retriever index ready | chunks=%s", len(chunks))
        except Exception as exc:
            logger.exception("Retriever index build failed | action=build_index")
            raise RetrievalError(f"Failed building retrieval index: {exc}") from exc

    def retrieve(self, question: str, top_k: int) -> list[RetrievedChunk]:
        if self._vectorizer is None or self._matrix is None:
            raise RetrievalError("Retriever index is not initialized.")

        try:
            query_vector = self._vectorizer.transform([question])
            similarity = cosine_similarity(query_vector, self._matrix).flatten()
            if similarity.size == 0:
                return []

            sorted_indices = np.argsort(similarity)[::-1]

            results: list[RetrievedChunk] = []
            seen_locations: set[tuple[str, int | None]] = set()
            seen_text_fingerprints: set[str] = set()
            for idx in sorted_indices:
                score: float = float(similarity[idx])
                if score <= 0:
                    continue
                chunk: Chunk = self._chunks[int(idx)]
                location_key: tuple[str, int | None] = (chunk.source_name, chunk.page_number)
                # Keep only one best chunk per source/page for diversity.
                if location_key in seen_locations:
                    continue
                # Skip near-duplicates created by overlap-heavy chunking.
                text_fingerprint: str = " ".join(chunk.text.lower().split())[:220]
                if text_fingerprint in seen_text_fingerprints:
                    continue

                results.append(
                    RetrievedChunk(
                        chunk=chunk,
                        similarity_score=score,
                    )
                )
                seen_locations.add(location_key)
                seen_text_fingerprints.add(text_fingerprint)
                if len(results) >= top_k:
                    break
            return results
        except Exception as exc:
            logger.exception(
                "Retrieval failed | question=%s | action=retrieve",
                question[:160],
            )
            raise RetrievalError(f"Failed during retrieval: {exc}") from exc
