from __future__ import annotations

from functools import lru_cache
import re
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from research_agent.config import get_settings
from research_agent.errors import KnowledgeBaseError
from research_agent.logging_utils import log_exception
from research_agent.schemas import RagFinding
from research_agent.services.llm_service import get_embedding_model


@lru_cache(maxsize=1)
def _get_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)


_VECTORSTORE: Optional[Chroma] = None


def _read_knowledge_base_files(kb_dir: Path) -> list[Document]:
    docs: list[Document] = []
    txt_files = sorted(kb_dir.glob("**/*.txt"))
    if not txt_files:
        raise KnowledgeBaseError(
            "Knowledge base directory is empty or has no .txt files",
            details={"knowledge_base_dir": str(kb_dir)},
        )

    for path in txt_files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")

        if not text.strip():
            continue

        doc_id = path.stem
        docs.append(
            Document(
                page_content=text,
                metadata={"source_path": str(path), "doc_id": doc_id},
            )
        )
    return docs


def _ensure_vectorstore_built(vs: Chroma, docs: list[Document], persist_marker: Path) -> Chroma:
    if persist_marker.exists():
        return vs

    splitter = _get_text_splitter()
    splits = splitter.split_documents(docs)
    if not splits:
        raise KnowledgeBaseError("Failed to split knowledge base documents")

    # Add documents to existing collection (persisted)
    vs.add_documents(splits)
    persist_marker.write_text("ok", encoding="utf-8")
    return vs


def _get_vectorstore() -> Chroma:
    global _VECTORSTORE
    if _VECTORSTORE is not None:
        return _VECTORSTORE

    settings = get_settings()
    kb_dir = Path(settings.knowledge_base_dir)
    persist_dir = Path(settings.chroma_persist_dir)
    persist_marker = persist_dir / ".kb_index_built"

    try:
        embedding = get_embedding_model()
        # Load (or create) a persisted Chroma collection.
        vs = Chroma(
            collection_name="knowledge_base",
            embedding_function=embedding,
            persist_directory=str(persist_dir),
        )

        docs = _read_knowledge_base_files(kb_dir)
        vs = _ensure_vectorstore_built(vs, docs, persist_marker)
        _VECTORSTORE = vs
        return vs
    except KnowledgeBaseError:
        raise
    except Exception as exc:  # noqa: BLE001
        log_exception("rag_vectorstore_init", exc, extra={"knowledge_base_dir": str(kb_dir)})
        raise KnowledgeBaseError("Failed to initialize vector store") from exc


def rag_retrieve(query: str, *, top_k: int) -> list[RagFinding]:
    if not query.strip():
        raise KnowledgeBaseError("Empty query passed to rag_retrieve")

    settings = get_settings()
    # Offline fallback when no OpenAI key is set: do lightweight keyword retrieval.
    if settings.openai_api_key is None:
        return _rag_retrieve_offline(
            query,
            top_k=top_k,
            kb_dir=Path(settings.knowledge_base_dir),
        )

    vs = _get_vectorstore()

    try:
        results = vs.similarity_search_with_relevance_scores(query, k=top_k)
        findings: list[RagFinding] = []

        for doc, score in results:
            md = doc.metadata or {}
            source_path = str(md.get("source_path", "unknown"))
            doc_id = str(md.get("doc_id", "unknown"))
            text = doc.page_content.strip()
            if not text:
                continue

            findings.append(
                RagFinding(
                    source_path=source_path,
                    doc_id=doc_id,
                    text=text[:2500],
                    score=float(score),
                )
            )
        return findings
    except Exception as exc:  # noqa: BLE001
        log_exception("rag_retrieve", exc, extra={"query": query, "top_k": str(top_k)})
        raise KnowledgeBaseError("RAG retrieval failed") from exc


def _rag_retrieve_offline(query: str, *, top_k: int, kb_dir: Path) -> list[RagFinding]:
    kb_dir = kb_dir
    txt_files = sorted(kb_dir.glob("**/*.txt"))
    if not txt_files:
        raise KnowledgeBaseError(
            "Knowledge base directory is empty or has no .txt files",
            details={"knowledge_base_dir": str(kb_dir)},
        )

    q = query.lower()
    q_terms = [t for t in re.split(r"[^a-z0-9]+", q) if t]
    if not q_terms:
        raise KnowledgeBaseError("Failed to tokenize query for offline retrieval")

    scored: list[tuple[float, Path]] = []
    for path in txt_files:
        try:
            text = path.read_text(encoding="utf-8").lower()
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1").lower()

        score = 0.0
        for term in q_terms:
            if term in text:
                # Simple presence scoring
                score += 1.0
        if score > 0:
            scored.append((score, path))

    # If nothing matches terms, fall back to first files as best-effort.
    if not scored:
        scored = [(0.0, p) for p in txt_files[:top_k]]

    scored.sort(key=lambda x: x[0], reverse=True)
    findings: list[RagFinding] = []
    for score, path in scored[:top_k]:
        doc_id = path.stem
        try:
            raw = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw = path.read_text(encoding="latin-1")
        excerpt = raw.strip()
        if len(excerpt) > 2500:
            excerpt = excerpt[:2500]
        findings.append(
            RagFinding(
                source_path=str(path),
                doc_id=doc_id,
                text=excerpt,
                score=score,
            )
        )
    return findings

