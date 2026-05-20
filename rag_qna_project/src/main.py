from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pydantic import ValidationError

from rag_qa.core.errors import (
    ConfigurationError,
    DocumentProcessingError,
    GenerationError,
    InputValidationError,
    RetrievalError,
)
from rag_qa.core.logging_config import configure_logging
from rag_qa.core.settings import export_settings_debug, load_settings
from rag_qa.models import QAQuery
from rag_qa.services.answer_generator import AnswerGenerator
from rag_qa.services.document_processor import DocumentProcessor
from rag_qa.services.qa_orchestrator import QAOrchestrator
from rag_qa.services.retriever import Retriever

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG QA CLI")
    parser.add_argument(
        "--context-path",
        required=True,
        help="Path to a context file or directory (.txt, .md, .pdf).",
    )
    parser.add_argument(
        "--top-k",
        default=4,
        type=int,
        help="How many chunks to retrieve for each question (1-10).",
    )
    return parser.parse_args()


def run() -> int:
    configure_logging()
    args = parse_args()

    try:
        settings = load_settings()
        logger.info("App config | details=%s", export_settings_debug(settings))

        processor = DocumentProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        retriever = Retriever()
        generator = AnswerGenerator(
            llm_mode=settings.llm_mode,
            model_name=settings.model_name,
            openai_api_key=settings.openai_api_key,
        )
        orchestrator = QAOrchestrator(processor, retriever, generator)

        context_path = Path(args.context_path).expanduser().resolve()
        chunk_count: int = orchestrator.build_context_index(context_path)
        print(f"Index built with {chunk_count} chunks.")
        print("Type your question. Enter 'exit' to quit.\n")

        while True:
            question: str = input("Q> ").strip()
            if question.lower() in {"exit", "quit"}:
                print("Goodbye.")
                break
            if not question:
                print("Please enter a non-empty question.")
                continue

            try:
                query = QAQuery(question=question, top_k=args.top_k)
                response = orchestrator.answer_question(query)
                print("\nAnswer:\n")
                print(response.answer)
                print("\nSources:")
                if response.citations:
                    for source in response.citations:
                        print(f"- {source}")
                else:
                    print("- No sources matched this question.")
                print("")
            except ValidationError as exc:
                print(f"Input validation error: {exc}")
            except (InputValidationError, RetrievalError, GenerationError) as exc:
                logger.error("Question handling failed | question=%s | error=%s", question, str(exc))
                print(f"Error: {exc}")

        return 0
    except (ConfigurationError, DocumentProcessingError, InputValidationError, RetrievalError) as exc:
        logger.error("Startup failed | error=%s", str(exc))
        print(f"Startup error: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 0
    except Exception as exc:
        logger.exception("Unhandled application error | action=run")
        print(f"Unexpected error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(run())
