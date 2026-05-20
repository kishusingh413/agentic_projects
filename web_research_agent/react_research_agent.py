from __future__ import annotations

import argparse
import json
import os
import textwrap
from dataclasses import dataclass
from enum import Enum

import requests
from tavily import TavilyClient
from dotenv import load_dotenv


class LLMProvider(str, Enum):
    OPENAI = "openai"
    GROQ = "groq"
    OLLAMA = "ollama"


@dataclass(frozen=True)
class AppConfig:
    llm_provider: LLMProvider
    llm_model: str
    tavily_api_key: str
    openai_api_key: str | None
    groq_api_key: str | None
    ollama_host: str | None


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    content: str


def load_and_validate_config() -> AppConfig:
    provider_raw: str = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    model: str = os.getenv("LLM_MODEL", "").strip()
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "").strip()
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    ollama_host: str | None = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    if not tavily_api_key:
        raise ValueError("Missing required environment variable: TAVILY_API_KEY")

    try:
        llm_provider: LLMProvider = LLMProvider(provider_raw)
    except ValueError as err:
        allowed_values: str = ", ".join(provider.value for provider in LLMProvider)
        raise ValueError(
            f"Invalid LLM_PROVIDER '{provider_raw}'. Allowed values: {allowed_values}"
        ) from err

    if llm_provider == LLMProvider.OPENAI:
        if not model:
            model = "gpt-4o-mini"
        if not openai_api_key or not openai_api_key.strip():
            raise ValueError(
                "OPENAI_API_KEY is required when LLM_PROVIDER is 'openai'."
            )
    elif llm_provider == LLMProvider.GROQ:
        if not model:
            model = "llama-3.1-8b-instant"
        if not groq_api_key or not groq_api_key.strip():
            raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER is 'groq'.")
    else:
        if not model:
            model = "llama3.1"
        if not ollama_host or not ollama_host.strip():
            raise ValueError("OLLAMA_HOST is required when LLM_PROVIDER is 'ollama'.")

    return AppConfig(
        llm_provider=llm_provider,
        llm_model=model,
        tavily_api_key=tavily_api_key,
        openai_api_key=openai_api_key.strip() if openai_api_key else None,
        groq_api_key=groq_api_key.strip() if groq_api_key else None,
        ollama_host=ollama_host.strip() if ollama_host else None,
    )


class ReActResearchAgent:
    def __init__(self, config: AppConfig, max_questions: int = 6, max_results: int = 3):
        self.config: AppConfig = config
        self.max_questions: int = max_questions
        self.max_results: int = max_results
        self.tavily_client: TavilyClient = TavilyClient(api_key=config.tavily_api_key)

    def run(self, topic: str) -> str:
        normalized_topic: str = topic.strip()
        if not normalized_topic:
            raise ValueError("Topic must be a non-empty string.")

        print(f"[Planning] Generating research questions for: {normalized_topic}")
        questions: list[str] = self.generate_research_questions(normalized_topic)
        if not questions:
            raise RuntimeError("The LLM did not return valid research questions.")

        print(f"[Acting] Searching the web for {len(questions)} questions...")
        findings: dict[str, list[SearchResult]] = {}
        for question in questions:
            findings[question] = self.search_web(question)

        print("[Synthesis] Compiling final report...")
        return self.compile_report(normalized_topic, questions, findings)

    def generate_research_questions(self, topic: str) -> list[str]:
        planning_prompt: str = textwrap.dedent(
            f"""
            You are a research planning assistant.
            Generate 5 to 6 concise, non-overlapping research questions about: "{topic}".

            Requirements:
            - Questions must cover different aspects (definition/background, trends/data, challenges, solutions/policies, future outlook).
            - Return ONLY valid JSON.
            - JSON format:
              {{
                "questions": ["question 1", "question 2", "..."]
              }}
            """
        ).strip()

        llm_raw: str = self._call_llm(planning_prompt)
        parsed_questions: list[str] = self._parse_questions_response(llm_raw)
        return parsed_questions[: self.max_questions]

    def search_web(self, query: str) -> list[SearchResult]:
        try:
            response: dict[str, object] = self.tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=self.max_results,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Tavily search failed for query='{query}'. Error: {exc}"
            ) from exc

        raw_results: object = response.get("results", [])
        if not isinstance(raw_results, list):
            return []

        results: list[SearchResult] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            title_obj: object = item.get("title")
            url_obj: object = item.get("url")
            content_obj: object = item.get("content")
            title: str = title_obj.strip() if isinstance(title_obj, str) else "Untitled"
            url: str = url_obj.strip() if isinstance(url_obj, str) else ""
            content: str = content_obj.strip() if isinstance(content_obj, str) else ""
            if not content:
                continue
            results.append(SearchResult(title=title, url=url, content=content))
        return results

    def compile_report(
        self, topic: str, questions: list[str], findings: dict[str, list[SearchResult]]
    ) -> str:
        lines: list[str] = []
        lines.append(f"# Web Research Report: {topic}")
        lines.append("")
        lines.append("## Introduction")
        lines.append(
            f"This report was generated by a ReAct-style agent. The agent planned "
            f"{len(questions)} guiding research questions, searched the web using Tavily, "
            "and synthesized the collected evidence into structured findings."
        )
        lines.append("")

        for index, question in enumerate(questions, start=1):
            lines.append(f"## {index}. {question}")
            question_results: list[SearchResult] = findings.get(question, [])
            if not question_results:
                lines.append("- No reliable search results were returned for this question.")
                lines.append("")
                continue

            for result in question_results:
                summary: str = result.content.replace("\n", " ").strip()
                trimmed_summary: str = (
                    summary[:450].rstrip() + "..." if len(summary) > 450 else summary
                )
                source_label: str = result.url if result.url else "URL unavailable"
                lines.append(f"- **Source:** {result.title} ({source_label})")
                lines.append(f"  - **Key Point:** {trimmed_summary}")
            lines.append("")

        lines.append("## Conclusion")
        lines.append(
            "The topic was analyzed through multiple focused questions that covered context, "
            "evidence, constraints, and future direction. The cited sources provide a "
            "foundation for deeper follow-up analysis, validation, and decision-making."
        )
        lines.append("")
        return "\n".join(lines)

    def _call_llm(self, prompt: str) -> str:
        if self.config.llm_provider == LLMProvider.OLLAMA:
            return self._call_ollama(prompt)
        if self.config.llm_provider == LLMProvider.GROQ:
            return self._call_openai_compatible(
                prompt=prompt,
                api_key=self.config.groq_api_key,
                base_url="https://api.groq.com/openai/v1",
                model=self.config.llm_model,
            )
        return self._call_openai_compatible(
            prompt=prompt,
            api_key=self.config.openai_api_key,
            base_url=None,
            model=self.config.llm_model,
        )

    def _call_openai_compatible(
        self, prompt: str, api_key: str | None, base_url: str | None, model: str
    ) -> str:
        if not api_key:
            raise ValueError("Missing API key for OpenAI-compatible provider.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package is not installed. Run: pip install openai"
            ) from exc

        client: OpenAI = OpenAI(api_key=api_key, base_url=base_url)
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise research planning assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
        except Exception as exc:
            raise RuntimeError(
                f"LLM call failed for provider={self.config.llm_provider.value}, model={model}. Error: {exc}"
            ) from exc

        message_content: str | None = completion.choices[0].message.content
        if not message_content or not message_content.strip():
            raise RuntimeError("LLM returned an empty response.")
        return message_content.strip()

    def _call_ollama(self, prompt: str) -> str:
        if not self.config.ollama_host:
            raise ValueError("OLLAMA_HOST is required for the Ollama provider.")
        payload: dict[str, object] = {
            "model": self.config.llm_model,
            "prompt": prompt,
            "stream": False,
        }
        endpoint: str = f"{self.config.ollama_host.rstrip('/')}/api/generate"
        try:
            response = requests.post(
                endpoint,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Ollama request failed for endpoint={endpoint}, model={self.config.llm_model}. Error: {exc}"
            ) from exc
        response_json: dict[str, object] = response.json()
        generated_text: object = response_json.get("response")
        if not isinstance(generated_text, str) or not generated_text.strip():
            raise RuntimeError("Ollama returned an empty response body.")
        return generated_text.strip()

    def _parse_questions_response(self, llm_response: str) -> list[str]:
        raw_text: str = llm_response.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            raw_text = raw_text.replace("json", "", 1).strip()

        try:
            parsed: object = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Failed to parse LLM planning output as JSON. "
                f"Raw output: {llm_response}"
            ) from exc

        if not isinstance(parsed, dict):
            raise RuntimeError("LLM planning output must be a JSON object.")
        questions_obj: object = parsed.get("questions")
        if not isinstance(questions_obj, list):
            raise RuntimeError(
                "LLM planning output missing 'questions' list in JSON response."
            )

        questions: list[str] = []
        for item in questions_obj:
            if not isinstance(item, str):
                continue
            cleaned: str = item.strip()
            if cleaned:
                questions.append(cleaned)

        if len(questions) < 3:
            raise RuntimeError(
                f"Expected 5-6 questions from the LLM, but got only {len(questions)} valid items."
            )
        return questions


def build_parser() -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="ReAct-based web research agent (LLM planning + Tavily acting)."
    )
    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help="Research topic (for example: 'Climate Change').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="report.md",
        help="Output markdown report file path.",
    )
    return parser


def main() -> None:
    load_dotenv()
    parser: argparse.ArgumentParser = build_parser()
    args = parser.parse_args()
    topic: str = args.topic.strip()
    output_path: str = args.output.strip()

    if not topic:
        raise ValueError("Topic must be a non-empty string.")
    if not output_path:
        raise ValueError("Output path must be a non-empty string.")

    config: AppConfig = load_and_validate_config()
    agent = ReActResearchAgent(config=config, max_questions=6, max_results=3)
    report_markdown: str = agent.run(topic)

    with open(output_path, "w", encoding="utf-8") as report_file:
        report_file.write(report_markdown)
    print(f"Report generated: {output_path}")


if __name__ == "__main__":
    main()
