"""
Smart Content Creation (AutoGen + Python)

Two-agent reflection workflow:
1) Content Creator drafts content on Agentic AI.
2) Content Critic evaluates language + technical correctness.
3) Creator revises based on feedback.
4) Reflection summary returns a final refined markdown draft.
"""

from __future__ import annotations

import os
from typing import Any
from importlib.metadata import version as pkg_version

from autogen import AssistantAgent
from dotenv import load_dotenv


def _major_minor(v: str) -> tuple[int, int]:
    parts = v.split(".")
    major = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return major, minor


def validate_runtime_dependencies() -> None:
    """
    Guard against known OpenAI/httpx incompatibility in older AutoGen stacks.

    openai 1.0.0b3 passes `proxies=` to httpx.Client; this breaks on httpx>=0.28.
    """
    openai_ver = pkg_version("openai")
    httpx_ver = pkg_version("httpx")

    if openai_ver == "1.0.0b3" and _major_minor(httpx_ver) >= (0, 28):
        raise RuntimeError(
            "Incompatible dependencies detected: openai==1.0.0b3 with "
            f"httpx=={httpx_ver}. Install a compatible stack with:\n"
            "pip install -r requirements.txt --upgrade --force-reinstall"
        )


def build_llm_config() -> dict[str, Any]:
    """Validate required environment variables and build LLM config."""
    # Load local .env values before validating required settings.
    load_dotenv()

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise EnvironmentError(
            "Missing OPENAI_API_KEY. Add it to .env or export it in your shell "
            "before running this script."
        )

    model = (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()

    return {
        "config_list": [
            {
                "model": model,
                "api_key": api_key,
            }
        ],
        "temperature": 0.4,
    }


def _as_text(reply: Any) -> str:
    """Normalize agent reply payload to plain text."""
    if isinstance(reply, str):
        return reply.strip()
    if isinstance(reply, dict):
        content = reply.get("content")
        return str(content).strip() if content is not None else ""
    return str(reply).strip()


def _build_transcript_markdown(turns: list[dict[str, str]]) -> str:
    chunks: list[str] = []
    for item in turns:
        round_no = item["round"]
        chunks.append(f"### Round {round_no} - Creator Draft\n\n{item['draft']}\n")
        chunks.append(f"### Round {round_no} - Critic Feedback\n\n{item['feedback']}\n")
    return "\n".join(chunks).strip()


def main() -> None:
    validate_runtime_dependencies()
    llm_config = build_llm_config()

    creator = AssistantAgent(
        name="content_creator",
        system_message=(
            "You are the Content Creator Agent. "
            "Your role is to draft and iteratively refine content about Agentic AI. "
            "Write clear, concise, technically accurate prose. "
            "When receiving criticism, revise the content directly and address each point."
        ),
        llm_config=llm_config,
    )

    critic = AssistantAgent(
        name="content_critic",
        system_message=(
            "You are the Content Critic Agent. "
            "Evaluate each draft on: (1) language clarity and structure, "
            "(2) technical correctness and nuance about Agentic AI. "
            "Provide constructive feedback with concrete, actionable improvements. "
            "When the draft is strong, clearly say it is publication-ready."
        ),
        llm_config=llm_config,
    )

    max_rounds = 5
    creator_input = (
        "Topic: Agentic AI.\n"
        "Create the initial draft in markdown for a reader with basic AI knowledge. "
        "Keep it clear, concise, and technically accurate."
    )
    transcript: list[dict[str, str]] = []

    for idx in range(max_rounds):
        round_no = str(idx + 1)
        draft_reply = creator.generate_reply(
            messages=[{"role": "user", "content": creator_input}]
        )
        draft_text = _as_text(draft_reply)

        feedback_reply = critic.generate_reply(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Evaluate this draft on language clarity and technical correctness. "
                        "Provide concise, constructive feedback.\n\n"
                        f"{draft_text}"
                    ),
                }
            ]
        )
        feedback_text = _as_text(feedback_reply)

        transcript.append(
            {"round": round_no, "draft": draft_text, "feedback": feedback_text}
        )

        if idx == max_rounds - 1:
            break

        creator_input = (
            "Revise the previous draft using this critic feedback. "
            "Address every major issue and improve clarity and correctness.\n\n"
            f"Feedback:\n{feedback_text}\n\n"
            f"Current draft:\n{draft_text}"
        )

    reflection_prompt = (
        "You are summarizing a reflection-based content workflow. "
        "Given the transcript below, return ONLY the final refined content in markdown. "
        "Do not include analysis, labels, or bullet-point feedback. "
        "Include a title and 3-5 short sections.\n\n"
        f"{_build_transcript_markdown(transcript)}"
    )
    summary_reply = creator.generate_reply(
        messages=[{"role": "user", "content": reflection_prompt}]
    )
    final_markdown = _as_text(summary_reply)

    print("# Final Refined Draft (Markdown)\n")
    if final_markdown:
        print(final_markdown)
    else:
        print(
            "No reflection summary was generated. "
            "Check model/provider compatibility for reflection_with_llm."
        )


if __name__ == "__main__":
    main()
