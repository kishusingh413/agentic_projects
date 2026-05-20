"""CrewAI crew: manager oversight, analysis (with CodeInterpreterTool), then correction."""

from __future__ import annotations

from typing import Any

from crewai import Agent, Crew, Process, Task
from crewai_tools import CodeInterpreterTool

from app.settings import AppSettings


def _build_interpreter_tool(settings: AppSettings) -> CodeInterpreterTool:
    """Configure code execution: Docker when safe; optional unsafe mode for local dev without Docker."""
    return CodeInterpreterTool(unsafe_mode=settings.code_interpreter_unsafe)


def build_debugging_crew(
    settings: AppSettings,
    python_code: str,
) -> Crew:
    """Assemble agents and tasks: sequential process, crew-level planning enabled."""
    llm_model: str = settings.openai_model
    code_interpreter = _build_interpreter_tool(settings)

    manager_agent = Agent(
        role="Debugging Workflow Manager",
        goal=(
            "Oversee the Python debugging pipeline so analysis runs before correction, "
            "risks are noted, and outputs stay structured."
        ),
        backstory=(
            "You are a senior engineering lead who keeps small agent teams aligned. "
            "You do not write fixed code yourself; you coordinate clarity and order."
        ),
        llm=llm_model,
        verbose=True,
        allow_delegation=False,
    )

    analyzer_agent = Agent(
        role="Code Analyzer",
        goal=(
            "Find syntax errors, indentation problems, and logical issues in Python source. "
            "Use the Code Interpreter tool to execute small validation snippets when helpful."
        ),
        backstory=(
            "You are a meticulous Python reviewer. You rely on tooling for facts "
            "(parse errors, tracebacks) and explain each issue clearly."
        ),
        llm=llm_model,
        tools=[code_interpreter],
        verbose=True,
        allow_delegation=False,
    )

    corrector_agent = Agent(
        role="Code Corrector",
        goal=(
            "Produce a corrected version of the Python code that fixes every issue "
            "identified in the analysis while preserving intended behavior."
        ),
        backstory=(
            "You are an expert Python maintainer. You apply minimal, correct fixes "
            "and return complete, runnable code."
        ),
        llm=llm_model,
        verbose=True,
        allow_delegation=False,
    )

    coordination_task = Task(
        description=(
            "Review the debugging workflow for the following Python submission.\n\n"
            f"```{python_code}```\n\n"
            "Confirm that the next steps are: (1) structured error analysis, "
            "(2) correction. Output a short coordination brief: "
            "3–5 bullet points including any risks (e.g. executing untrusted code). "
            "Do not rewrite or fix the code in this step."
        ),
        expected_output=(
            "A concise markdown bullet list coordinating analysis then correction."
        ),
        agent=manager_agent,
    )

    analysis_task = Task(
        description=(
            "Analyze this Python code for syntax, indentation, and logical errors:\n\n"
            f"```{python_code}```\n\n"
            "Use the Code Interpreter when it adds evidence. "
            "When you call it, your snippet must assign a string to the variable `result` "
            "(the tool returns `result`). Example: parse with `ast.parse` inside try/except "
            "and set `result` to the error message or 'syntax ok'.\n\n"
            "Deliver a numbered list of issues. Each item: title, severity "
            "(syntax / indentation / logic), and brief explanation."
        ),
        expected_output=(
            "A numbered list of errors (syntax, indentation, logic) with clear explanations, "
            "referencing tool output where used."
        ),
        agent=analyzer_agent,
        context=[coordination_task],
    )

    correction_task = Task(
        description=(
            "Using the coordination brief and the numbered error list from prior tasks, "
            "produce the full corrected Python program. Fix every listed issue; preserve "
            "the original intent (e.g. Fibonacci behavior). "
            "Return the final code inside a single ```python fenced block."
        ),
        expected_output=(
            "The complete corrected Python source in one markdown ```python code block, "
            "with no extra commentary outside the block."
        ),
        agent=corrector_agent,
        context=[coordination_task, analysis_task],
    )

    return Crew(
        agents=[manager_agent, analyzer_agent, corrector_agent],
        tasks=[coordination_task, analysis_task, correction_task],
        process=Process.sequential,
        planning=True,
        planning_llm=llm_model,
        verbose=True,
    )


def run_debugging_crew(settings: AppSettings, python_code: str) -> Any:
    """Kick off the crew and return CrewAI's output object."""
    crew = build_debugging_crew(settings, python_code)
    return crew.kickoff()
