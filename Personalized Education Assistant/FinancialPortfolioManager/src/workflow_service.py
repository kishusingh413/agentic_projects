"""Orchestrates the multi-agent portfolio workflow (business logic lives here, not in main)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Callable

from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console

from src.agents import build_all_participants, build_model_client
from src.config import AppSettings, get_settings
from src.errors import PortfolioWorkflowError
from src.models import UserPortfolioInput
from src.stateflow import PortfolioStateFlow

logger = logging.getLogger(__name__)


def _make_selector(
    participant_names: frozenset[str],
) -> Callable[[Sequence[BaseAgentEvent | BaseChatMessage]], str | None]:
    def _selector(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
        return PortfolioStateFlow.select_next_speaker(messages, participant_names=participant_names)

    return _selector


async def run_portfolio_workflow(
    portfolio: UserPortfolioInput,
    *,
    settings: AppSettings | None = None,
) -> TaskResult:
    """Run the group chat with StateFlow routing; returns the final ``TaskResult``."""
    resolved = settings if settings is not None else get_settings()
    model_client = build_model_client(resolved)

    participant_names = frozenset(
        {
            "UserProxyAgent",
            "GroupChatManager",
            "PortfolioAnalysisAgent",
            "GrowthInvestmentAgent",
            "ValueInvestmentAgent",
            "InvestmentAdvisorAgent",
        }
    )

    prompt_block = portfolio.to_prompt_block()

    def make_user_input_fn(initial_payload: str):
        sent = False

        def _input_fn(user_prompt: str) -> str:
            nonlocal sent
            if not sent:
                sent = True
                return (
                    "I want to manage my investments and get a structured plan. "
                    "Here are my financial details:\n\n"
                    f"{initial_payload}\n\n"
                    "Please coordinate the team to analyze my portfolio and produce a full report."
                )
            return "Understood. Proceed with the workflow using the details I already provided."

        return _input_fn

    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(max_messages=28)

    team = SelectorGroupChat(
        build_all_participants(model_client, make_user_input_fn(prompt_block)),
        model_client=model_client,
        termination_condition=termination,
        selector_func=_make_selector(participant_names),
        allow_repeated_speaker=True,
    )

    initial_task = (
        "Start the portfolio management session. The UserProxyAgent will provide "
        "salary and holdings next."
    )

    task_result: TaskResult | None = None
    try:
        result = await Console(team.run_stream(task=initial_task))
        if isinstance(result, TaskResult):
            task_result = result
    except Exception as exc:
        logger.exception(
            "Portfolio workflow failed",
            extra={"action": "run_portfolio_workflow", "model": resolved.openai_model},
        )
        raise PortfolioWorkflowError(
            "The portfolio workflow failed during execution.",
            context={"error": str(exc)},
        ) from exc
    finally:
        await model_client.close()

    if task_result is None:
        raise PortfolioWorkflowError(
            "Group chat did not return a TaskResult.",
            context={"action": "run_portfolio_workflow"},
        )

    return task_result
