"""Group chat orchestration: User Proxy -> Bill Processing -> Expense Summarization."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from bill_management_agent.models import CategorizedExpenses
from bill_management_agent.services.bill_extraction import extract_categorized_expenses
from bill_management_agent.services.expense_summarization import summarize_expenses

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentTurn:
    """One message in the group conversation."""

    agent_name: str
    role: str
    content: str


class UserProxyAgent:
    """Initiates the workflow and hands the bill image to the group."""

    name = "User_Proxy"

    def __init__(self, role: str) -> None:
        self.role = role

    def build_message(self, image_path: Path) -> str:
        return (
            f"[{self.name}] {self.role}\n"
            f"Request to Group Manager: please process this bill image: {image_path}"
        )


class BillProcessingAgent:
    """Extracts line items from the image and assigns categories."""

    name = "Bill_Processing"

    def __init__(self, role: str) -> None:
        self.role = role

    def process(
        self,
        image_path: Path,
        *,
        allow_packaged_demo: bool = False,
    ) -> tuple[str, CategorizedExpenses]:
        data = extract_categorized_expenses(
            image_path,
            allow_packaged_demo=allow_packaged_demo,
        )
        body = (
            f"[{self.name}] {self.role}\n"
            f"Categorized expenses (structured JSON):\n{data.model_dump_json(indent=2)}"
        )
        return body, data


class ExpenseSummarizationAgent:
    """Aggregates totals per category and highlights spending patterns."""

    name = "Expense_Summarization"

    def __init__(self, role: str) -> None:
        self.role = role

    def summarize(self, data: CategorizedExpenses) -> str:
        summary = summarize_expenses(data)
        insight_lines = "\n".join(f"- {line}" for line in summary.insights)
        return (
            f"[{self.name}] {self.role}\n"
            f"Summary (JSON):\n{summary.model_dump_json(indent=2)}\n\n"
            f"Insights:\n{insight_lines}\n\n"
            f"Highest category: {summary.highest_category.value} "
            f"({summary.highest_amount} {data.currency}). "
            f"Grand total: {summary.grand_total} {data.currency}."
        )


class GroupChatManager:
    """
    Coordinates agents in a fixed sequential order (group chat with explicit handoff).

    Flow: User Proxy -> Bill Processing -> Expense Summarization.
    """

    def __init__(
        self,
        user_proxy: UserProxyAgent,
        bill_agent: BillProcessingAgent,
        summarizer: ExpenseSummarizationAgent,
    ) -> None:
        self.user_proxy = user_proxy
        self.bill_agent = bill_agent
        self.summarizer = summarizer
        self.turns: list[AgentTurn] = []

    def run_session(
        self,
        image_path: str | Path,
        *,
        allow_packaged_demo: bool = False,
    ) -> str:
        path = Path(image_path).resolve()
        self.turns.clear()

        def _add(agent_name: str, role: str, content: str) -> None:
            self.turns.append(AgentTurn(agent_name=agent_name, role=role, content=content))

        user_text = self.user_proxy.build_message(path)
        _add(self.user_proxy.name, self.user_proxy.role, user_text)

        try:
            bill_text, categorized = self.bill_agent.process(
                path,
                allow_packaged_demo=allow_packaged_demo,
            )
        except Exception as exc:
            logger.exception(
                "group_chat.bill_processing_failed",
                extra={"path": str(path), "action": "process"},
            )
            err = f"[{self.bill_agent.name}] Failed to extract bill data: {exc}"
            _add(self.bill_agent.name, self.bill_agent.role, err)
            return self._format_transcript()

        _add(self.bill_agent.name, self.bill_agent.role, bill_text)

        try:
            summ_text = self.summarizer.summarize(categorized)
        except Exception as exc:
            logger.exception(
                "group_chat.summarization_failed",
                extra={"path": str(path), "action": "summarize"},
            )
            err = f"[{self.summarizer.name}] Failed to summarize: {exc}"
            _add(self.summarizer.name, self.summarizer.role, err)
            return self._format_transcript()

        _add(self.summarizer.name, self.summarizer.role, summ_text)
        return self._format_transcript()

    def _format_transcript(self) -> str:
        blocks: list[str] = []
        for turn in self.turns:
            blocks.append(turn.content)
        return "\n\n---\n\n".join(blocks)


def _default_agents() -> GroupChatManager:
    user = UserProxyAgent(
        role="Initiates conversation with the Group Manager and provides the bill image to process.",
    )
    bill = BillProcessingAgent(
        role="Extracts data from bill images and organizes expenses into categories "
        "(groceries, dining, utilities, shopping, entertainment, other).",
    )
    summ = ExpenseSummarizationAgent(
        role="Analyzes categorized expenses: totals per category, trends, and unusual or high spending.",
    )
    return GroupChatManager(user, bill, summ)


def run_native_group_chat(
    image_path: str,
    *,
    allow_packaged_demo: bool = False,
) -> str:
    """Sequential group chat with explicit agent roles (no external agent framework)."""
    manager = _default_agents()
    return manager.run_session(image_path, allow_packaged_demo=allow_packaged_demo)


def run_sequential_agent_pipeline(
    image_path: str,
    *,
    allow_packaged_demo: bool = False,
) -> str:
    """Backward-compatible alias for the same native workflow."""
    return run_native_group_chat(image_path, allow_packaged_demo=allow_packaged_demo)


def run_autogen_group_chat(
    image_path: str,
    *,
    allow_packaged_demo: bool = False,
) -> str:
    """
    Deprecated name kept for CLI compatibility.

    AutoGen is not pinned here (version conflicts on modern Python). The same
    collaborative workflow runs via :func:`run_native_group_chat`.
    """
    logger.info(
        "group_chat.using_native_orchestrator",
        extra={"path": image_path, "action": "run_autogen_group_chat"},
    )
    return run_native_group_chat(image_path, allow_packaged_demo=allow_packaged_demo)
