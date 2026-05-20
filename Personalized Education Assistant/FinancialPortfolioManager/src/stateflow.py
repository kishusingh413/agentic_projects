"""StateFlow: deterministic workflow routing based on portfolio analysis output."""

from enum import Enum
from typing import Sequence

from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage


class InvestmentTrack(str, Enum):
    """Primary investment style chosen after portfolio analysis."""

    GROWTH = "growth"
    VALUE = "value"


# Agent names (must match AssistantAgent / UserProxyAgent names exactly)
USER_PROXY = "UserProxyAgent"
GROUP_CHAT_MANAGER = "GroupChatManager"
PORTFOLIO_ANALYSIS = "PortfolioAnalysisAgent"
GROWTH_INVESTMENT = "GrowthInvestmentAgent"
VALUE_INVESTMENT = "ValueInvestmentAgent"
INVESTMENT_ADVISOR = "InvestmentAdvisorAgent"


class PortfolioStateFlow:
    """Routes the group chat using rules derived from the Portfolio Analysis Agent output.

    The analysis agent must end with a line: PRIMARY_TRACK: GROWTH | VALUE
    so routing remains auditable without guessing from free text.
    """

    PRIMARY_TRACK_PREFIX = "PRIMARY_TRACK:"

    @classmethod
    def parse_investment_track(cls, analysis_text: str) -> InvestmentTrack:
        prefix = cls.PRIMARY_TRACK_PREFIX.upper()
        for line in analysis_text.splitlines():
            stripped = line.strip().upper()
            if not stripped.startswith(prefix):
                continue
            parts = stripped.split(":", 1)
            if len(parts) != 2:
                continue
            token = parts[1].strip()
            if token == "VALUE":
                return InvestmentTrack.VALUE
            if token == "GROWTH":
                return InvestmentTrack.GROWTH
        return InvestmentTrack.GROWTH

    @classmethod
    def _sources_in_thread(cls, messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> set[str]:
        return {m.source for m in messages if hasattr(m, "source")}

    @classmethod
    def select_next_speaker(
        cls,
        messages: Sequence[BaseAgentEvent | BaseChatMessage],
        *,
        participant_names: frozenset[str],
    ) -> str:
        """Return the next speaker name; raises if the thread is in an unexpected state."""
        if not messages:
            raise ValueError("Message thread is empty.")

        last = messages[-1]
        last_source = last.source

        if last_source == "user":
            if USER_PROXY not in participant_names:
                raise ValueError("UserProxyAgent is not a participant.")
            return USER_PROXY

        if last_source == USER_PROXY:
            return GROUP_CHAT_MANAGER

        if last_source == GROUP_CHAT_MANAGER:
            return PORTFOLIO_ANALYSIS

        if last_source == PORTFOLIO_ANALYSIS:
            text = last.to_text() if hasattr(last, "to_text") else str(last)
            track = cls.parse_investment_track(text)
            if track == InvestmentTrack.GROWTH:
                return GROWTH_INVESTMENT
            return VALUE_INVESTMENT

        sources = cls._sources_in_thread(messages)

        if last_source == GROWTH_INVESTMENT:
            if VALUE_INVESTMENT not in sources:
                return VALUE_INVESTMENT
            return INVESTMENT_ADVISOR

        if last_source == VALUE_INVESTMENT:
            if GROWTH_INVESTMENT not in sources:
                return GROWTH_INVESTMENT
            return INVESTMENT_ADVISOR

        raise ValueError(f"Unhandled last speaker for StateFlow: {last_source}")
