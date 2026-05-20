"""Agent definitions for the financial portfolio group chat."""

from __future__ import annotations

from typing import TYPE_CHECKING

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from src.stateflow import (
    GROUP_CHAT_MANAGER,
    GROWTH_INVESTMENT,
    INVESTMENT_ADVISOR,
    PORTFOLIO_ANALYSIS,
    USER_PROXY,
    VALUE_INVESTMENT,
)

if TYPE_CHECKING:
    from src.config import AppSettings


def build_model_client(settings: "AppSettings") -> OpenAIChatCompletionClient:
    return OpenAIChatCompletionClient(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
    )


def build_user_proxy(input_func) -> UserProxyAgent:
    return UserProxyAgent(
        USER_PROXY,
        description=(
            "Represents the end user: states their goal to manage investments "
            "and supplies salary and portfolio details for the team."
        ),
        input_func=input_func,
    )


def build_group_chat_manager(model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    return AssistantAgent(
        GROUP_CHAT_MANAGER,
        model_client=model_client,
        description=(
            "Coordinates the group chat: acknowledges the user and hands off "
            "to portfolio analysis next."
        ),
        system_message=(
            "You are the Group Chat Manager. Keep replies short. "
            "Acknowledge the user's intent to manage investments. "
            "Explicitly ask the PortfolioAnalysisAgent to analyze the user's salary and portfolio "
            "and to output PRIMARY_TRACK. Do not perform portfolio math yourself."
        ),
    )


def build_portfolio_analysis_agent(model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    return AssistantAgent(
        PORTFOLIO_ANALYSIS,
        model_client=model_client,
        description=(
            "Summarizes the user's portfolio, risk context, and recommends a primary style: "
            "growth vs value."
        ),
        system_message=(
            "You are the Portfolio Analysis Agent. "
            "Summarize the user's salary, savings rate implied by SIPs, and allocation across "
            "fixed deposits, equities, real estate, and other notes. "
            "Decide whether the PRIMARY focus should be growth-oriented investments or "
            "value / wealth-preservation investments based on age implied by context (if missing, "
            "assume mid-career), liquidity needs, and concentration risk. "
            "You MUST end your message with a single line exactly in this form:\n"
            "PRIMARY_TRACK: GROWTH\n"
            "or\n"
            "PRIMARY_TRACK: VALUE\n"
            "Choose one. Do not add extra text on that line."
        ),
    )


def build_growth_investment_agent(model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    return AssistantAgent(
        GROWTH_INVESTMENT,
        model_client=model_client,
        description=(
            "Suggests growth-oriented ideas (e.g., diversified equity, thematic growth, "
            "higher risk-reward allocations) with clear caveats."
        ),
        system_message=(
            "You are the Growth Investment Agent. Provide concrete growth-focused suggestions "
            "(e.g., equity index funds, sector considerations, reinvestment of SIPs) "
            "tailored to the user's profile. "
            "Include risk warnings and diversification. Do not claim guaranteed returns."
        ),
    )


def build_value_investment_agent(model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    return AssistantAgent(
        VALUE_INVESTMENT,
        model_client=model_client,
        description=(
            "Suggests value-oriented, stability-focused ideas (quality at reasonable price, "
            "dividend tilt, defensive allocation) with caveats."
        ),
        system_message=(
            "You are the Value Investment Agent. Provide value- and stability-focused suggestions "
            "(e.g., broad quality equity, dividend strategies where appropriate, bonds/CD ladders "
            "for stability) tailored to the user. "
            "Include risk warnings. Do not claim guaranteed returns."
        ),
    )


def build_investment_advisor_agent(model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    return AssistantAgent(
        INVESTMENT_ADVISOR,
        model_client=model_client,
        description=(
            "Produces the final personalized report combining portfolio analysis and both "
            "growth and value angles."
        ),
        system_message=(
            "You are the Investment Advisor Agent. Produce a clear, personalized financial "
            "report with sections: Executive summary, Portfolio snapshot, Growth ideas (from the "
            "Growth agent context), Value/stability ideas (from the Value agent context), "
            "Risks and monitoring, Next steps. "
            "This is educational information, not licensed financial advice. "
            "End your entire message with a new line containing only: TERMINATE"
        ),
    )


def build_all_participants(
    model_client: OpenAIChatCompletionClient,
    input_func,
) -> list:
    """Return ordered participants for SelectorGroupChat."""
    return [
        build_user_proxy(input_func),
        build_group_chat_manager(model_client),
        build_portfolio_analysis_agent(model_client),
        build_growth_investment_agent(model_client),
        build_value_investment_agent(model_client),
        build_investment_advisor_agent(model_client),
    ]
