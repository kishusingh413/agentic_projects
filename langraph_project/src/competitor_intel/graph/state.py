from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from competitor_intel.schemas import BeneficiaryAudience


class CompetitorIntelState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    location: str | None
    store_focus: str
    audience: BeneficiaryAudience | None
    needs_clarification: bool
    clarification_question: str | None
    search_context: str
    report: str | None
    last_user_text: str
