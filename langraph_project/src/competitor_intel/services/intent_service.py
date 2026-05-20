from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from competitor_intel.schemas import ParsedQuery


def parse_user_intent(llm: ChatOpenAI, user_text: str) -> ParsedQuery:
    structured = llm.with_structured_output(ParsedQuery)
    system = SystemMessage(
        content=(
            "You extract structured intent for a competitor intelligence assistant focused on "
            "local clothing and apparel retail. "
            "If the user does not name a neighborhood, city, or recognizable local area, set "
            "needs_clarification to true and ask one concise clarification_question "
            "(e.g. which city or neighborhood). "
            "If they mention Koramangala, Bangalore, or similar, location must include both "
            "locality and city when possible. "
            "Map phrases like 'for our marketing team' to audience marketing_strategy."
        )
    )
    human = HumanMessage(content=user_text)
    result = structured.invoke([system, human])
    if not isinstance(result, ParsedQuery):
        return ParsedQuery(
            needs_clarification=True,
            clarification_question="Which neighborhood or city should we analyze?",
        )
    return result
