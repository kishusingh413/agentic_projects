from langchain_openai import ChatOpenAI

from competitor_intel.config import get_settings


def get_chat_model() -> ChatOpenAI:
    settings = get_settings()
    kwargs: dict[str, object] = {
        "model": settings.openai_model,
        "api_key": settings.openai_api_key,
    }
    if settings.openai_base_url is not None:
        kwargs["base_url"] = settings.openai_base_url
    return ChatOpenAI(**kwargs)
