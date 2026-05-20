from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import TextMessage, ToolCallSummaryMessage


def final_agent_text(result: TaskResult) -> str:
    """Return the last text-like assistant message from a task result."""
    for message in reversed(result.messages):
        if isinstance(message, (TextMessage, ToolCallSummaryMessage)):
            return str(message.content)
    return ""
