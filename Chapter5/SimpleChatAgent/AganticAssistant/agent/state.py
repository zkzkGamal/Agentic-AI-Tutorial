from typing import TypedDict, List, Optional, Literal
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    # Full conversation history (HumanMessage / AIMessage objects)
    messages: List[BaseMessage]

    # Classified intent set by the router node
    intent: Optional[Literal["math", "email", "conversation"]]

    # Raw results returned by the execute node
    tool_results: Optional[str]

    # Human-readable summary produced by the summarize node
    summary: Optional[str]