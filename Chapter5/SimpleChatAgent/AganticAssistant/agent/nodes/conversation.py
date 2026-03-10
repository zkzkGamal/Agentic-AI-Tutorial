"""
Conversation Node
-----------------
Handles general chitchat and knowledge questions — no tool calls.
Passes the full message history to the LLM and appends its reply.
"""
import pathlib
import logging
import environ
from langchain_core.messages import SystemMessage, AIMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

_base = pathlib.Path(__file__).parent.parent.parent
_e = environ.Env()
_e.read_env(str(_base / ".env"))

_OPENAI_API_KEY = _e("OPENAI_API_KEY", default="")

_SYSTEM = SystemMessage(
    content=(
        "You are a friendly, knowledgeable AI assistant. "
        "Engage naturally with the user and answer their questions helpfully. "
        "Be concise but thorough."
    )
)


def _build_llm():
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key=_OPENAI_API_KEY,
        temperature=0.7,
    )


def conversation(state: dict) -> dict:
    """Generate a conversational reply and append it to messages."""
    llm = _build_llm()

    messages_with_sys = [_SYSTEM] + list(state["messages"])
    response = llm.invoke(messages_with_sys)

    ai_msg = AIMessage(content=response.content)
    logger.info(f"[conversation] reply: {response.content[:80]}…")

    return {"messages": state["messages"] + [ai_msg]}
