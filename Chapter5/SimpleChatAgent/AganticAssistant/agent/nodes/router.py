"""
Router Node
-----------
Classifies the latest user message into one of three intents:
  - "math"         → needs the add / multiply tools
  - "email"        → needs the email tools
  - "conversation" → general chitchat, no tools needed
"""
import os
import logging
import pathlib
import environ
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# ── env ──────────────────────────────────────────────────────────────────────
_base = pathlib.Path(__file__).parent.parent.parent
_e = environ.Env()
_e.read_env(str(_base / ".env"))

_OPENAI_API_KEY = _e("OPENAI_API_KEY", default="")

_SYSTEM_PROMPT = """\
You are an intent classifier. Given the user message below, reply with EXACTLY
one of these three words — nothing else:
  math
  email
  conversation

Rules:
- "math"         → the user wants to do arithmetic (add, multiply, calculate …)
- "email"        → the user wants to send, read, check or reply to email
- "conversation" → anything else (greetings, general questions, chitchat …)
"""


def _build_llm():
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key=_OPENAI_API_KEY,
        temperature=0,
    )


def router(state: dict) -> dict:
    """Classify intent and store it in state['intent']."""
    llm = _build_llm()

    last_message = state["messages"][-1]
    # Extract text whether it's a BaseMessage or a plain string
    user_text = (
        last_message.content
        if hasattr(last_message, "content")
        else str(last_message)
    )

    response = llm.invoke(
        [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_text),
        ]
    )

    raw = response.content.strip().lower()
    intent = raw if raw in {"math", "email", "conversation"} else "conversation"
    logger.info(f"[router] intent classified as: {intent!r} (raw={raw!r})")

    return {"intent": intent}
