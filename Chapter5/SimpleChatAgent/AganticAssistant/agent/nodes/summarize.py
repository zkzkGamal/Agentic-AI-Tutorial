"""
Summarize Node
--------------
Takes the raw tool_results string and asks the LLM to turn it into a
concise, user-friendly answer.  The answer is appended to messages.
"""
import pathlib
import logging
import environ
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

_base = pathlib.Path(__file__).parent.parent.parent
_e = environ.Env()
_e.read_env(str(_base / ".env"))

_OPENAI_API_KEY = _e("OPENAI_API_KEY", default="")

_SYSTEM = """\
You are a result summarizer. You will be given the raw output of a tool execution.
Summarize it in a clear, friendly, and concise manner for the end-user.
Do NOT include technical jargon or internal tool names.
"""


def _build_llm():
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key=_OPENAI_API_KEY,
        temperature=0.3,
    )


def summarize(state: dict) -> dict:
    """Summarise tool_results into a human-friendly reply."""
    llm = _build_llm()

    tool_output = state.get("tool_results") or "No result was returned."
    user_query = state["messages"][-1]
    user_text = (
        user_query.content if hasattr(user_query, "content") else str(user_query)
    )

    prompt = (
        f"User asked: {user_text}\n\n"
        f"Tool returned:\n{tool_output}\n\n"
        "Please provide a clear, friendly summary of the result."
    )

    response = llm.invoke(
        [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=prompt),
        ]
    )

    summary_text = response.content
    logger.info(f"[summarize] summary: {summary_text[:80]}…")

    ai_msg = AIMessage(content=summary_text)
    return {
        "summary": summary_text,
        "messages": state["messages"] + [ai_msg],
    }
