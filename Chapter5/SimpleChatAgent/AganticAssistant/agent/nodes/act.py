"""
Execute Node
------------
Runs the user's request through a LangChain ReAct agent that has access
to ALL MCP-backed tools (math + email).  The final text output is stored
in state['tool_results'] so the summarize node can format it.
"""
import pathlib
import logging
import environ
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage

# Import every MCP tool wrapper
from tools.mcp_math import add, multiply
from tools.mcp_mail import send_email, check_inbox, read_email, reply_to_email

logger = logging.getLogger(__name__)

_base = pathlib.Path(__file__).parent.parent.parent
_e = environ.Env()
_e.read_env(str(_base / ".env"))

_OPENAI_API_KEY = _e("OPENAI_API_KEY", default="")

ALL_TOOLS = [add, multiply, send_email, check_inbox, read_email, reply_to_email]


def _build_agent_executor():
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key=_OPENAI_API_KEY,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a helpful assistant with access to math and email tools. "
                    "Use the tools as needed to fulfill the user's request. "
                    "Be precise and thorough."
                ),
            ),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, ALL_TOOLS, prompt)
    return AgentExecutor(agent=agent, tools=ALL_TOOLS, verbose=True)


async def execute(state: dict) -> dict:
    """Run the ReAct agent and store the raw output in tool_results."""
    executor = _build_agent_executor()

    last_message = state["messages"][-1]
    user_text = (
        last_message.content
        if hasattr(last_message, "content")
        else str(last_message)
    )

    # Pass previous messages as chat history (skip the last — it's the input)
    history = state["messages"][:-1]

    result = await executor.ainvoke(
        {
            "input": user_text,
            "chat_history": history,
        }
    )

    raw_output = result.get("output", "")
    logger.info(f"[execute] tool output: {raw_output[:120]}…")

    return {"tool_results": raw_output}
