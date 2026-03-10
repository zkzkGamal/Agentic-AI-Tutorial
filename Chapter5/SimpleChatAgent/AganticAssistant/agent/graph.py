"""
LangGraph - Agent Graph
------------------------
Wires the four nodes into a compiled StateGraph:

    START
      │
    router  ─── (intent=math|email) ──► execute ──► summarize ──► END
      │
      └────── (intent=conversation) ──► conversation ──────────── END
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.nodes.router import router
from agent.nodes.act import execute
from agent.nodes.summarize import summarize
from agent.nodes.conversation import conversation


# ── Conditional edge: decide which branch to follow after routing ────────────

def _route_by_intent(state: AgentState) -> str:
    intent = state.get("intent", "conversation")
    if intent in ("math", "email"):
        return "execute"
    return "conversation"


# ── Build the graph ──────────────────────────────────────────────────────────

def build_graph(checkpointer=None):
    """Build and compile the multi-node agent graph.

    Args:
        checkpointer: optional LangGraph checkpointer for multi-turn memory.
                      Pass ``MemorySaver()`` to enable persistent conversation
                      history across invocations.

    Returns:
        A compiled LangGraph ``CompiledGraph`` ready to ``.invoke()`` or
        ``.ainvoke()``.
    """
    builder = StateGraph(AgentState)

    # ── Add nodes ───────────────────────────────────────────────────────────
    builder.add_node("router", router)
    builder.add_node("execute", execute)
    builder.add_node("summarize", summarize)
    builder.add_node("conversation", conversation)

    # ── Edges ────────────────────────────────────────────────────────────────
    # Always start at router
    builder.add_edge(START, "router")

    # Router decides: tool path or conversation path
    builder.add_conditional_edges(
        "router",
        _route_by_intent,
        {
            "execute": "execute",
            "conversation": "conversation",
        },
    )

    # Tool path: execute → summarize → END
    builder.add_edge("execute", "summarize")
    builder.add_edge("summarize", END)

    # Conversation path: conversation → END
    builder.add_edge("conversation", END)

    return builder.compile(checkpointer=checkpointer)


# ── Default compiled graph (with in-memory checkpointer for multi-turn) ──────
memory = MemorySaver()
graph = build_graph(checkpointer=memory)
from IPython.display import Image

png_bytes = graph.get_graph().draw_mermaid_png()

with open("Chapter5/SimpleChatAgent/AganticAssistant/graph.png", "wb") as f:
    f.write(png_bytes)