# 🧠 Agentic Assistant

This directory contains the **LangGraph** orchestrator (the "Brain") for Chapter 5. It is responsible for maintaining conversation state, reasoning about user intent, and deciding when to call external tools.

## 📁 Project Structure

```text
AganticAssistant/
├── main.py                 # The interactive terminal loop
├── .env                    # API keys (e.g., OPENAI_API_KEY)
├── tools/                  # The remote clients that connect to MCP
│   ├── mcp_mail.py         # Proxies email requests to the server
│   └── mcp_math.py         # Proxies math requests to the server
└── agent/                  # The LangGraph logic
    ├── __init__.py         # Graph export
    ├── graph.py            # Compiles the StateGraph and edges
    ├── state.py            # Defined the TypedDict for AgentState
    └── nodes/              # The functional graph steps
        ├── __init__.py 
        ├── conversation.py # Handles chitchat
        ├── execute.py      # The ReAct Tool-Calling Agent
        ├── router.py       # The Intent Classifier
        └── summarize.py    # Formats raw output into human text
```

## 🎯 What we built here

We designed a highly efficient **StateGraph** architecture that splits responsibilities across distinct functional nodes, avoiding the cost and latency of passing every single user query into a massive, heavy-weight tool-calling agent.

### 🛠️ Detailed Node Explanations:
1. **`router.py` (The Fast Path)**: Uses a fast, cheap LLM prompting technique to classify intent (`math`, `email`, `conversation`). It reads the user message and simply modifies the `intent` key in the `AgentState`.
2. **`execute.py` (The Heavy Lifter)**: If the router picks a tool-based intent, the graph shifts here. This node uses LangChain's `create_tool_calling_agent`. Crucially, it dynamically binds to the `tools/mcp_mail.py` and `tools/mcp_math.py` clients. Those clients open Server-Sent Event (SSE) connections to `127.0.0.1:8000`. The agent doesn't execute the tools locally!
3. **`summarize.py` (The Formatter)**: The `execute` node often spits out raw tool JSON or error codes. The `summarize` node takes that raw string and uses an LLM to synthesize a polite, conversational response for the user.
4. **`conversation.py` (The Chitchat Fallback)**: If the user just says "Hello," we skip `execute` and `summarize` entirely. This node feeds the conversation history to the LLM for a direct response, saving tokens, money, and time.

### The Orchestrator (`agent/graph.py`):
This file glues the nodes above together using `langgraph`. We map conditional edges mapping the output of the Router to the respective nodes. We also use `MemorySaver` to provide a persistent `thread_id`, giving the terminal application memory across multiple turns.

## 📈 Tutorial: How to expand this agent
In this architecture, scaling is structured and manageable:

1. **Add new intents**: Modify the system prompt in `agent/nodes/router.py` to recognize a new intent (e.g., `"weather"`). Update the `should_continue` conditional edge routing logic in `graph.py` to direct `"weather"` intents.
2. **Add more execution branches**: You could add a completely new `agent/nodes/research.py` node that hooks into Wikipedia or Web Browsing tools, and route to it when the intent is `"research"`.

