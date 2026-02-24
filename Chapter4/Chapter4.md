# Chapter 4: Autonomous Agents with LangGraph

In Chapter 3, we mastered Memory & RAG. Now we take the final leap: building **Autonomous Agents** that can reason, use tools, collaborate, and self-improve — all orchestrated by **LangGraph**.

---

## 🎯 Lesson Objectives

By the end of this chapter, you will be able to:

- Build **State Graphs** using LangGraph's `StateGraph`, `START`, and `END`.
- Create a **Minimal LLM Agent** (single-node chat loop).
- Implement **Streaming Responses** for a real-time chat experience.
- Build a **ReAct Agent** (Reason + Act) with custom tools.
- Handle **Parallel Tool Calls** within an agent loop.
- Design **Sequential Pipelines** for multi-step text transformation.
- Implement **Router / Conditional Branching** to classify & dispatch queries.
- Orchestrate **Multi-Agent Collaboration** (network/peer-to-peer pattern).
- Apply the **Reflection / Self-Refine Loop** for iterative quality improvement.
- Understand the **Human-in-the-Loop** pattern for safety & control.

---

## 🧠 Core Concept: LangGraph StateGraph

LangGraph models agent workflows as **directed graphs**:

- **Nodes** — functions that read & update a shared `State`.
- **Edges** — define the flow between nodes (`START → node → END`).
- **Conditional Edges** — route dynamically based on state.

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(AgentState)
graph.add_node("process", process_fn)
graph.add_edge(START, "process")
graph.add_edge("process", END)
app = graph.compile()
```

---

## 💻 Patterns Implemented

### 1. Minimal LLM Agent

One-node graph → LLM call → append response to messages.

```python
class AgentState(TypedDict):
    messages: list[Union[AIMessage, HumanMessage]]

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state['messages'])
    state['messages'].append(AIMessage(content=response.content))
    return state
```

### 2. Streaming Agent

Same graph, but uses `llm.stream()` for token-by-token output.

```python
def stream_process(state: AgentState) -> AgentState:
    response = ""
    for chunk in llm.stream(state['messages']):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            response += chunk.content
    state['messages'].append(AIMessage(content=response))
    return state
```

### 3. ReAct Agent (Reason + Act)

The classic loop: LLM decides → call tools or give final answer → repeat.

```text
START → our_agent → [has tool calls?]
                      ├── Yes → tools → our_agent (loop)
                      └── No  → END
```

```python
@tool
def add(a: int, b: int) -> int:
    """Addition function"""
    return a + b

model_with_tools = llm.bind_tools([add, subtract, multiply])
```

### 4. Parallel Tool Handling

Custom tool executor that processes **all tool calls at once**, returning multiple `ToolMessage` results in a single step.

### 5. Sequential Pipeline

Fixed linear chain for multi-step text transformation:

```text
START → summarize → translate → critique → END
```

Each node reads previous output from state and writes its own result — no branching, just step-by-step processing.

### 6. Router / Conditional Branching

Classify the user query, then dispatch to a specialist node:

```text
START → classify → [category?]
                    ├── "math"    → math_node    → END
                    ├── "joke"    → joke_node    → END
                    └── "general" → general_node → END
```

### 7. Multi-Agent Collaboration

Peer-to-peer pattern with **Researcher**, **Writer**, and **Critic** agents:

- All agents share the same message history.
- Each has its own system prompt & capabilities (e.g., only Researcher has web search tools).
- A **router** cycles through agents until `FINAL ANSWER` is detected.

### 8. Reflection / Self-Refine Loop

Iterative quality improvement:

```text
START → generate → critique → [approved?]
                                ├── No  → generate (loop)
                                └── Yes → END
```

The generator produces a draft, the critic evaluates it, and the loop repeats until the output is approved or max iterations are reached.

### 9. Human-in-the-Loop

Interrupt & wait for human input — essential for safety and quality control in production agents. For a full implementation, see: [zkzkAgent Human-in-the-Loop](https://github.com/zkzkGamal/zkzkAgent/blob/main/core/agent.py)

---

## 🛠️ Environment Setup

```bash
pip install langchain langchain-openai langchain-google-genai langchain-ollama langchain-community langchain-core langgraph graphviz transformers sentence-transformers ddgs
```

---

## ⚖️ When to use which pattern?

| Pattern           | Best For                                       |
| :---------------- | :--------------------------------------------- |
| **Minimal Agent** | Simple Q&A, single LLM call                    |
| **ReAct**         | Tool-using agents, step-by-step reasoning      |
| **Sequential**    | Multi-step text transformation pipelines       |
| **Router**        | Intent classification & specialist dispatch    |
| **Multi-Agent**   | Complex tasks requiring diverse expertise      |
| **Self-Refine**   | Iterative quality improvement (writing, code)  |
| **Human-in-Loop** | Safety-critical, production-grade applications |

---

## 🏁 Summary

You have now completed the full Agentic AI Tutorial! Your toolkit:

1. **Chapter 1**: Raw AI calls (direct SDK usage).
2. **Chapter 2**: Orchestration & Tools (LangChain + LCEL).
3. **Chapter 3**: Custom Knowledge & Context (Memory + RAG).
4. **Chapter 4**: Autonomous Agents (LangGraph patterns).

You are now equipped to build production-ready autonomous AI agents that can reason, act, collaborate, and self-improve.

---

_Created with ❤️ by the Zkzk
