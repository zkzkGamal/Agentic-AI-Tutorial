# Agentic AI Engineering

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-121212?style=flat&logo=chainlink)](https://langchain.com/)
[![LangGraph](https://img.shields.io/badge/Framework-LangGraph-1C3C3C?style=flat)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Agentic AI Engineering** is a production-grade engineering resource for building modern agentic AI systems with LangChain, LangGraph, RAG, MCP, local models, and deployable Python services.

The repository leads toward the architecture implemented in **Chapter 5**: a multi-node LangGraph assistant connected to a standalone MCP server, with intent routing, tool execution, response summarization, email tooling, math tools, automated tests, and GitHub Actions CI. Earlier chapters build the required layers underneath it: provider abstraction, LCEL orchestration, vector retrieval, memory, ReAct agents, router graphs, sequential workflows, multi-agent collaboration, and human-in-the-loop control.

This is not a beginner chatbot walkthrough. It is a structured engineering path for developers building systems that need state, tools, routing, retrieval, observability, modularity, and model-provider flexibility.

## What You'll Build

- **Multi-node LangGraph assistant** with router, execution, summarization, and conversation nodes.
- **MCP tool server** exposing isolated math and email tools over a decoupled server boundary.
- **Tool-using ReAct workflows** that call external capabilities through typed tool contracts.
- **RAG pipelines** using vector stores, embeddings, retrieval chains, and local document context.
- **Provider-flexible LLM interfaces** across OpenAI, Gemini, and Ollama.
- **Agent routing systems** for sequential, router-based, ReAct, and multi-agent workflows.
- **Human-in-the-loop execution paths** for safer agent behavior in production-style flows.
- **Tested agent components** with pytest coverage for Chapter 5 node behavior and tool contracts.
- **CI-backed agent validation** through GitHub Actions for repeatable checks on graph and MCP behavior.

## Engineering Roadmap

| Chapter | Engineering Milestone | Core Systems | Status |
| :-- | :-- | :-- | :-- |
| **[Chapter 1](./Chapter1/Chapter1.md)** | LLM provider foundation | OpenAI, Gemini, Ollama, streaming, system prompts | Complete |
| **[Chapter 2](./Chapter2/Chapter2.md)** | LangChain orchestration layer | LCEL, chains, tool binding, router and sequential composition | Complete |
| **[Chapter 3](./Chapter3/Chapter3.md)** | Retrieval and memory infrastructure | Conversation memory, entity tracking, ChromaDB, FAISS, Sentence Transformers | Complete |
| **[Chapter 4](./Chapter4/Chapter4.md)** | Agent graph patterns | LangGraph StateGraph, ReAct, routers, sequential agents, multi-agent collaboration, HITL | Complete |
| **[Chapter 5](./Chapter5/SimpleChatAgent/README.md)** | Production-style agent runtime | Multi-node LangGraph assistant, MCP server integration, tool isolation, tests, CI | Complete |

## System Architecture Focus

Chapter 5 is the reference implementation for the repository's production architecture:

- **LangGraph StateGraph** coordinates routing, execution, summarization, and conversation paths.
- **Router node** classifies user intent before allocating work to the right execution path.
- **Execution node** binds LangChain tool-calling agents to MCP-exposed capabilities.
- **Summarization node** converts raw tool responses into user-facing output.
- **Conversation node** handles non-tool interactions without invoking the heavier tool path.
- **FastMCP server** isolates operational tools from the agent runtime.
- **Pytest suite and GitHub Actions CI** validate tool behavior, routing assumptions, and graph execution contracts.

Start with the full implementation here: **[Chapter 5: Multi-Node LangGraph Agent with MCP Tools](./Chapter5/SimpleChatAgent/README.md)**.

## Tech Stack

- **Agent frameworks**: LangChain, LangGraph
- **Protocols**: Model Context Protocol (MCP), FastMCP
- **LLM providers**: OpenAI, Google Gemini, Ollama
- **Retrieval**: ChromaDB, FAISS, Sentence Transformers
- **Backend tooling**: Python, FastAPI, pytest, GitHub Actions
- **ML ecosystem**: PyTorch, TensorFlow, Hugging Face

## Repository Structure

```text
.
├── Chapter1/                         # LLM providers, streaming, prompt foundations
├── Chapter2/                         # LangChain LCEL, chains, tools, routing
├── Chapter3/                         # Memory, retrieval, vector stores, RAG
├── Chapter4/                         # LangGraph agent patterns and HITL workflows
├── Chapter5/
│   └── SimpleChatAgent/
│       ├── AganticAssistant/         # LangGraph assistant runtime
│       ├── McpServer/                # MCP tool server
│       ├── tests/                    # pytest coverage for tools and nodes
│       └── demo/                     # screenshots and demo recordings
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Clone the Repository

```bash
git clone git@github.com:zkzkGamal/agentic-ai-engineering.git
cd agentic-ai-engineering
```

Or with HTTPS:

```bash
git clone https://github.com/zkzkGamal/agentic-ai-engineering.git
cd agentic-ai-engineering
```

### 2. Create an Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

Install the full repository stack:

```bash
pip install -r requirements.txt
```

Or install only the chapter you are running:

```bash
pip install -r Chapter1/requirements.txt
pip install -r Chapter2/requirements.txt
pip install -r Chapter3/requirements.txt
pip install -r Chapter4/requirements.txt
pip install -r Chapter5/SimpleChatAgent/AganticAssistant/requirements.txt
pip install -r Chapter5/SimpleChatAgent/McpServer/requirements.txt
```

### 4. Configure Environment Variables

Each chapter that needs credentials includes its own `.env.example`.

```bash
cp Chapter1/.env.example Chapter1/.env
cp Chapter2/.env.example Chapter2/.env
cp Chapter3/.env.example Chapter3/.env
cp Chapter4/.env.example Chapter4/.env
```

Chapter 5 has separate runtime boundaries for the assistant and MCP server:

```bash
cp Chapter5/SimpleChatAgent/AganticAssistant/.env.example Chapter5/SimpleChatAgent/AganticAssistant/.env
cp Chapter5/SimpleChatAgent/McpServer/.env.example Chapter5/SimpleChatAgent/McpServer/.env
```

## Running the Chapter 5 Agent System

Start the MCP server:

```bash
python3 Chapter5/SimpleChatAgent/McpServer/main.py
```

In another terminal, run the LangGraph assistant:

```bash
python3 Chapter5/SimpleChatAgent/AganticAssistant/main.py
```

Run the Chapter 5 test suite:

```bash
cd Chapter5/SimpleChatAgent
pytest tests
```

## Chapter Details

### [Chapter 1: LLM Provider Foundation](./Chapter1/Chapter1.md)

Direct integration with OpenAI, Gemini, and Ollama. Covers provider calls, streaming responses, prompt structure, and model-facing interfaces used later by orchestration and agent layers.

### [Chapter 2: LangChain Orchestration Layer](./Chapter2/Chapter2.md)

Builds LCEL pipelines, sequential chains, router chains, prompt templates, output handling, and tool binding. This chapter establishes the composition patterns used by larger agent systems.

### [Chapter 3: Retrieval and Memory Infrastructure](./Chapter3/Chapter3.md)

Implements conversation memory, entity tracking, embeddings, local vector stores, and RAG flows with ChromaDB, FAISS, and Sentence Transformers.

### [Chapter 4: Agent Graph Patterns](./Chapter4/Chapter4.md)

Moves from chains into stateful agent workflows with LangGraph StateGraph. Covers ReAct, router agents, sequential agents, multi-agent collaboration, self-refine loops, and human-in-the-loop control.

### [Chapter 5: Multi-Node LangGraph and MCP Runtime](./Chapter5/SimpleChatAgent/README.md)

Implements a decoupled agent runtime with a LangGraph assistant and MCP server. The assistant routes requests across specialized nodes, invokes MCP tools, summarizes tool outputs, and validates behavior with automated tests and CI.

## Production Repositories

| Repository | Focus |
| :-- | :-- |
| **[zkzkAgent](https://github.com/zkzkGamal/zkzkAgent)** | Production agent platform work and applied agent engineering patterns. |
| **[concurrent-llm-serving](https://github.com/zkzkGamal/concurrent-llm-serving)** | Concurrent LLM serving patterns for higher-throughput inference systems. |

## Author

**Zkaria Gamal - AI Engineer**

- GitHub: [@zkzkGamal](https://github.com/zkzkGamal)
- LinkedIn: [Zkaria Gamal](https://www.linkedin.com/in/zkaria-gamal/)

## License

This repository is available under the [MIT License](LICENSE).
