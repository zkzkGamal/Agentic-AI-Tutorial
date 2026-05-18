# 🤖 Agentic AI Tutorial: A Comprehensive Guide

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-121212?style=flat&logo=chainlink)](https://langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Welcome to the **Agentic AI Tutorial**! This repository is your ultimate, hands-on guide to mastering the world of **Autonomous Agents**. We go beyond simple chat interfaces to build systems that can **reason, plan, and execute actions** using state-of-the-art Large Language Models (LLMs).

---

## 🌟 Why Agentic AI?

Traditional AI responds to prompts. **Agentic AI** takes it a step further:

- **Autonomy**: It decides which tools to use and how to solve a problem.
- **Reasoning**: It breaks down complex tasks into manageable steps.
- **Persistence**: It maintains state and memory over long interactions.
- **Action**: It interacts with the real world (APIs, databases, files).

---

## 🗺️ Learning Roadmap

| Chapter                                 | Level           | Focus Area                                         | Status      |
| :-------------------------------------- | :-------------- | :------------------------------------------------- | :---------- |
| **[Chapter 1](./Chapter1/Chapter1.md)** | 🟢 Beginner     | LLM Fundamentals, Providers (Ollama/OpenAI/Gemini) | ✅ Complete |
| **[Chapter 2](./Chapter2/Chapter2.md)** | 🔵 Intermediate | LangChain Orchestration, LCEL, Chains & Tools      | ✅ Complete |
| **[Chapter 3](./Chapter3/Chapter3.md)** | 🔵 Intermediate | Memory Systems, Entity Tracking & RAG              | ✅ Complete |
| **[Chapter 4](./Chapter4/Chapter4.md)** | 🟠 Advanced     | Autonomous Agents & LangGraph Patterns             | ✅ Complete |
| **[Chapter 5](./Chapter5/SimpleChatAgent/README.md)** | 🔴 Expert       | Multi-Node Agents & MCP Server Integration         | ✅ Complete |

---

## 🛠️ Core Tech Stack

- **Frameworks**: [LangChain](https://www.langchain.com/), [LangGraph](https://langchain-ai.github.io/langgraph/)
- **Protocols**: Model Context Protocol (MCP) by Anthropic
- **Models**: OpenAI (GPT-4o, GPT-3.5), Google Gemini (2.0 Flash), Ollama (Local)
- **Vector DB**: Chroma, FAISS
- **Embeddings**: Sentence Transformers (HuggingFace)

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8 or higher.
- API Keys for OpenAI/Google (optional if using Ollama exclusively).

### 2. Installation

Choose your preferred method:

#### SSH

```bash
git clone git@github.com:zkzkGamal/Agentic-AI-Tutorial.git
cd Agentic-AI-Tutorial
```

#### HTTPS

```bash
git clone https://github.com/zkzkGamal/Agentic-AI-Tutorial.git
cd Agentic-AI-Tutorial
```

### 3. Environment Setup

We recommend using a virtual environment for each chapter or a global one for the project.

```bash
# Create & Activate
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# OR: venv\Scripts\activate  # Windows

# Install Base Dependencies
pip install -r requirements.txt
```

### 4. Configuration

Each chapter contains its own `.env.example`. Copy it to `.env` and fill in your keys.

```bash
# Example for Chapter 1
cp Chapter1/.env.example Chapter1/.env
```

---

## 📚 Deep Dives

### [Chapter 1: LLM Fundamentals](./Chapter1/Chapter1.md)

- Direct API calls to OpenAI, Gemini, and Ollama.
- Streaming techniques.
- System prompt engineering (Personas).

### [Chapter 2: LangChain Orchestration](./Chapter2/Chapter2.md)

- Mastering **LCEL** (LangChain Expression Language).
- Building sequential and router chains.
- Binding and calling external tools.

### [Chapter 3: Memory & Context](./Chapter3/Chapter3.md)

- `ConversationBufferMemory` for full history.
- `ConversationEntityMemory` for fact extraction.
- **RAG (Retrieval-Augmented Generation)** with local vector stores.

### [Chapter 4: Autonomous Agents](./Chapter4/Chapter4.md)

- LangGraph **StateGraph** fundamentals.
- **ReAct**, **Router**, and **Sequential Pipeline** patterns.
- **Multi-Agent Collaboration** and **Self-Refine** loops.
- **Human-in-the-Loop** for production safety.

### [Chapter 5: Multi-Node LangGraph & MCP](./Chapter5/SimpleChatAgent/README.md)

- Building a decoupled architecture using the **Model Context Protocol (MCP)**.
- Deploying a local FastMCP Server with Mail and Math tools.
- Routing requests intelligently across multiple highly-specialized LangGraph nodes.
- Handling multi-turn state cleanly between Router, Execution, and Summary nodes.
- **Real-Time Automated Testing & GitHub Actions CI**: Features unbuffered real-time test execution and continuous integration via GitHub Actions (`.github/workflows/chapter5-ci.yml`). Automated testing is essential in Agentic AI to verify non-deterministic LLM intent routing, validate precise MCP tool schemas/contracts, and guarantee pipeline resilience when upgrading underlying foundation models.

### Chapter 5 Demo: See Agentic Workflow in Action

This chapter includes a live demo of the multi-node assistant handling different intents:
 - **Conversation** routed to a chitchat node
 - **Math requests** executed by the MCP tool server
 - **Email composition and sending** via a secure tool layer

Sample interaction:
```text
User: "Please add 42 and 17, then send the result to my email."
Router: detects math + tool request
Execute: calls MCP Math tool, then MCP Email tool
Summarize: returns a human-friendly response with results and confirmation
```

Read more in the Chapter 5 guide and view the demo flow: [Chapter 5 Demo](./Chapter5/SimpleChatAgent/README.md)

---

## 🔗 Related Repositories

Explore more tutorials and tools by the same author:

| Repository | Description |
|---|---|
| 🤖 [**Hands On AI Tutorials**](https://github.com/zkzkGamal/hands-on-ai-tutorial/) | A comprehensive, open-source AI tutorial covering ML Fundamentals (Regression, Classification, Clustering) and Deep Learning (NLP, CV) with practical code and mathematical intuition. |
| ⚡ [**Concurrent LLM Serving**](https://github.com/zkzkGamal/concurrent-llm-serving) | Hands-on guide to serving Large Language Models efficiently at scale with concurrency |

> These repositories complement this tutorial — once you've mastered the fundamentals here, explore agents and production LLM deployment next!

---

## 🤝 Contributing

We love contributions! Whether you're fixing a typo or adding a new agent pattern:

1. Fork the project.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 👤 Author

**Zkzk** - _AI Engineer & Educator_

- GitHub: [@zkzkGamal](https://github.com/zkzkGamal)

---

_Disclaimer: This tutorial is for educational purposes. Costs may apply for cloud LLM usage._
