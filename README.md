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
| **Chapter 4**                           | 🟠 Advanced     | ReAct Pattern & Basic Agents                       | 🚧 Upcoming |
| **Chapter 5**                           | 🔴 Expert       | Multi-Agent Systems with LangGraph                 | 📅 Planned  |
| **Chapter 6**                           | 💼 Real-World   | Production Deployment & Case Studies               | 📅 Planned  |

---

## 🛠️ Core Tech Stack

- **Frameworks**: [LangChain](https://www.langchain.com/), [LangGraph](https://langchain-ai.github.io/langgraph/)
- **Models**: OpenAI (GPT-4o), Google Gemini (2.0 Flash), Ollama (Local Llama 3/Mistral)
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
