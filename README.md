# Agentic AI Tutorial

Welcome to the **Agentic AI Tutorial**! This repository provides a comprehensive, hands-on guide to building intelligent agents that can reason, plan, and act autonomously using Large Language Models (LLMs).

Whether you're a beginner taking your first steps into AI or an intermediate developer looking to master agentic systems, this tutorial will guide you from basic LLM interactions to building sophisticated, multi-agent workflows.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- An API key for at least one LLM provider (OpenAI, Google Gemini, or Ollama)

### Installation

1. **Clone the repository:**

   #### ssh

   ```bash
   git clone git@github.com:zkzkGamal/Agentic-AI-Tutorial.git
   cd Agentic-AI-Tutorial
   ```

   #### https

   ```bash
   git clone https://github.com/zkzkGamal/Agentic-AI-Tutorial.git
   cd Agentic-AI-Tutorial
   ```

2. **Create a virtual environment:**

   #### linux

   ```bash
   python3 -m venv chapter_env
   source chapter_env/bin/activate
   ```

   #### windows

   ```bash
   python -m venv chapter_env
   chapter_env\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your API keys:**
   Create a `.env` file in the `Chapter1` directory (or copy the provided `.env.example`):
   ```bash
   cp Chapter1/.env.example Chapter1/.env
   ```
   Edit `.env` and add your API keys:
   ```env
   OPENAI_API_KEY="your_openai_key"
   GOOGLE_API_KEY="your_google_key"
   OLLAMA_BASE_URL="http://localhost:11434"  # Optional, for local Ollama
   ```

## 📚 Tutorial Structure

### Chapter 1: LLM Fundamentals ✅

- [x] Understanding Large Language Models
- [x] Working with OpenAI, Google Gemini, and Ollama
- [x] Prompt Engineering basics
- [x] Handling responses and streaming
- [Read Chapter 1 Summary](./Chapter1/Chapter1.md)

### Chapter 2: Building Chains

- Sequential Chains
- Router Chains
- Custom Chains with LangChain Expression Language (LCEL)

### Chapter 3: Memory and Context

- ConversationBufferMemory
- Entity Memory
- Vector Stores and RAG (Retrieval-Augmented Generation)

### Chapter 4: Introduction to Agents

- What makes an agent? (Reasoning + Action)
- Tools and Toolkits
- ReAct (Reason + Act) pattern

### Chapter 5: Advanced Agents with LangGraph

- Building stateful, multi-agent systems
- Conditional edges and loops
- Checkpointing and persistence

### Chapter 6: Real-World Applications

- Customer Support Bots
- Data Analysis Agents
- Autonomous Research Agents

## 🛠️ Tools Used

- **LangChain**: Core orchestration framework
- **LangGraph**: For building stateful, multi-agent workflows
- **OpenAI**: GPT-4o, GPT-4o-mini
- **Google Gemini**: Gemini 2.0 Flash, Gemini 1.5 Pro
- **Ollama**: Local LLMs (Llama 3, Mistral, etc.)
- **FAISS**: Vector store for RAG
- **Sentence Transformers**: Local embeddings

## 🤝 Contributing

Contributions are welcome! Whether it's fixing typos, adding new examples, or suggesting improvements, please feel free to open an issue or submit a pull request.
