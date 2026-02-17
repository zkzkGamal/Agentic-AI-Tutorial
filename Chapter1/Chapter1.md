# Chapter 1: Calling LLM Models

## 📚 Overview

In this chapter, we explored the foundational concepts of **Large Language Models (LLMs)** and learned how to interact with them using Python. We moved from understanding theory to practical implementation by calling three major LLM providers: **Ollama (Local)**, **OpenAI**, and **Google Gemini**.

## 🧠 Key Concepts Learned

### 1. What is an LLM?

- **Definition**: Artificial intelligence models trained on massive datasets to understand and generate human-like text.
- **Core Function**: They are essentially "next-token predictors" that calculate the statistical probability of the next word.
- **Scale**: "Large" refers to both **Parameters** (memory knobs, billions/trillions) and **Training Data** (trillions of tokens).

### 2. The Transformer Architecture

- The backbone of modern LLMs (introduced in "Attention Is All You Need", 2017).
- **Key Innovation**: The **Attention Mechanism**, allowing models to focus on relevant parts of input regardless of distance.
- **Context Window**: The amount of text a model can process at once (now up to 2M+ tokens).

## 💻 Code Implementation

We created a Python environment and implemented the following:

### 🛠️ Setup

1.  Created a virtual environment (`chapter_env`).
2.  Installed dependencies: `openai`, `google-genai`, `ollama`, `python-dotenv`.
3.  Loaded API keys securely using `.env`.

### 🤖 Models Used

#### 1. Ollama (Local) 🦙

- **Model**: `gemma3:270m` (lightweight local model).
- **Usage**: Great for privacy, no cost, and offline capability.
- **Code Snippet**:
  ```python
  response = ollama.chat(model="gemma3:270m", messages=[{"role": "user", "content": "why the sky blue"}])
  ```

#### 2. OpenAI (Cloud) ☁️

- **Model**: `gpt-4o-mini` (efficient, high-performance).
- **Usage**: Industry standard for reasoning and general tasks.
- **Code Snippet**:
  ```python
  response = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[{"role": "user", "content": "why the sky is blue"}]
  )
  ```

#### 3. Google Gemini (Cloud) ✨

- **Model**: `gemini-2.5-flash` (fast, large context).
- **Usage**: Strong multimodal capabilities and speed.
- **Code Snippet**:
  ```python
  response = client.models.generate_content(model="gemini-2.5-flash", contents="why the sky blue")
  ```

## ⚖️ Comparison & Techniques

### Streaming vs. Non-Streaming

- **Non-Streaming**: Waits for the entire response to be generated before showing it. (Good for simple tasks).
- **Streaming**: Displays the response chunk-by-chunk as it's generated. (Better user experience for long text).

### System Instructions (Personas)

We learned how to shape the model's behavior using **System Prompts**.

- **Example**: _"You are a zkzk AI assistant who answers in short, punchy sentences and uses plenty of emojis"_
- **Result**: The models adopted a specific personality, proving we can control _how_ they answer, not just _what_ they answer.

## 🔜 Next Steps: Chapter 2

Now that we can call raw LLMs, we need to build more complex workflows.

### 🔗 LangChain

We will learn about **LangChain**, a framework that helps us:

- Chain multiple LLM calls together.
- Manage prompt templates efficiently.
- Connect LLMs to external data sources.

### 🎨 Prompt Engineering

We will dive deeper into the art of crafting inputs to get the best possible outputs:

- Techniques for better reasoning (Chain-of-Thought).
- Structuring data for consistent results.

---

_Created by zkzkAgent_
