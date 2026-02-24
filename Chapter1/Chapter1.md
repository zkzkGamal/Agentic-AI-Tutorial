# Chapter 1: LLM Foundations & API Interaction

Welcome to the first chapter of the Agentic AI Tutorial! In this lesson, we lay the groundwork for building autonomous systems by mastering the basic interaction with Large Language Models (LLMs).

---

## 🎯 Lesson Objectives

By the end of this chapter, you will be able to:

- Understand the core architecture of modern LLMs (Transformers).
- Programmatically call **Ollama** (Local), **OpenAI**, and **Google Gemini** using Python.
- Implement **streaming** for real-time response generation.
- Use **System Instructions** to define AI "Personas" and control model behavior.

---

## 🧠 Theoretical Background

### What exactly is an LLM?

At its heart, an LLM is a complex statistical model trained on massive amounts of text.

- **Next-Token Prediction**: The model predicts the most likely next word (or token) based on the preceding text.
- **The Transformer**: Modern LLMs use the Transformer architecture, which uses **Self-Attention** to weigh the importance of different words in a sentence, regardless of their position.

### Local vs. Cloud Models

| Feature     | Local (Ollama)                    | Cloud (OpenAI/Gemini)            |
| :---------- | :-------------------------------- | :------------------------------- |
| **Cost**    | Free (Unlimited)                  | Pay-per-token                    |
| **Privacy** | High (Data stays on your machine) | Low (Data processed by provider) |
| **Speed**   | Depends on your GPU/RAM           | Usually high (Scalable)          |
| **Setup**   | Requires installation             | Requires API Key                 |

---

## 💻 Implementation Guide

### 🛠️ Environment Setup

1. **Initialize your workspace:**

   ```bash
   python3 -m venv chapter_env
   source chapter_env/bin/activate
   ```

2. **Install core dependencies:**

   ```bash
   pip install openai google-genai ollama python-dotenv
   ```

3. **Secure your Keys:**
   Create a `.env` file and add your credentials:
   ```env
   OPENAI_API_KEY="sk-..."
   GOOGLE_API_KEY="AIza..."
   ```

---

## 🤖 Hands-On: Calling the Models

### 1. Ollama (The Privacy King)

Ollama allows you to run powerful models like Llama 3 or Gemma locally.

```python
import ollama

response = ollama.chat(
    model="gemma3:270m",
    messages=[{"role": "user", "content": "Explain gravity to a 5-year-old."}]
)
print(response['message']['content'])
```

### 2. OpenAI (The Industry Standard)

Using `gpt-4o-mini` for high-speed, high-reasoning tasks.

```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Why is the sky blue?"}]
)
print(response.choices[0].message.content)
```

### 3. Google Gemini (The Performance Leader)

Gemini 2.0 Flash offers incredible speed and a massive context window.

```python
from google import genai
import os

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Tell me a space fact."
)
print(response.text)
```

---

## 🎨 Advanced Techniques

### 🌊 Streaming Responses

Don't make your users wait! Streaming allows you to show text as it's being generated.

```python
# Streaming with OpenAI
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a long story about a robot."}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### 🎭 System Prompts (Personas)

You can force the model to adopt a specific tone or role using the `system` role.

- **System Prompt**: _"You are a sarcastic pirate who loves coding."_
- **User Prompt**: _"How do I fix a bug?"_
- **Result**: _"Arrr! Ye scurvy dog, ye forgot a semicolon in yer treasure chest!"_

---

## 🏁 Summary & Next Steps

In this chapter, we learned how to "talk" to AI programmatically. However, calling raw APIs is just the beginning.

**In Chapter 2**, we will introduce **LangChain**, a framework that helps us chain these calls together, manage complex prompts, and build truly autonomous workflows.

---

_Created with ❤️ by the Zkzk_
