# Chapter 2: LangChain Orchestration & LCEL

In Chapter 1, we learned how to call LLMs directly. In this chapter, we level up by using **LangChain**, the industry-standard framework for building complex, production-ready AI applications.

---

## 🎯 Lesson Objectives

By the end of this chapter, you will be able to:

- Use **ChatPromptTemplate** to create reusable and structured prompts.
- Master **LCEL (LangChain Expression Language)** for declarative chain building.
- Implement **Persistent Memory** to maintain context over long conversations.
- Define and bind **Custom Tools** to extend your LLM's capabilities.
- Build advanced **Sequential** and **Router** chains.

---

## 🔗 What is LCEL?

**LangChain Expression Language (LCEL)** is a declarative way to compose chains. It uses the Unix-style pipe operator (`|`) to flow data from one component to another.

### The Basic Flow:

```text
Prompt | LLM | OutputParser
```

1. **Prompt**: Takes raw input and formats it into a prompt.
2. **LLM**: Takes the prompt and generates a response.
3. **OutputParser**: Takes the model's raw output and turns it into a useful string or JSON.

---

## 💻 Technical Setup

Ensure you have your virtual environment active and the following packages installed:

```bash
pip install langchain langchain-openai langchain-google-genai langchain-ollama langchain-community
```

---

## 🛠️ Core Components

### 1. ChatPromptTemplate

Stop hardcoding strings! Use templates with placeholders.

```python
from langchain_core.prompts.chat import ChatPromptTemplate

template = ChatPromptTemplate(messages=[
    ("system", "You are a helpful assistant named {name}."),
    ("human", "{user_input}")
])
```

### 2. Building your First LCEL Chain

Here's how we connect the pieces. Notice how easy it is to swap models!

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# The Chain
chain = template | llm | parser

response = chain.invoke({"name": "Jarvis", "user_input": "Hello!"})
print(response)
```

---

## 🧠 Adding Memory

LLMs are stateless by default. To create a chatbot that remembers you, we use `RunnableWithMessageHistory`.

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Wrap your chain with memory
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="user_input",
    history_messages_key="history",
)
```

---

## 🧰 Extending with Tools

Tools are the "hands" of your agent. They allow the model to interact with external code.

```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers together."""
    return a * b

# Bind tools to the model
llm_with_tools = llm.bind_tools([multiply])
```

---

## 🛣️ Advanced Architectures

### Sequential Chains

Step-by-step processing where the output of one step leads to the next.
_Example: Topic → Detailed Explanation → 1-Sentence Summary._

### Router Chains

Branching logic based on intent.
_Example: If input has "math", use math_chain; otherwise, use general_chain._

---

## 🏁 Summary

You've now moved from calling models to **orchestrating systems**. You can manage prompts, maintain memory, and give your AI tools to use.

**In Chapter 3**, we will explore **Advanced Memory** and **RAG (Retrieval-Augmented Generation)** to give your AI access to your private data and documents.

---

_Created with ❤️ by the Agentic AI Team_
