# Chapter 2: LangChain Chains, Memory & Tools

> **Course**: Agentic AI Tutorial  
> **Chapter**: 2 of 6  
> **Prerequisites**: Chapter 1 — LLM Fundamentals  
> **Notebook**: [`chapter2-code.ipynb`](./chapter2-code.ipynb)

---

## 🎯 Learning Objectives

By the end of this chapter you will be able to:

- Use **ChatPromptTemplate** to create reusable, parameterised prompts
- Connect prompts to any LLM using the **LCEL pipe (`|`) operator**
- Add **persistent memory** to a chain so it remembers previous turns
- Define and call **custom tools** that extend what an LLM can do
- Build **Sequential Chains** — pipelines where each step feeds the next
- Build **Router Chains** — pipelines that branch based on user intent
- Build **Custom LCEL Chains** that run steps in parallel and merge results

---

## 1. Environment Setup

Before running any code, set up an isolated Python environment for this chapter.

### 1.1 Create & Activate a Virtual Environment

```bash
# Linux / macOS
python3 -m venv chapter_env
source chapter_env/bin/activate

# Windows
python -m venv chapter_env
chapter_env\Scripts\activate
```

### 1.2 Install Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes:

| Package                  | Purpose                          |
| ------------------------ | -------------------------------- |
| `langchain`              | Core orchestration framework     |
| `langchain-openai`       | OpenAI chat model integration    |
| `langchain-google-genai` | Google Gemini integration        |
| `langchain-ollama`       | Local LLM via Ollama             |
| `langchain-community`    | Community tools & chat histories |
| `langchain-core`         | Runnables, prompts, messages     |
| `openai`                 | OpenAI SDK                       |
| `python-dotenv`          | Load API keys from `.env`        |
| `tiktoken`               | Token counting                   |

### 1.3 Configure API Keys

Create a `.env` file inside the `chapter2/` directory:

```env
GOOGLE_API_KEY="your_google_key"
OPENAI_API_KEY="your_openai_key"
```

> **Tip**: Never commit your `.env` file to version control. It is already listed in `.gitignore`.

---

## 2. Importing Libraries

This section imports every module used throughout the chapter.

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import os, dotenv, tiktoken

dotenv.load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY") or ""
openai_api_key = os.getenv("OPENAI_API_KEY") or ""
```

> **Key concept**: `dotenv.load_dotenv()` reads your `.env` file and injects the keys into the process environment so `os.getenv()` can retrieve them safely.

---

## 3. ChatPromptTemplate — Structured Prompts

`ChatPromptTemplate` lets you define a reusable prompt with **placeholders** that are filled in at call time.

### 3.1 Define a Template

```python
template = ChatPromptTemplate(
    messages=[
        ("system", "You are a helpful AI bot. Your name is {name}."),
        ("human",  "{user_input}")
    ]
)
```

### 3.2 Inspect the Rendered Prompt

```python
prompt_value = template.invoke({
    "name": "AI From Ollama",
    "user_input": "hello, what is your name"
})

print(prompt_value.messages)
```

**Output**:

```
[SystemMessage(content='You are a helpful AI bot. Your name is AI From Ollama.'),
 HumanMessage(content='hello , what is your name')]
```

You can iterate over the messages to inspect each one individually:

```python
for msg in prompt_value.messages:
    print(msg)
```

**Output**:

```
content='You are a helpful AI bot. Your name is AI From Ollama.'
content='hello , what is your name'
```

---

## 4. Building Your First Chain (Prompt → LLM)

The **pipe operator `|`** connects components into a chain. Data flows left to right.

### 4.1 With Ollama (Local LLM)

```python
llm = ChatOllama(
    model="gemma3:270m",
    timeout=30,
    base_url="http://127.0.0.1:11434",
    use_mmap=True,
)

chain = template | llm

response = chain.invoke({
    "name": "AI From Ollama",
    "user_input": "hello, what is your name?"
})

print(response.content)
```

**Output**:

```
I am Gemma, an open-weights AI model created by the Gemma team.
```

### 4.2 With OpenAI

```python
openai_llm = ChatOpenAI(name="gpt-4o-mini", temperature=0.5, api_key=openai_api_key)

template_openai = ChatPromptTemplate(
    messages=[
        ("system", "You are a helpful AI bot. Your name is {name}."),
        ("human",  "{user_input}")
    ]
)

openai_chain = template_openai | openai_llm

response = openai_chain.invoke({
    "name": "AI From Ollama",
    "user_input": "hello, what is your name?"
})

print(response.content)
```

**Output**:

```
Hello! My name is AI From Ollama. How can I assist you today?
```

> **Key concept**: The same chain structure works with any LLM — swap `ChatOllama` for `ChatOpenAI` and the rest stays the same. This is the power of LCEL's unified interface.

---

## 5. Adding Memory to a Chain

By default, each `chain.invoke()` call is stateless — the LLM does not remember previous messages. We use `RunnableWithMessageHistory` to add **session-based memory**.

### How It Works

```
User message
     ↓
Retrieve history from store (keyed by session_id)
     ↓
Prepend history to prompt → LLM → Response
     ↓
Append new (user, AI) pair back to store
```

### 5.1 Prompt with History Placeholder

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot. Your name is {name}."),
    MessagesPlaceholder(variable_name="history"),   # 👈 chat history goes here
    ("human", "{user_input}")
])
```

### 5.2 In-Memory Session Store

```python
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
```

### 5.3 Wrap the Chain with Memory

```python
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="user_input",
    history_messages_key="history",
)
```

### 5.4 Multi-Turn Conversation (Ollama)

```python
# Turn 1 — introduce name
response1 = chain_with_memory.invoke(
    {"name": "AI From Ollama", "user_input": "Hello!, my name is zkzk"},
    config={"configurable": {"session_id": "user1"}}
)
print(response1.content)
```

**Output**: `Hello! I'm zkzk, your friendly AI assistant. How can I help you today?`

```python
# Turn 2 — ask something that needs previous context
response2 = chain_with_memory.invoke(
    {"name": "AI From Ollama", "user_input": "What is your name?"},
    config={"configurable": {"session_id": "user1"}}
)
print(response2.content)
```

**Output**: `I am Gemma, an open-weights AI model.`

> **Note**: The `session_id` key is what separates different users or conversations. Use a unique value per conversation.

### 5.5 Memory with OpenAI

The exact same pattern applies to OpenAI — just wrap `openai_chain` instead:

```python
store_open_ai = {}

def get_session_history_openai(session_id: str):
    if session_id not in store_open_ai:
        store_open_ai[session_id] = ChatMessageHistory()
    return store_open_ai[session_id]

chain_with_memory_openai = RunnableWithMessageHistory(
    openai_chain,
    get_session_history_openai,
    input_messages_key="user_input",
    history_messages_key="history",
)

response1 = chain_with_memory_openai.invoke(
    {"name": "AI From Ollama", "user_input": "Hello!, my name is zkzk"},
    config={"configurable": {"session_id": "user1"}}
)
print(response1.content)
```

**Output**: `Hello zkzk! How can I assist you today?`

---

## 6. Tool Calling — Extending the LLM

Tools are Python functions that the LLM can **decide to call** when they are needed. You decorate a function with `@tool` and bind it to a model.

### 6.1 Define and Bind a Tool

```python
@tool
def add(a, b):
    """This is a tool to add two numbers a and b and return their sum."""
    return a + b

openai_llm_with_tools = openai_llm.bind_tools([add])
```

### 6.2 Step 1 — Ask the LLM (It Returns a Tool Call)

```python
response = openai_llm_with_tools.invoke([
    SystemMessage(content="You are a helpful AI bot. You have access to an add tool."),
    HumanMessage(content="Can you add 5 and 3 for me?")
])

print(response.tool_calls)
```

**Output**:

```python
[{'name': 'add', 'args': {'a': 5, 'b': 3}, 'id': 'call_rItK4q9FW...', 'type': 'tool_call'}]
```

### 6.3 Step 2 — Execute the Tool & Return the Result

```python
if response.tool_calls:
    tool_call = response.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    tool_id   = tool_call["id"]

    # Execute the tool
    result = add.invoke(tool_args)
    print(f"Tool result: {result}")   # → Tool result: 8

    # Send the result back to the LLM so it can respond naturally
    final_response = openai_llm_with_tools.invoke([
        SystemMessage(content="You are a helpful AI bot. You have access to an add tool."),
        HumanMessage(content="Can you add 5 and 3 for me?"),
        response,                                    # the AI's tool-call message
        ToolMessage(content=str(result), tool_call_id=tool_id)
    ])

    print(final_response.content)
```

**Output**:

```
Tool result: 8
The sum of 5 and 3 is 8.
```

> **Key concept**: The full tool-calling loop is:  
> `User → LLM (decides to call tool) → Tool executes → Result sent back → LLM produces final answer`

---

## 7. Sequential Chains — Step-by-Step Pipelines

**Use when**: the output of one step must become the input of the next step.

### Example Flow

```
topic → [explain prompt → LLM] → explanation → [summarize prompt → LLM] → one-sentence summary
```

### 7.1 Build the Two-Step Chain (Ollama)

```python
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="gemma3:270m", timeout=30, base_url="http://127.0.0.1:11434", use_mmap=True)
parser = StrOutputParser()

explain_prompt   = ChatPromptTemplate.from_template("Explain the following topic in detail:\n{topic}")
summarize_prompt = ChatPromptTemplate.from_template("Summarize this in one sentence:\n{explanation}")

chain = (
    explain_prompt
    | llm                                                  # Step 1: generate explanation
    | parser                                               # parse AIMessage → plain string
    | (lambda explanation: {"explanation": explanation})   # wrap in dict for next prompt
    | summarize_prompt
    | llm                                                  # Step 2: summarize
    | parser
)

response = chain.invoke({"topic": "Artificial Intelligence"})
print(response)
```

**Output**:

```
AI is a broad field encompassing computer learning, problem-solving, understanding,
creativity, and automation. It's crucial to understand its potential, the challenges
it faces, and the ethical considerations surrounding its development and deployment.
```

### 7.2 Same Chain with OpenAI

Simply replace `llm` with `openai_llm`:

```python
chain_openai = (
    explain_prompt
    | openai_llm
    | parser
    | (lambda explanation: {"explanation": explanation})
    | summarize_prompt
    | openai_llm
    | parser
)

response = chain_openai.invoke({"topic": "Artificial Intelligence"})
print(response)
```

**Output**:

```
Artificial Intelligence is a branch of computer science focused on creating machines
that can perform tasks requiring human intelligence, with different types and subfields
such as machine learning, natural language processing, and robotics, used in various
industries for applications like virtual assistants and medical diagnosis, while also
raising ethical concerns regarding job displacement and bias in algorithms.
```

---

## 8. Router Chains — Dynamic Routing Based on Intent

**Use when**: different types of input require different prompts or models.

### Example Flow

```
User input → condition check → math_chain  (if "add" or "solve" found)
                             ↘ writing_chain (otherwise — default)
```

### 8.1 Define the Sub-Chains (Ollama)

```python
from langchain_core.runnables.branch import RunnableBranch

math_chain = (
    ChatPromptTemplate.from_template("Solve this math problem:\n{input}")
    | llm
    | parser
)

writing_chain = (
    ChatPromptTemplate.from_template("Write a short creative paragraph about:\n{input}")
    | llm
    | parser
)
```

### 8.2 Create and Invoke the Router

```python
router = RunnableBranch(
    (
        lambda x: "add" in x["input"] or "solve" in x["input"],   # condition
        math_chain,                                                 # route to math
    ),
    writing_chain,  # default route
)

print(router.invoke({"input": "Solve 5 + 3"}))
print(router.invoke({"input": "Write about space exploration"}))
```

**Output**:

```
The answer to 5 + 3 is 8.

In the vast expanse of space, the universe whispers secrets untold...
```

### 8.3 Router with OpenAI

```python
math_chain = (
    ChatPromptTemplate.from_template("Solve this math problem:\n{input}")
    | openai_llm | parser
)
writing_chain = (
    ChatPromptTemplate.from_template("Write a short creative paragraph about:\n{input}")
    | openai_llm | parser
)

router = RunnableBranch(
    (lambda x: "add" in x["input"] or "solve" in x["input"], math_chain),
    writing_chain,
)

print(router.invoke({"input": "Solve 5 + 3"}))
print(router.invoke({"input": "Write about space exploration"}))
```

> **Key concept**: `RunnableBranch` takes a list of `(condition, chain)` tuples plus a default chain at the end. The first condition that evaluates to `True` wins.

---

## 9. Custom Chains with LCEL — Parallel & Merged Outputs

**LCEL (LangChain Expression Language)** is the composability layer that makes LangChain chains declarative and powerful.

### LCEL lets you:

1. **Parallelize** — run multiple chains on the same input simultaneously
2. **Transform** — pass data through arbitrary Python functions with `RunnableLambda`
3. **Merge** — combine results from parallel branches into one output
4. **Compose** — build arbitrarily complex pipelines from small, testable pieces

### 9.1 `RunnableParallel` — Run Two Chains Simultaneously (Ollama)

```python
from langchain_core.runnables import RunnableParallel, RunnableLambda

joke_chain = (
    ChatPromptTemplate.from_template("Tell a joke about {topic}") | llm | parser
)

serious_chain = (
    ChatPromptTemplate.from_template("Explain seriously what {topic} is") | llm | parser
)

parallel_chain = RunnableParallel(
    joke=joke_chain,
    serious=serious_chain
)

merge = RunnableLambda(
    lambda x: f"JOKE:\n{x['joke']}\n\nSERIOUS:\n{x['serious']}"
)

final_chain = parallel_chain | merge

print(final_chain.invoke({"topic": "AI Agents"}))
```

**Output**:

```
JOKE:
Why did the AI agent get fired? Because it couldn't handle the challenge!

SERIOUS:
AI Agents are a type of machine learning model designed to take a given task as input
and perform a specific action or set of actions...
```

### 9.2 `RunnableParallel` — Summarize & Explain Simultaneously (OpenAI)

```python
summarize_chain = (
    ChatPromptTemplate.from_template("summarize this {topic}") | openai_llm | parser
)

main_chain = (
    ChatPromptTemplate.from_template("Explain what {topic} is") | openai_llm | parser
)

parallel_chain = RunnableParallel(summarize=summarize_chain, main=main_chain)

merge = RunnableLambda(
    lambda x: f"Summarize:\n{x['summarize']}\n\nMAIN:\n{x['main']}"
)

final_chain = parallel_chain | merge
print(final_chain.invoke({"topic": "AI Agents"}))
```

**Output**:

```
Summarize:
AI agents are software programs designed to act autonomously, make decisions, and carry
out tasks in a way that mimics human intelligence...

MAIN:
AI agents are software programs or algorithms that have the ability to perceive their
environment, make decisions, and take actions to achieve a specific goal...
```

> **Key concept**: `RunnableParallel` executes both branches **concurrently** — so the total time is roughly `max(branch1_time, branch2_time)` rather than the sum.

---

## 10. Chapter Summary

Here is everything you covered in this chapter:

| Concept             | Class / Function                           | What It Does                                         |
| ------------------- | ------------------------------------------ | ---------------------------------------------------- |
| Structured prompts  | `ChatPromptTemplate`                       | Reusable prompts with named placeholders             |
| Chain composition   | `\|` (pipe operator)                       | Connect prompt → LLM → parser in one line            |
| Session memory      | `RunnableWithMessageHistory`               | Persist chat history across turns                    |
| In-memory history   | `ChatMessageHistory`                       | Simple dict-backed message store                     |
| Tool definition     | `@tool` decorator                          | Turn a Python function into an LLM-callable tool     |
| Tool binding        | `llm.bind_tools([...])`                    | Register tools with a model                          |
| Sequential pipeline | `prompt \| llm \| parser \| lambda \| ...` | Multi-step chains, output of each step feeds next    |
| Dynamic routing     | `RunnableBranch`                           | Condition-based chain selection                      |
| Parallel execution  | `RunnableParallel`                         | Run multiple chains on same input simultaneously     |
| Output merging      | `RunnableLambda`                           | Combine and transform outputs with a Python function |

---

## 📝 Exercises

1. **Template Practice**: Create a `ChatPromptTemplate` with placeholders for `language` and `code_snippet`. Invoke it with a Python snippet and ask the LLM to explain it.

2. **Memory Challenge**: Build a memory-enabled chain that remembers the user's **favorite programming language** and uses it in subsequent responses.

3. **Custom Tool**: Create a `multiply` tool that multiplies two numbers. Bind it alongside `add` and test the LLM's ability to choose the right tool.

4. **Long Sequential Chain**: Build a 3-step sequential chain:  
   `topic → explain → translate to French → summarize the French text`

5. **Smart Router**: Extend the router to handle three routes: `math`, `creative writing`, and `code generation` using keyword-based condition functions.

6. **LCEL Parallel**: Create a parallel chain that generates a poem AND bullet-point facts about a given topic at the same time, then merges them into a single formatted output.

---

## 🔗 What's Next

In **Chapter 3: Memory and Context**, you will go beyond simple in-memory chat history and explore:

- `ConversationBufferMemory` and `ConversationSummaryMemory`
- Entity-aware memory that tracks people, places, and things
- **Vector Stores** and **RAG (Retrieval-Augmented Generation)** — giving your agent access to external knowledge bases

---

_For all code examples, refer to [`chapter2-code.ipynb`](./chapter2-code.ipynb)._
