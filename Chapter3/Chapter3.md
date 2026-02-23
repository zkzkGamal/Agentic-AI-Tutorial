# Chapter 3: Memory, Entity Memory & RAG in LangChain

> **Course**: Agentic AI Tutorial  
> **Chapter**: 3 of 6  
> **Prerequisites**: Chapter 2 — LangChain Chains, Memory & Tools  
> **Notebook**: [`chapter3-code.ipynb`](./chapter3-code.ipynb)

---

## 🎯 Learning Objectives

By the end of this chapter you will be able to:

- Understand the difference between **stateless** and **stateful** LLM conversations
- Use **`ConversationBufferMemory`** to preserve the full conversation history across turns
- Use **`ConversationEntityMemory`** to automatically extract and track named entities (people, places, things)
- Build a **RAG (Retrieval-Augmented Generation)** pipeline using a **Chroma vector store** and **HuggingFace embeddings**
- Apply all three strategies with both **Ollama (local)** and **OpenAI (cloud)** models

---

## 1. Environment Setup

### 1.1 Create and Activate a Virtual Environment

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

New packages added in this chapter on top of Chapter 2:

| Package                 | Purpose                                                   |
| ----------------------- | --------------------------------------------------------- |
| `langchain-classic`     | Legacy chain classes (`ConversationChain`, `RetrievalQA`) |
| `langchain-chroma`      | Chroma vector store integration                           |
| `sentence-transformers` | Local HuggingFace embedding model                         |
| `transformers`          | HuggingFace model backbone                                |
| `chromadb`              | Chroma vector database engine                             |

### 1.3 Configure API Keys

Create a `.env` file in the `Chapter3/` directory:

```env
GOOGLE_API_KEY="your_google_key"
OPENAI_API_KEY="your_openai_key"
```

---

## 2. LangChain Setup and Imports

```python
from langchain_classic.chains.conversation.base import ConversationChain
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_classic.text_splitter import CharacterTextSplitter
from langchain_classic.memory import ConversationBufferMemory, ConversationEntityMemory
from langchain_classic.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain_community.llms.openai import OpenAI
from langchain_community.llms.ollama import Ollama
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts.chat import PromptTemplate
import os, dotenv

dotenv.load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY") or ""
openai_api_key = os.getenv("OPENAI_API_KEY") or ""
```

### Architecture Overview

```
LangChain Chapter 3
├── Memory
│   ├── ConversationBufferMemory  → stores full chat log as plain text
│   └── ConversationEntityMemory  → extracts and stores named entities via LLM
└── RAG Pipeline
    ├── HuggingFaceEmbeddings     → converts text → vectors (locally, free)
    ├── Chroma                    → vector database: store and retrieve by similarity
    └── RetrievalQA               → retrieves matching docs, feeds them to LLM
```

> **Tip**: Make sure Ollama is running locally before executing Ollama cells.  
> Download from [ollama.com](https://ollama.com) and pull your model:  
> `ollama pull gemma3:270m`

---

## 3. ConversationBufferMemory

### What Is It?

`ConversationBufferMemory` stores the **entire conversation history** as a plain-text buffer. On every new call the full buffer is prepended to the prompt, so the LLM always has the complete context of what was said.

```
Turn 1:  Human: "My name is Zkzk"
         → Buffer: ["Human: My name is Zkzk", "AI: Hi Zkzk!"]

Turn 2:  Human: "What's my name?"
         → LLM receives the buffer above + new question → answers correctly
```

**When to use**: Chat applications that need full recall across all turns.  
**Trade-off**: Buffer grows indefinitely. For long conversations you will eventually hit the model's context window limit — see Chapter's exercises for how to handle this.

---

### 3.1 ConversationBufferMemory with Ollama

```python
llm = Ollama(
    model="gemma3:270m",
    timeout=30,
    temperature=0.7,
)

memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)
```

> `ConversationChain` automatically reads from and writes to `memory` on every `.predict()` call — no extra wiring needed.

**Turn 1 — introduce yourself:**

```python
response1 = conversation.predict(
    input="Hi, I'm Zkzk. What's the capital of Egypt? And can you tell me a joke?"
)
print(response1)
```

**Output**:

```
The capital of Egypt is Cairo.
```

**Turn 2 — test that the model remembers:**

```python
response2 = conversation.predict(input="What's my name?")
print(response2)
```

**Output**:

```
Your name is Zkzk!
```

**Inspect the buffer contents:**

```python
print(memory.buffer)
```

**Output**:

```
Human: Hi, I'm Zkzk. What's the capital of Egypt? And can you tell me a joke?
AI: The capital of Egypt is Cairo.
```
