# Chapter 3: Advanced Memory & RAG

In Chapter 2, we mastered orchestration. Now, we dive into the most critical part of production AI: **Knowledge & Context**. This chapter covers how to give your agent a custom memory and access to private documents.

---

## 🎯 Lesson Objectives

By the end of this chapter, you will be able to:

- Understand the difference between **Short-term** (Buffer) and **Deep** (Entity) memory.
- Build a **RAG (Retrieval-Augmented Generation)** pipeline.
- Use **Vector Stores** (Chroma) to store and retrieve private information.
- Use local **HuggingFace Embeddings** for free, privacy-focused vector search.

---

## 🧠 Memory Strategies

### 1. ConversationBufferMemory

The simplest form of memory. It stores the entire chat transcript.

- **Pros**: Full context, easy to implement.
- **Cons**: Grows infinitely, eventually hits model token limits.

### 2. ConversationEntityMemory

A more intelligent approach. It uses an LLM to extract and track specific facts about people, places, and things (entities).

- **Pros**: Focuses on key facts, stays relevant over long interactions.
- **Cons**: Requires additional LLM calls for extraction.

---

## 🏗️ RAG: Retrieval-Augmented Generation

RAG is how we give LLMs access to data they weren't trained on (like your personal notes or company docs).

### The RAG Pipeline Architecture:

1. **Load**: Import text from PDF, CSV, or plain text.
2. **Split**: Break long text into smaller "chunks".
3. **Embed**: Convert text chunks into numerical vectors.
4. **Store**: Save vectors in a **Vector Store** (like Chroma).
5. **Retrieve**: When the user asks a question, find the most similar chunks.
6. **Generate**: Send the chunks + the question to the LLM.

---

## 💻 Tech Implementation

### 🛠️ Environment Setup

```bash
pip install langchain-chroma sentence-transformers transformers chromadb
```

### 🤖 Implementing RAG

```python
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.text_splitter import CharacterTextSplitter

# 1. Prepare Embeddings (Local & Free)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Text Splitting
text = "Agentic AI is the future. It uses reasoning to solve tasks..."
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
docs = text_splitter.create_documents([text])

# 3. Create Vector Store
vector_store = Chroma.from_documents(docs, embeddings)

# 4. Retrieval
query = "What is the future?"
results = vector_store.similarity_search(query)
print(results[0].page_content)
```

---

## ⚖️ When to use RAG vs. Memory?

| Feature       | Memory                     | RAG                         |
| :------------ | :------------------------- | :-------------------------- |
| **Duration**  | Single user session        | Persistent across all users |
| **Data Type** | User-specific chat history | Large databases/documents   |
| **Cost**      | Low                        | Higher (Vector DB storage)  |

---

## 🏁 Summary

You have now completed the intermediate core of the tutorial! You have a powerful toolkit:

1. **Chapter 1**: Raw AI calls.
2. **Chapter 2**: Orchestration & Tools.
3. **Chapter 3**: Custom Knowledge & Context.

**In Chapter 4**, we will finally put it all together to build **Autonomous Agents** that can use these tools and knowledge to solve real-world problems.

---

_Created with ❤️ by the Agentic AI Team_
