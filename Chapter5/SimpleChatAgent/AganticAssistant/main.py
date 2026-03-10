"""
main.py — Interactive CLI for the Multi-Node Agentic Assistant
===============================================================
Run:
    cd Chapter5/SimpleChatAgent/AganticAssistant
    python main.py

The agent routes each message through:
    router → execute → summarize   (math / email requests)
    router → conversation          (general chat)

A fixed thread_id keeps conversation memory across turns (MemorySaver).
"""

import asyncio
import uuid
import logging
from langchain_core.messages import HumanMessage

from agent.graph import graph

logging.basicConfig(
    level=logging.WARNING,          # hide internal LangChain noise by default
    format="%(levelname)s | %(name)s | %(message)s",
)

THREAD_ID = str(uuid.uuid4())       # unique session ID — shared across all turns


async def chat_loop() -> None:
    
    print("\n🤖  Agentic Assistant  (type 'exit' to quit)")
    print("────────────────────────────────────────────")
    print("  I can help with math, email, or general questions!\n")

    config = {"configurable": {"thread_id": THREAD_ID}}

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        # Build initial state for this turn
        state = {
            "messages": [HumanMessage(content=user_input)],
            "intent": None,
            "tool_results": None,
            "summary": None,
        }

        try:
            result = await graph.ainvoke(state, config=config)
        except Exception as exc:
            print(f"[error] {exc}\n")
            continue

        # The last message added is the assistant's reply
        messages = result.get("messages", [])
        if messages:
            last = messages[-1]
            reply = last.content if hasattr(last, "content") else str(last)
        else:
            reply = result.get("summary") or "I'm not sure how to respond."

        intent = result.get("intent", "?")
        print(f"\nAssistant [{intent}]: {reply}\n")


if __name__ == "__main__":
    asyncio.run(chat_loop())
