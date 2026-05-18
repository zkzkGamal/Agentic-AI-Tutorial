import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "AganticAssistant"))
sys.path.insert(1, str(ROOT / "McpServer"))

for module_name in list(sys.modules):
    if module_name == "tools" or module_name.startswith("tools."):
        sys.modules.pop(module_name)

from langchain_core.messages import HumanMessage

import importlib

router_module = importlib.import_module("agent.nodes.router")
summarize_module = importlib.import_module("agent.nodes.summarize")


class DummyResponse:
    def __init__(self, content):
        self.content = content


class DummyLLM:
    def __init__(self, response_content):
        self._response_content = response_content

    def invoke(self, messages):
        return DummyResponse(self._response_content)


def test_router_returns_math_intent():
    state = {"messages": [SimpleNamespace(content="Please add 3 and 4")]}  # noqa: E501

    with patch.object(router_module, "_build_llm", return_value=DummyLLM("math")):
        result = router_module.router(state)

    assert result == {"intent": "math"}


def test_router_fallbacks_to_conversation_for_unknown_labels():
    state = {"messages": [SimpleNamespace(content="Tell me a story")]}  # noqa: E501

    with patch.object(router_module, "_build_llm", return_value=DummyLLM("unknown")):
        result = router_module.router(state)

    assert result == {"intent": "conversation"}


def test_summarize_appends_ai_message():
    state = {
        "messages": [HumanMessage(content="What is the answer?")],
        "tool_results": "Result: 42",
    }

    with patch.object(summarize_module, "_build_llm", return_value=DummyLLM("The result is 42.")):
        result = summarize_module.summarize(state)

    assert result["summary"] == "The result is 42."
    assert result["messages"][-1].content == "The result is 42."
