import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MCP_SERVER = ROOT / "McpServer"

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

add_module = _load_module("mcp_add", MCP_SERVER / "tools" / "math" / "add.py")
multiply_module = _load_module("mcp_multiply", MCP_SERVER / "tools" / "math" / "multiply.py")

add = add_module.add
multiply = multiply_module.multiply

def test_add_flatten_and_kwargs():
    assert add(args=[1, [2, 3], (4,)], kwargs={"x": 5, "y": [6, 7]}) == 28.0


def test_add_handles_empty_values():
    assert add(args=None, kwargs={}) == 0.0


def test_multiply_flatten_and_kwargs():
    assert multiply(args=[2, [3, 2]], kwargs={"x": 5}) == 60.0


def test_multiply_handles_no_args_returns_one():
    assert multiply(args=None, kwargs=None) == 1.0
