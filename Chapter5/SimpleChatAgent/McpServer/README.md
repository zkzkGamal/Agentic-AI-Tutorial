# 🔌 FastMCP Server

This directory contains the **Model Context Protocol (MCP)** server for Chapter 5. It acts as the secure execution environment (the "Hands") for our agent, completely decoupled from the LangGraph logic (the "Brain").

## 📁 Project Structure

```text
McpServer/
├── main.py                 # The server entry point (uvicorn & FastMCP)
├── server.py               # The MCP server instance config
└── tools/                  # The directory containing all MCP tools
    ├── __init__.py         # Tool registry/imports
    ├── .env                # Private credentials (not pushed to git!)
    ├── emails/             # Email-specific capabilities
    │   ├── __init__.py
    │   ├── check_inbox.py
    │   ├── read_email.py
    │   ├── reply_to_email.py
    │   └── send_mail.py
    └── math/               # Math-specific capabilities
        ├── __init__.py
        ├── add.py
        └── multiply.py
```

## 🎯 What we built here

We used the `mcp` (FastMCP) Python package to expose Python functions as standardized tools over **Server-Sent Events (SSE)**. 

### Current Tools:
1. **Math Tools** (`tools/math/`): Supports dynamic addition and multiplication.
2. **Email Tools** (`tools/emails/`): Connects to SMTP/IMAP to securely send, read, check, and reply to emails.

## 🛠️ Deep Dive: How it works

Unlike traditional agents where the tool executing logic is bundled inside the agent's code, the FastMCP server separates concerns:
1. **The Server (`server.py`)**: Declares an MCP instance.
2. **The Tools**: Any Python function decorated with `@mcp.tool()` is automatically parsed by the server. The MCP server reads the function arguments (e.g. `subject: str, to_email: list[str]`) and docstrings, and converts them into an OpenAPI-style JSON schema. 
3. **The Transport (`main.py`)**: The server runs on an ASGI server (`uvicorn`) and listens on port `8000`. 
4. **Execution**: When the `AganticAssistant` connects, it asks the server "What tools do you have?". The server replies with the JSON schema. When the LLM decides to use a tool, the Assistant sends the arguments to port 8000, the Server executes the underlying Python, and streams the result back over SSE.

## 📈 Tutorial: Adding Your Own Tools
Because of the MCP standard, extending the agent's capabilities is incredibly easy! You don't need to touch the LangChain agent at all.

1. Create a new file in `tools/` (e.g. `tools/weather/get_weather.py`).
2. Write a standard Python function and decorate it with `@mcp.tool()`. It is **critical** to use type hints and docstrings—this is how the LLM knows how to use your tool!
   ```python
   from server import mcp

   @mcp.tool()
   def get_temperature(city: str) -> str:
       """Get the current temperature for a given city."""
       return f"The weather in {city} is 72°F."
   ```
3. Import your new tool in `tools/__init__.py` to register it.
   ```python
   # inside tools/__init__.py
   from .weather.get_weather import get_temperature
   ```
4. Restart the server! The `AganticAssistant` will automatically discover the new tool on startup and the LLM will know how to use it immediately.
   ```python
   from server import mcp

   @mcp.tool()
   def get_temperature(city: str) -> str:
       """Get the current temperature for a city."""
       return f"The weather in {city} is 72°F."
   ```
3. Import your new tool in `tools/__init__.py`.
   ```python
   # inside tools/__init__.py
   from .weather.get_weather import get_temperature
   ```
4. Restart the server! The `AganticAssistant` will automatically discover the new tool on startup and the LLM will know how to use it.
