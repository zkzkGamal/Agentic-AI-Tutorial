from server import mcp

@mcp.tool()
def draft_reply(original_subject: str, original_body: str, tone: str = "professional") -> str:
    """Generate a draft email reply (uses LLM internally if needed, but simple here)."""
    # Placeholder: In a real setup, call an LLM here via MCP sampling if integrated.
    return f"Re: {original_subject}\n\nThank you for your message. Regarding '{original_body[:50]}...', my response is: [Your reply here]. Best regards."