from langchain.tools import tool
from mcp import ClientSession  # MCP client
from mcp.client.sse import sse_client
import environ , pathlib , logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_path = pathlib.Path(__file__).parent.parent
e = environ.Env()
e.read_env(str(base_path / ".env"))

MCP_SERVER_URL = e('MCP_SERVER_URL', default='http://localhost:8000')

@tool()
async def send_email(subject: str, body: str, to_email: list[str]):
    """
        Send an email.
        args:
            subject: str
            body: str
            to_email: list[str]
        returns:
            bool
    """
    try:
        async with sse_client(f"{MCP_SERVER_URL.rstrip('/')}/sse") as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await session.call_tool("send_email", arguments={"subject": subject, "body": body, "to_email": to_email})
        logger.info(f"Email sent successfully to {to_email}.")
        return True
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False

@tool()
async def check_inbox(limit: int = 5) -> list:
    """Fetch recent emails from inbox. Returns list of dicts with subject, from, body snippet."""
    try:
        async with sse_client(f"{MCP_SERVER_URL.rstrip('/')}/sse") as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                emails = await session.call_tool("check_inbox", arguments={"limit": limit})
        logger.info(f"Fetched {len(emails)} emails from inbox.")
        return emails
    except Exception as e:
        logger.error(f"Failed to check inbox: {e}")
        return []

@tool()
async def read_email(email_id: str) -> dict:
    """Read a specific email by its ID."""
    try:
        async with sse_client(f"{MCP_SERVER_URL.rstrip('/')}/sse") as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                email = await session.call_tool("read_email", arguments={"email_id": email_id})
        logger.info(f"Read email with ID: {email_id}")
        return email
    except Exception as e:
        logger.error(f"Failed to read email: {e}")
        return {}

@tool()
async def reply_to_email(email_id: str, body: str) -> bool:
    """Reply to a specific email by its ID."""
    try:
        async with sse_client(f"{MCP_SERVER_URL.rstrip('/')}/sse") as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await session.call_tool("reply_to_email", arguments={"email_id": email_id, "body": body})
        logger.info(f"Replied to email with ID: {email_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to reply to email: {e}")
        return False
