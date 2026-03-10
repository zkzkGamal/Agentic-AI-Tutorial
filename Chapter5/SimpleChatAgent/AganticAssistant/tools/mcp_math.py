from langchain.tools import tool
from mcp.client.sse import sse_client
from mcp import ClientSession  # MCP client
import environ , pathlib , logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_path = pathlib.Path(__file__).parent.parent
e = environ.Env()
e.read_env(str(base_path / ".env"))

MCP_SERVER_URL = e('MCP_SERVER_URL', default='http://localhost:8000')

@tool()
async def add(*args , **kwargs) -> int:
    """Add any number of arguments."""
    try:
        async with sse_client(f"{MCP_SERVER_URL.rstrip('/')}/sse") as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("add", arguments={"args": args, "kwargs": kwargs} if (args or kwargs) else kwargs or {})
        logger.info(f"Added {args} and {kwargs} to get {result}.")
        return result
    except Exception as e:
        logger.error(f"Failed to add {args} and {kwargs}: {e}")
        return 0

@tool()
async def multiply(*args , **kwargs) -> int:
    """Multiply any number of arguments."""
    try:
        async with sse_client(f"{MCP_SERVER_URL.rstrip('/')}/sse") as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("multiply", arguments={"args": args, "kwargs": kwargs} if (args or kwargs) else kwargs or {})
        logger.info(f"Multiplied {args} and {kwargs} to get {result}.")
        return result
    except Exception as e:
        logger.error(f"Failed to multiply {args} and {kwargs}: {e}")
        return 0
