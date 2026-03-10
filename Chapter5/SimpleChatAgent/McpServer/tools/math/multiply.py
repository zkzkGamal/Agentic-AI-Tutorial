from server import mcp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import math

@mcp.tool()
def multiply(args: list = None, kwargs: dict = None) -> float:
    """Multiply any number of arguments."""
    try:
        res = 1.0

        def flatten(item):
            if isinstance(item, (list, tuple)):
                for x in item:
                    yield from flatten(x)
            elif item is not None:
                yield item

        if args:
            clean_args = list(flatten(args))
            if clean_args:
                res *= math.prod(clean_args)
        if kwargs:
            clean_kwargs = list(flatten(list(kwargs.values())))
            if clean_kwargs:
                res *= math.prod(clean_kwargs)
        return res
    except Exception as e:
        logger.error(f"Failed to multiply: {e}")
        return f"Error executing tool multiply: {e}"