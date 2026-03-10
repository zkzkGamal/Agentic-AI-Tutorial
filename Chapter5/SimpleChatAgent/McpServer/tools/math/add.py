from server import mcp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@mcp.tool()
def add(args: list = None, kwargs: dict = None) -> float:
    """Add any number of arguments."""
    try:
        total = 0.0
        def flatten(item):
            if isinstance(item, (list, tuple)):
                for x in item:
                    yield from flatten(x)
            elif item is not None:
                yield item

        if args:
            clean_args = list(flatten(args))
            total += sum(clean_args)
        if kwargs:
            clean_kwargs = list(flatten(list(kwargs.values())))
            total += sum(clean_kwargs)
        return total
    except Exception as e:
        logger.error(f"Failed to add: {e}")
        return f"Error executing tool add: {e}"
