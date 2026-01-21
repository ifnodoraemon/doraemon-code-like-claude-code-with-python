import logging
import os

from mcp.server.fastmcp import FastMCP

from src.core.security import validate_path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("PolymathFileSystemWriter")


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Write text content to a file."""
    try:
        valid_path = validate_path(path)
    except (PermissionError, ValueError) as e:
        logger.warning(f"Path validation failed for '{path}': {e}")
        return f"Error: {e}"

    try:
        # Create parent directories if needed
        parent_dir = os.path.dirname(valid_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        with open(valid_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Successfully wrote {len(content)} bytes to {path}")
        return f"Successfully wrote to {path}"
    except Exception as e:
        logger.error(f"Failed to write file '{path}': {e}")
        return f"Error writing file: {str(e)}"


if __name__ == "__main__":
    mcp.run()
