"""
Precise file editing MCP server.
Provides search/replace editing similar to Claude Code's Edit tool.
"""

import logging
import os

from mcp.server.fastmcp import FastMCP

from src.core.security import validate_path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("PolymathFileSystemEditor")


@mcp.tool()
def edit_file(path: str, old_string: str, new_string: str, count: int = -1) -> str:
    """
    Edit a file by replacing specific content with search/replace pattern.

    This is more efficient than rewriting entire files. The search string
    must match exactly (including whitespace and indentation).

    Args:
        path: File path to edit
        old_string: Exact text to search for (must match exactly)
        new_string: Text to replace with
        count: Number of occurrences to replace (-1 = all, 1 = first only)

    Returns:
        Success message or error description
    """
    try:
        valid_path = validate_path(path)
    except (PermissionError, ValueError) as e:
        logger.warning(f"Path validation failed for '{path}': {e}")
        return f"Error: {e}"

    if not os.path.exists(valid_path):
        logger.warning(f"File not found: {valid_path}")
        return f"Error: File not found: {path}"

    try:
        # Read current content
        with open(valid_path, encoding="utf-8") as f:
            content = f.read()

        # Check if search string exists
        if old_string not in content:
            logger.debug(f"Search string not found in {path}")
            return f"Error: Search string not found in {path}:\n'{old_string[:100]}...'"

        # Count occurrences
        occurrences = content.count(old_string)

        if occurrences == 0:
            return f"Error: Search string not found in {path}"

        # Perform replacement
        if count == -1:
            # Replace all occurrences
            new_content = content.replace(old_string, new_string)
            replaced_count = occurrences
        else:
            # Replace specific number of occurrences
            new_content = content.replace(old_string, new_string, count)
            replaced_count = min(count, occurrences)

        # Write back
        with open(valid_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        logger.info(f"Edited {path}: {replaced_count} replacement(s)")
        return f"Successfully edited {path} ({replaced_count} replacement(s) made)"

    except Exception as e:
        logger.error(f"Failed to edit file '{path}': {e}")
        return f"Error editing file: {str(e)}"


@mcp.tool()
def edit_file_multiline(path: str, edits: list[dict]) -> str:
    """
    Apply multiple search/replace edits in sequence.

    Useful for making several related changes in one operation.
    Each edit is applied in order, so later edits see the results of earlier ones.

    Args:
        path: File path to edit
        edits: List of edit operations, each with 'old_string' and 'new_string'

    Example:
        edits = [
            {"old_string": "DEBUG = False", "new_string": "DEBUG = True"},
            {"old_string": "PORT = 8000", "new_string": "PORT = 3000"}
        ]

    Returns:
        Success message with number of edits applied
    """
    valid_path = validate_path(path)

    if not os.path.exists(valid_path):
        return f"Error: File not found: {path}"

    try:
        # Read current content
        with open(valid_path, encoding="utf-8") as f:
            content = f.read()

        successful_edits = 0

        # Apply each edit in sequence
        for i, edit in enumerate(edits):
            old_str = edit.get("old_string")
            new_str = edit.get("new_string")

            if not old_str or new_str is None:
                return f"Error: Edit #{i + 1} missing 'old_string' or 'new_string'"

            if old_str not in content:
                return f"Error: Edit #{i + 1} search string not found:\n'{old_str[:100]}...'"

            # Apply the edit
            content = content.replace(old_str, new_str, 1)  # Replace first occurrence
            successful_edits += 1

        # Write back
        with open(valid_path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"Successfully applied {successful_edits} edit(s) to {path}"

    except Exception as e:
        return f"Error applying edits: {str(e)}"


if __name__ == "__main__":
    mcp.run()
