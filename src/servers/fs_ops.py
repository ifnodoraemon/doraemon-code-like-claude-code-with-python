"""
File operations MCP server.
Provides file manipulation tools (move, copy, delete, rename).
"""

import logging
import os
import shutil

from mcp.server.fastmcp import FastMCP

from src.core.security import validate_path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("PolymathFileSystemOperations")


@mcp.tool()
def move_file(src: str, dst: str) -> str:
    """
    Move a file or directory to a new location.

    Args:
        src: Source path
        dst: Destination path

    Returns:
        Success message or error
    """
    try:
        src_path = validate_path(src)
        dst_path = validate_path(dst)
    except (PermissionError, ValueError) as e:
        logger.warning(f"Path validation failed: {e}")
        return f"Error: {e}"

    if not os.path.exists(src_path):
        logger.warning(f"Source not found: {src_path}")
        return f"Error: Source not found: {src}"

    try:
        # Create destination directory if needed
        dst_dir = os.path.dirname(dst_path)
        if dst_dir:
            os.makedirs(dst_dir, exist_ok=True)

        shutil.move(src_path, dst_path)
        logger.info(f"Moved {src} to {dst}")
        return f"Successfully moved {src} to {dst}"

    except Exception as e:
        logger.error(f"Failed to move {src} to {dst}: {e}")
        return f"Error moving file: {str(e)}"


@mcp.tool()
def copy_file(src: str, dst: str, overwrite: bool = False) -> str:
    """
    Copy a file or directory to a new location.

    Args:
        src: Source path
        dst: Destination path
        overwrite: Whether to overwrite if destination exists

    Returns:
        Success message or error
    """
    src_path = validate_path(src)
    dst_path = validate_path(dst)

    if not os.path.exists(src_path):
        return f"Error: Source not found: {src}"

    if os.path.exists(dst_path) and not overwrite:
        return f"Error: Destination already exists: {dst}. Use overwrite=True to replace."

    try:
        # Create destination directory if needed
        dst_dir = os.path.dirname(dst_path)
        if dst_dir:
            os.makedirs(dst_dir, exist_ok=True)

        if os.path.isdir(src_path):
            # Copy directory
            if os.path.exists(dst_path) and overwrite:
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
        else:
            # Copy file
            shutil.copy2(src_path, dst_path)

        return f"Successfully copied {src} to {dst}"

    except Exception as e:
        return f"Error copying: {str(e)}"


@mcp.tool()
def delete_file(path: str, recursive: bool = False) -> str:
    """
    Delete a file or directory.

    WARNING: This operation is irreversible. Use with caution.

    Args:
        path: Path to delete
        recursive: If True, delete directories and their contents

    Returns:
        Success message or error
    """
    try:
        valid_path = validate_path(path)
    except (PermissionError, ValueError) as e:
        logger.warning(f"Path validation failed for '{path}': {e}")
        return f"Error: {e}"

    if not os.path.exists(valid_path):
        logger.warning(f"Path not found: {valid_path}")
        return f"Error: Path not found: {path}"

    try:
        if os.path.isdir(valid_path):
            if not recursive:
                return f"Error: {path} is a directory. Use recursive=True to delete it."

            shutil.rmtree(valid_path)
            logger.info(f"Deleted directory: {path}")
            return f"Successfully deleted directory: {path}"
        else:
            os.remove(valid_path)
            logger.info(f"Deleted file: {path}")
            return f"Successfully deleted file: {path}"

    except Exception as e:
        logger.error(f"Failed to delete '{path}': {e}")
        return f"Error deleting: {str(e)}"


@mcp.tool()
def rename_file(old_path: str, new_name: str) -> str:
    """
    Rename a file or directory (in the same directory).

    Args:
        old_path: Current file path
        new_name: New name (just the name, not full path)

    Returns:
        Success message or error
    """
    old_valid_path = validate_path(old_path)

    if not os.path.exists(old_valid_path):
        return f"Error: File not found: {old_path}"

    try:
        # Build new path in same directory
        directory = os.path.dirname(old_valid_path)
        new_path = os.path.join(directory, new_name)
        new_valid_path = validate_path(new_path)

        if os.path.exists(new_valid_path):
            return f"Error: A file with name '{new_name}' already exists"

        os.rename(old_valid_path, new_valid_path)
        return f"Successfully renamed {old_path} to {new_name}"

    except Exception as e:
        return f"Error renaming: {str(e)}"


@mcp.tool()
def create_directory(path: str) -> str:
    """
    Create a new directory (and parent directories if needed).

    Args:
        path: Directory path to create

    Returns:
        Success message or error
    """
    valid_path = validate_path(path)

    try:
        os.makedirs(valid_path, exist_ok=True)
        return f"Successfully created directory: {path}"

    except Exception as e:
        return f"Error creating directory: {str(e)}"


if __name__ == "__main__":
    mcp.run()
