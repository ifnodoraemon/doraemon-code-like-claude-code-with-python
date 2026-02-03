"""
Unified Filesystem Tools - Simplified MCP Server

This module provides 3 unified tools that replace 15 scattered tools:
- read: Unified reading (file, outline, directory, tree)
- write: Unified writing (create, edit, delete, move, copy)
- search: Unified searching (content, files, symbol)

Design Philosophy:
- Single Responsibility: Each tool has one clear purpose
- Functional Cohesion: Related operations grouped together
- Parameterized Design: Use parameters instead of multiple tools
- MCP Best Practices: Follows Model Context Protocol guidelines
"""

import logging
import os
from typing import Literal

from mcp.server.fastmcp import FastMCP

# Import existing implementations
from src.servers.filesystem import (
    copy_file,
    create_directory,
    delete_file,
    edit_file,
    find_symbol,
    glob_files,
    grep_search,
    list_directory,
    list_directory_tree,
    move_file,
    read_file,
    read_file_outline,
    write_file,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("DoraemonFilesystemUnified")


# ========================================
# Unified Read Tool
# ========================================


@mcp.tool()
def read(
    path: str,
    mode: Literal["file", "outline", "directory", "tree"] = "file",
    offset: int = 0,
    limit: int | None = None,
    depth: int = 2,
    show_hidden: bool = False,
    encoding: str = "utf-8",
) -> str:
    """
    Unified read tool for all file/directory reading operations.

    This tool replaces: read_file, read_file_outline, list_directory, list_directory_tree

    Modes:
    - file: Read file content (supports offset/limit for large files)
    - outline: Get file structure/outline (classes, functions)
    - directory: List directory contents
    - tree: Show directory tree structure

    Args:
        path: File or directory path
        mode: Read mode (default: "file")
        offset: Starting line for file mode (default: 0)
        limit: Max lines to read for file mode (default: None = all)
        depth: Max depth for tree mode (default: 2, max: 10)
        show_hidden: Include hidden files for directory mode (default: False)
        encoding: Text encoding for file mode (default: "utf-8")

    Returns:
        File content, outline, directory listing, or tree structure

    Examples:
        read("main.py")  # Read file
        read("main.py", mode="outline")  # Get structure
        read("src", mode="directory")  # List directory
        read("src", mode="tree", depth=3)  # Show tree
    """
    try:
        if mode == "file":
            return read_file(path, offset=offset, limit=limit, encoding=encoding)
        elif mode == "outline":
            return read_file_outline(path)
        elif mode == "directory":
            return list_directory(path, show_hidden=show_hidden, detailed=True)
        elif mode == "tree":
            return list_directory_tree(path, depth=depth)
        else:
            return f"Error: Invalid mode '{mode}'. Must be one of: file, outline, directory, tree"
    except Exception as e:
        logger.error(f"Read operation failed: {e}", exc_info=True)
        return f"Error in read operation: {str(e)}"


# ========================================
# Unified Write Tool
# ========================================


@mcp.tool()
def write(
    path: str,
    content: str | None = None,
    operation: Literal["create", "edit", "delete", "move", "copy"] = "create",
    old_string: str | None = None,
    new_string: str | None = None,
    count: int = -1,
    destination: str | None = None,
    overwrite: bool = False,
    recursive: bool = False,
) -> str:
    """
    Unified write tool for all file modification operations.

    This tool replaces: write_file, edit_file, delete_file, move_file, copy_file,
                        rename_file, create_directory

    Operations:
    - create: Create new file or directory (if content is None, creates directory)
    - edit: Edit existing file (search/replace)
    - delete: Delete file or directory
    - move: Move file or directory (also used for rename)
    - copy: Copy file or directory

    Args:
        path: Target file or directory path
        content: Content for create operation (None = create directory)
        operation: Write operation type (default: "create")
        old_string: Search string for edit operation
        new_string: Replacement string for edit operation
        count: Number of replacements for edit (-1 = all)
        destination: Destination path for move/copy operations
        overwrite: Allow overwriting for copy operation
        recursive: Recursive delete for directories

    Returns:
        Success message or error description

    Examples:
        write("test.txt", content="hello")  # Create file
        write("test.txt", operation="edit", old_string="hello", new_string="world")  # Edit
        write("test.txt", operation="delete")  # Delete
        write("old.txt", operation="move", destination="new.txt")  # Move/rename
        write("src.txt", operation="copy", destination="dst.txt")  # Copy
        write("mydir", content=None)  # Create directory
    """
    try:
        if operation == "create":
            if content is None:
                # Create directory
                return create_directory(path)
            else:
                # Create file
                return write_file(path, content)

        elif operation == "edit":
            if old_string is None or new_string is None:
                return "Error: edit operation requires both old_string and new_string"
            return edit_file(path, old_string, new_string, count)

        elif operation == "delete":
            return delete_file(path, recursive=recursive)

        elif operation == "move":
            if destination is None:
                return "Error: move operation requires destination parameter"
            return move_file(path, destination)

        elif operation == "copy":
            if destination is None:
                return "Error: copy operation requires destination parameter"
            return copy_file(path, destination, overwrite=overwrite)

        else:
            return f"Error: Invalid operation '{operation}'. Must be one of: create, edit, delete, move, copy"

    except Exception as e:
        logger.error(f"Write operation failed: {e}", exc_info=True)
        return f"Error in write operation: {str(e)}"


# ========================================
# Unified Search Tool
# ========================================


@mcp.tool()
def search(
    query: str,
    mode: Literal["content", "files", "symbol"] = "content",
    path: str = ".",
    include: str = "*",
    exclude: list[str] | None = None,
    max_results: int = 1000,
) -> str:
    """
    Unified search tool for finding files and content.

    This tool replaces: grep_search, glob_files, find_symbol

    Modes:
    - content: Search file contents (grep/regex search)
    - files: Search file names (glob pattern matching)
    - symbol: Search code symbols (semantic search for definitions)

    Args:
        query: Search query (pattern for content/files, symbol name for symbol)
        mode: Search mode (default: "content")
        path: Root directory to search (default: ".")
        include: File pattern to include for content mode (default: "*")
        exclude: Patterns to exclude for files mode
        max_results: Maximum results to return (default: 1000)

    Returns:
        Newline-separated list of matches with context

    Examples:
        search("def main", mode="content")  # Search content
        search("*.py", mode="files")  # Find Python files
        search("MyClass", mode="symbol")  # Find class definition
    """
    try:
        if mode == "content":
            return grep_search(query, include=include, path=path)

        elif mode == "files":
            return glob_files(query, exclude=exclude, max_results=max_results)

        elif mode == "symbol":
            return find_symbol(query, path=path)

        else:
            return f"Error: Invalid mode '{mode}'. Must be one of: content, files, symbol"

    except Exception as e:
        logger.error(f"Search operation failed: {e}", exc_info=True)
        return f"Error in search operation: {str(e)}"


# ========================================
# Server Entry Point
# ========================================

if __name__ == "__main__":
    mcp.run()
