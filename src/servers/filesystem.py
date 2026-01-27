"""
Doraemon Unified Filesystem MCP Server

A comprehensive filesystem server that combines all file operations:
- Reading: read_file, read_file_outline, list_directory, list_directory_tree, glob_files, grep_search, find_symbol
- Writing: write_file
- Editing: edit_file, edit_file_multiline
- Operations: move_file, copy_file, delete_file, rename_file, create_directory

This replaces the previous fs_read.py, fs_write.py, fs_edit.py, and fs_ops.py.
"""

import fnmatch
import glob as glob_module
import itertools
import logging
import os
import re
import shutil
from datetime import datetime

from mcp.server.fastmcp import FastMCP

from src.core.security import validate_path
from src.services import code_nav, document, outline, vision

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("DoraemonFilesystem")


# ========================================
# Reading Tools
# ========================================


@mcp.tool()
def read_file_outline(path: str) -> str:
    """
    Read the structural outline of a file (Classes, Functions) to understand code without reading the whole content.
    Highly recommended for exploring new files.

    Args:
        path: Path to the file (Python, JS, TS, etc.)
    """
    try:
        valid_path = validate_path(path)
    except (PermissionError, ValueError) as e:
        logger.warning(f"Path validation failed for '{path}': {e}")
        return f"Error: {e}"

    if not os.path.exists(valid_path):
        logger.warning(f"File not found: {valid_path}")
        return "Error: File not found."

    logger.debug(f"Reading outline of: {valid_path}")
    return outline.parse_outline(valid_path)


@mcp.tool()
def list_directory(path: str = ".", show_hidden: bool = False, detailed: bool = True) -> str:
    """
    List files and directories at the given path.

    Args:
        path: Directory path to list
        show_hidden: Include hidden files (starting with .)
        detailed: Show detailed metadata (size, modified time, type)

    Returns:
        Formatted directory listing
    """
    valid_path = validate_path(path)
    if not os.path.exists(valid_path):
        return "Error: Path not found."

    try:
        entries = []

        for item in sorted(os.listdir(valid_path)):
            # Skip hidden files if requested
            if item.startswith(".") and not show_hidden:
                continue

            full_path = os.path.join(valid_path, item)

            if detailed:
                try:
                    stat = os.stat(full_path)

                    # Format size
                    size = _human_size(stat.st_size)

                    # Modified time
                    mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")

                    # Type
                    if os.path.isdir(full_path):
                        ftype = "dir"
                        size = "<DIR>"
                    else:
                        ftype = "file"

                    entries.append(f"{item:<40} {size:>10} {mtime} [{ftype}]")
                except OSError:
                    # Fallback if stat fails
                    entries.append(item)
            else:
                entries.append(item)

        if not entries:
            return "(empty directory)"

        return "\n".join(entries)

    except Exception as e:
        return f"Error: {e}"


def _human_size(bytes_size):
    """Convert bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}PB"


@mcp.tool()
def read_file(path: str, offset: int = 0, limit: int | None = None, encoding: str = "utf-8") -> str:
    """
    Intelligently read a file with optional partial reading.
    Supports: .txt, .md, .py, .pdf, .docx, .pptx, .xlsx, .png, .jpg

    Args:
        path: File path to read
        offset: Starting line number (0-based, for text files only)
        limit: Maximum number of lines to read (for text files only)
        encoding: Text encoding (default: utf-8)

    Returns:
        File content or error message
    """
    valid_path = validate_path(path)
    if not os.path.exists(valid_path):
        return "Error: File not found."

    ext = os.path.splitext(path)[1].lower()

    try:
        # Document formats (no offset/limit support)
        if ext == ".pdf":
            return document.parse_pdf(valid_path)
        elif ext == ".docx":
            return document.parse_docx(valid_path)
        elif ext == ".pptx":
            return document.parse_pptx(valid_path)
        elif ext in [".xlsx", ".xls"]:
            return document.parse_xlsx(valid_path)

        # Image formats
        elif ext in [".png", ".jpg", ".jpeg", ".webp"]:
            return vision.process_image(valid_path)

        # Text formats (support offset/limit)
        else:
            if offset == 0 and limit is None:
                with open(valid_path, encoding=encoding, errors="replace") as f:
                    return f.read()

            with open(valid_path, encoding=encoding, errors="replace") as f:
                # Use islice to efficiently skip to offset and read 'limit' lines
                iterator = itertools.islice(f, offset, offset + limit if limit else None)
                selected_lines = list(iterator)

                if not selected_lines:
                    return f"No lines found at offset {offset}."

                result = f"[Lines {offset + 1}-{offset + len(selected_lines)}]\n\n"
                result += "".join(selected_lines)

                if limit and len(selected_lines) == limit:
                    result += "\n\n[... (more lines may exist)]"

                return result

    except Exception as e:
        return f"Error reading file: {str(e)}"


@mcp.tool()
def glob_files(pattern: str, exclude: list[str] | None = None, max_results: int = 1000) -> str:
    """
    Find files matching a glob pattern.

    Supports wildcards:
    - * matches any characters
    - ** matches any directories (recursive)
    - ? matches single character
    - [abc] matches one character in brackets

    Args:
        pattern: Glob pattern (e.g., "**/*.py", "src/**/test_*.py")
        exclude: List of patterns to exclude (e.g., ["**/node_modules/**", "**/__pycache__/**"])
        max_results: Maximum number of files to return (default: 1000)

    Examples:
        glob_files("**/*.py")  # All Python files recursively
        glob_files("src/**/*.js")  # All JS files in src/
        glob_files("*.{py,js}")  # All .py and .js files in current dir

    Returns:
        Newline-separated list of matching file paths
    """
    try:
        # Security check: prevent path traversal in pattern
        if '..' in pattern:
            return "Error: Pattern cannot contain '..' for security reasons."

        # Validate pattern doesn't start with absolute path or escape workspace
        if pattern.startswith('/') or pattern.startswith('~'):
            return "Error: Pattern cannot be an absolute path."

        # Find matching files
        matches = glob_module.glob(pattern, recursive=True)

        # Filter out results that are outside the workspace
        validated_matches = []
        for match in matches:
            try:
                # Validate each matched path is within workspace
                validate_path(match)
                validated_matches.append(match)
            except (PermissionError, ValueError):
                # Skip paths outside workspace
                continue
        matches = validated_matches

        # Apply exclusions
        if exclude:
            filtered = []
            for match in matches:
                # Check if match should be excluded
                should_exclude = False
                for excl_pattern in exclude:
                    if fnmatch.fnmatch(match, excl_pattern):
                        should_exclude = True
                        break

                if not should_exclude:
                    filtered.append(match)

            matches = filtered

        # Limit results
        if len(matches) > max_results:
            matches = matches[:max_results]
            truncated = True
        else:
            truncated = False

        # Sort for consistent output
        matches = sorted(matches)

        if not matches:
            return f"No files found matching pattern: {pattern}"

        result = "\n".join(matches)

        if truncated:
            result += f"\n\n[Showing first {max_results} of {len(matches)} matches]"
        else:
            result = f"Found {len(matches)} file(s):\n\n" + result

        return result

    except Exception as e:
        return f"Error in glob search: {str(e)}"


@mcp.tool()
def find_symbol(symbol: str, path: str = ".") -> str:
    """
    Find the DEFINITION of a class or function (Semantic Search).
    Use this instead of grep when you want to know "Where is class X defined?".

    Args:
        symbol: The name of the class or function (e.g., "MultiServerMCPClient")
        path: Root directory to search (default: current dir)
    """
    valid_path = validate_path(path)
    return code_nav.find_definition(valid_path, symbol)


@mcp.tool()
def grep_search(pattern: str, include: str = "*", path: str = ".") -> str:
    """
    Search for a text pattern within files (recursive).
    Similar to 'grep -r'. Useful for finding where a variable or function is defined.

    Args:
        pattern: Regex pattern to search for
        include: Glob pattern for files to include (e.g. "*.py")
        path: Directory to search in
    """
    valid_path = validate_path(path)
    results = []

    try:
        regex = re.compile(pattern)

        for root, _, files in os.walk(valid_path):
            for file in files:
                if not fnmatch.fnmatch(file, include):
                    continue

                full_path = os.path.join(root, file)
                try:
                    with open(full_path, encoding="utf-8", errors="ignore") as f:
                        for i, line in enumerate(f, 1):
                            if regex.search(line):
                                # Format: path/to/file:line_num:content
                                rel_path = os.path.relpath(full_path, valid_path)
                                results.append(f"{rel_path}:{i}:{line.strip()}")

                                if len(results) >= 100:  # Hard limit to prevent massive outputs
                                    results.append("... (limit reached)")
                                    return "\n".join(results)
                except Exception:
                    # Skip binary or unreadable files
                    continue

        if not results:
            return f"No matches found for '{pattern}'."

        return "\n".join(results)

    except Exception as e:
        return f"Error searching: {str(e)}"


@mcp.tool()
def list_directory_tree(path: str = ".", depth: int = 2) -> str:
    """
    Show a recursive directory tree to understand project structure.

    Args:
        path: Root directory
        depth: Maximum recursion depth (max: 10)
    """
    # Limit depth to prevent excessive recursion
    depth = min(max(1, depth), 10)
    valid_path = validate_path(path)

    def get_tree(current_path, current_depth):
        if current_depth > depth:
            return ""

        try:
            items = sorted(os.listdir(current_path))
        except Exception:
            return ""

        tree = ""
        for item in items:
            if item.startswith("."):
                continue

            full_path = os.path.join(current_path, item)
            is_dir = os.path.isdir(full_path)

            indent = "  " * (depth - current_depth)
            tree += f"{indent}├── {item}{'/' if is_dir else ''}\n"

            if is_dir:
                tree += get_tree(full_path, current_depth + 1)
        return tree

    tree_output = get_tree(valid_path, 1)
    return (
        f"Project Tree for {path}:\n{tree_output}"
        if tree_output
        else "Empty or inaccessible directory."
    )


# ========================================
# Writing Tools
# ========================================


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


# ========================================
# Editing Tools
# ========================================


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


# ========================================
# File Operation Tools
# ========================================


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
