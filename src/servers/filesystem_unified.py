"""
Unified Filesystem Tools - Consolidated MCP Server

This module provides 3 unified tools that replace 15+ scattered tools:
- read: Unified reading (file, outline, directory, tree)
- write: Unified writing (create, edit, delete, move, copy)
- search: Unified searching (content, files, symbol)

Design Philosophy:
- Occam's Razor: Fewer, well-designed tools
- Single Responsibility: Each tool has one clear purpose
- Functional Cohesion: Related operations grouped together
- Parameterized Design: Use parameters instead of multiple tools

This is the single source of truth for filesystem operations.
All legacy modules (filesystem_read.py, etc.) have been consolidated here.
"""

import fnmatch
import glob as glob_module
import itertools
import logging
import os
import re
import shutil
from datetime import datetime
from typing import Literal

from mcp.server.fastmcp import FastMCP

from src.core.logger import configure_root_logger
from src.core.security import validate_path
from src.services import code_nav, document, outline, vision

# Setup logging
configure_root_logger()
logger = logging.getLogger(__name__)

mcp = FastMCP("AgentFilesystemUnified")


# ========================================
# Helper Functions
# ========================================


def _human_size(bytes_size: int) -> str:
    """Convert bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}PB"


# ========================================
# Core Read Functions (Internal)
# ========================================


def read_file(path: str, offset: int = 0, limit: int | None = None, encoding: str = "utf-8") -> str:
    """
    Intelligently read a file with optional partial reading.
    Supports: .txt, .md, .py, .pdf, .docx, .pptx, .xlsx, .png, .jpg
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


def read_file_outline(path: str) -> str:
    """Read the structural outline of a file (Classes, Functions)."""
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


def list_directory(path: str = ".", show_hidden: bool = False, detailed: bool = True) -> str:
    """List files and directories at the given path."""
    valid_path = validate_path(path)
    if not os.path.exists(valid_path):
        return "Error: Path not found."

    try:
        entries = []

        for item in sorted(os.listdir(valid_path)):
            if item.startswith(".") and not show_hidden:
                continue

            full_path = os.path.join(valid_path, item)

            if detailed:
                try:
                    stat = os.stat(full_path)
                    size = _human_size(stat.st_size)
                    mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")

                    if os.path.isdir(full_path):
                        ftype = "dir"
                        size = "<DIR>"
                    else:
                        ftype = "file"

                    entries.append(f"{item:<40} {size:>10} {mtime} [{ftype}]")
                except OSError:
                    entries.append(item)
            else:
                entries.append(item)

        if not entries:
            return "(empty directory)"

        return "\n".join(entries)

    except Exception as e:
        return f"Error: {e}"


def list_directory_tree(path: str = ".", depth: int = 2) -> str:
    """Show a recursive directory tree."""
    depth = min(max(1, depth), 10)
    valid_path = validate_path(path)

    def get_tree(current_path: str, current_depth: int) -> str:
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


def find_symbol(symbol: str, path: str = ".") -> str:
    """Find the DEFINITION of a class or function (Semantic Search)."""
    valid_path = validate_path(path)
    return code_nav.find_definition(valid_path, symbol)


# ========================================
# Core Search Functions (Internal)
# ========================================


def glob_files(pattern: str, exclude: list[str] | None = None, max_results: int = 1000) -> str:
    """Find files matching a glob pattern."""
    try:
        if ".." in pattern:
            return "Error: Pattern cannot contain '..' for security reasons."

        if pattern.startswith("/") or pattern.startswith("~"):
            return "Error: Pattern cannot be an absolute path."

        matches = glob_module.glob(pattern, recursive=True)

        validated_matches = []
        for match in matches:
            try:
                validate_path(match)
                validated_matches.append(match)
            except (PermissionError, ValueError):
                continue
        matches = validated_matches

        if exclude:
            filtered = []
            for match in matches:
                should_exclude = False
                for excl_pattern in exclude:
                    if fnmatch.fnmatch(match, excl_pattern):
                        should_exclude = True
                        break
                if not should_exclude:
                    filtered.append(match)
            matches = filtered

        if len(matches) > max_results:
            total_count = len(matches)
            matches = matches[:max_results]
            truncated = True
        else:
            total_count = len(matches)
            truncated = False

        matches = sorted(matches)

        if not matches:
            return f"No files found matching pattern: {pattern}"

        result = "\n".join(matches)

        if truncated:
            result += f"\n\n[Showing first {max_results} of {total_count} matches]"
        else:
            result = f"Found {len(matches)} file(s):\n\n" + result

        return result

    except Exception as e:
        return f"Error in glob search: {str(e)}"


def grep_search(pattern: str, include: str = "*", path: str = ".") -> str:
    """Search for a text pattern within files (recursive)."""
    valid_path = validate_path(path)
    results = []

    # Directories to skip during recursive search
    skip_dirs = {
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        "dist",
        "build",
        ".eggs",
        ".egg-info",
    }
    # Max file size to search (10 MB)
    max_file_size = 10 * 1024 * 1024

    try:
        regex = re.compile(pattern)

        for root, dirs, files in os.walk(valid_path):
            # Prune skipped directories in-place
            dirs[:] = [d for d in dirs if d not in skip_dirs]

            for file in files:
                if not fnmatch.fnmatch(file, include):
                    continue

                full_path = os.path.join(root, file)
                try:
                    # Skip files that are too large
                    if os.path.getsize(full_path) > max_file_size:
                        continue

                    with open(full_path, encoding="utf-8", errors="ignore") as f:
                        for i, line in enumerate(f, 1):
                            if regex.search(line):
                                rel_path = os.path.relpath(full_path, valid_path)
                                results.append(f"{rel_path}:{i}:{line.strip()}")

                                if len(results) >= 100:
                                    results.append("... (limit reached)")
                                    return "\n".join(results)
                except Exception:
                    continue

        if not results:
            return f"No matches found for '{pattern}'."

        return "\n".join(results)

    except Exception as e:
        return f"Error searching: {str(e)}"


# ========================================
# Core Write Functions (Internal)
# ========================================


def write_file(path: str, content: str) -> str:
    """Write text content to a file."""
    try:
        valid_path = validate_path(path)
    except (PermissionError, ValueError) as e:
        logger.warning(f"Path validation failed for '{path}': {e}")
        return f"Error: {e}"

    try:
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


def edit_file(path: str, old_string: str, new_string: str, count: int = -1) -> str:
    """Edit a file by replacing specific content."""
    try:
        valid_path = validate_path(path)
    except (PermissionError, ValueError) as e:
        logger.warning(f"Path validation failed for '{path}': {e}")
        return f"Error: {e}"

    if not os.path.exists(valid_path):
        logger.warning(f"File not found: {valid_path}")
        return f"Error: File not found: {path}"

    try:
        with open(valid_path, encoding="utf-8") as f:
            content = f.read()

        if old_string not in content:
            logger.debug(f"Search string not found in {path}")
            return f"Error: Search string not found in {path}:\n'{old_string[:100]}...'"

        occurrences = content.count(old_string)

        if count == -1:
            new_content = content.replace(old_string, new_string)
            replaced_count = occurrences
        else:
            new_content = content.replace(old_string, new_string, count)
            replaced_count = min(count, occurrences)

        with open(valid_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        logger.info(f"Edited {path}: {replaced_count} replacement(s)")
        return f"Successfully edited {path} ({replaced_count} replacement(s) made)"

    except Exception as e:
        logger.error(f"Failed to edit file '{path}': {e}")
        return f"Error editing file: {str(e)}"


def edit_file_multiline(path: str, edits: list[dict]) -> str:
    """Apply multiple search/replace edits in sequence."""
    valid_path = validate_path(path)

    if not os.path.exists(valid_path):
        return f"Error: File not found: {path}"

    try:
        with open(valid_path, encoding="utf-8") as f:
            content = f.read()

        successful_edits = 0

        for i, edit in enumerate(edits):
            old_str = edit.get("old_string")
            new_str = edit.get("new_string")

            if not old_str or new_str is None:
                return f"Error: Edit #{i + 1} missing 'old_string' or 'new_string'"

            if old_str not in content:
                return f"Error: Edit #{i + 1} search string not found:\n'{old_str[:100]}...'"

            content = content.replace(old_str, new_str, 1)
            successful_edits += 1

        with open(valid_path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"Successfully applied {successful_edits} edit(s) to {path}"

    except Exception as e:
        return f"Error applying edits: {str(e)}"


# ========================================
# Core File Operations (Internal)
# ========================================


def move_file(src: str, dst: str) -> str:
    """Move a file or directory to a new location."""
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
        dst_dir = os.path.dirname(dst_path)
        if dst_dir:
            os.makedirs(dst_dir, exist_ok=True)

        shutil.move(src_path, dst_path)
        logger.info(f"Moved {src} to {dst}")
        return f"Successfully moved {src} to {dst}"

    except Exception as e:
        logger.error(f"Failed to move {src} to {dst}: {e}")
        return f"Error moving file: {str(e)}"


def copy_file(src: str, dst: str, overwrite: bool = False) -> str:
    """Copy a file or directory to a new location."""
    src_path = validate_path(src)
    dst_path = validate_path(dst)

    if not os.path.exists(src_path):
        return f"Error: Source not found: {src}"

    if os.path.exists(dst_path) and not overwrite:
        return f"Error: Destination already exists: {dst}. Use overwrite=True to replace."

    try:
        dst_dir = os.path.dirname(dst_path)
        if dst_dir:
            os.makedirs(dst_dir, exist_ok=True)

        if os.path.isdir(src_path):
            if os.path.exists(dst_path) and overwrite:
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)

        return f"Successfully copied {src} to {dst}"

    except Exception as e:
        return f"Error copying: {str(e)}"


def delete_file(path: str, recursive: bool = False) -> str:
    """Delete a file or directory."""
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


def rename_file(old_path: str, new_name: str) -> str:
    """Rename a file or directory (in the same directory)."""
    old_valid_path = validate_path(old_path)

    if not os.path.exists(old_valid_path):
        return f"Error: File not found: {old_path}"

    try:
        directory = os.path.dirname(old_valid_path)
        new_path = os.path.join(directory, new_name)
        new_valid_path = validate_path(new_path)

        if os.path.exists(new_valid_path):
            return f"Error: A file with name '{new_name}' already exists"

        os.rename(old_valid_path, new_valid_path)
        return f"Successfully renamed {old_path} to {new_name}"

    except Exception as e:
        return f"Error renaming: {str(e)}"


def create_directory(path: str) -> str:
    """Create a new directory (and parent directories if needed)."""
    valid_path = validate_path(path)

    try:
        os.makedirs(valid_path, exist_ok=True)
        return f"Successfully created directory: {path}"

    except Exception as e:
        return f"Error creating directory: {str(e)}"


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
                return create_directory(path)
            else:
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
# Notebook Operations (Claude Code Compatible)
# ========================================


@mcp.tool()
def notebook_read(path: str) -> str:
    """
    Read a Jupyter notebook (.ipynb) file.

    Returns all cells with their types, source code, and outputs.

    Args:
        path: Path to the notebook file

    Returns:
        Formatted notebook content with cell numbers, types, and outputs
    """
    import json

    try:
        valid_path = validate_path(path)
        if not os.path.exists(valid_path):
            return "Error: Notebook file not found."

        if not path.endswith(".ipynb"):
            return "Error: File must be a Jupyter notebook (.ipynb)"

        with open(valid_path, encoding="utf-8") as f:
            notebook = json.load(f)

        cells = notebook.get("cells", [])
        if not cells:
            return "Notebook is empty."

        result = []
        for i, cell in enumerate(cells):
            cell_type = cell.get("cell_type", "unknown")
            source = "".join(cell.get("source", []))

            result.append(f"=== Cell {i} ({cell_type}) ===")
            result.append(source)

            # Include outputs for code cells
            if cell_type == "code":
                outputs = cell.get("outputs", [])
                if outputs:
                    result.append("\n--- Output ---")
                    for output in outputs:
                        output_type = output.get("output_type", "")
                        if output_type == "stream":
                            text = "".join(output.get("text", []))
                            result.append(text)
                        elif output_type in ("execute_result", "display_data"):
                            data = output.get("data", {})
                            if "text/plain" in data:
                                result.append("".join(data["text/plain"]))
                        elif output_type == "error":
                            result.append(
                                f"Error: {output.get('ename', '')}: {output.get('evalue', '')}"
                            )

            result.append("")

        return "\n".join(result)

    except json.JSONDecodeError:
        return "Error: Invalid notebook format (not valid JSON)"
    except Exception as e:
        return f"Error reading notebook: {str(e)}"


@mcp.tool()
def notebook_edit(
    path: str,
    cell_index: int,
    new_source: str,
    cell_type: str | None = None,
    operation: Literal["replace", "insert", "delete"] = "replace",
) -> str:
    """
    Edit a Jupyter notebook cell.

    Args:
        path: Path to the notebook file
        cell_index: Index of the cell to edit (0-based)
        new_source: New source code for the cell
        cell_type: Cell type ("code" or "markdown"), only for insert
        operation: "replace" (default), "insert", or "delete"

    Returns:
        Success message or error
    """
    import json

    try:
        valid_path = validate_path(path)
        if not os.path.exists(valid_path):
            return "Error: Notebook file not found."

        with open(valid_path, encoding="utf-8") as f:
            notebook = json.load(f)

        cells = notebook.get("cells", [])

        if operation == "delete":
            if cell_index < 0 or cell_index >= len(cells):
                return f"Error: Cell index {cell_index} out of range (0-{len(cells) - 1})"
            del cells[cell_index]
            notebook["cells"] = cells

        elif operation == "insert":
            if cell_index < 0 or cell_index > len(cells):
                return f"Error: Cell index {cell_index} out of range for insert (0-{len(cells)})"
            new_cell = {
                "cell_type": cell_type or "code",
                "source": new_source.splitlines(keepends=True),
                "metadata": {},
            }
            if cell_type == "code" or cell_type is None:
                new_cell["outputs"] = []
                new_cell["execution_count"] = None
            cells.insert(cell_index, new_cell)
            notebook["cells"] = cells

        elif operation == "replace":
            if cell_index < 0 or cell_index >= len(cells):
                return f"Error: Cell index {cell_index} out of range (0-{len(cells) - 1})"
            cells[cell_index]["source"] = new_source.split("\n")
            # Clear outputs for code cells
            if cells[cell_index].get("cell_type") == "code":
                cells[cell_index]["outputs"] = []
                cells[cell_index]["execution_count"] = None

        else:
            return f"Error: Invalid operation '{operation}'"

        with open(valid_path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=1)

        return f"Successfully {operation}d cell {cell_index} in {path}"

    except json.JSONDecodeError:
        return "Error: Invalid notebook format"
    except Exception as e:
        return f"Error editing notebook: {str(e)}"


@mcp.tool()
def multi_edit(
    path: str,
    edits: list[dict],
) -> str:
    """
    Make multiple sequential edits to a single file.

    Each edit is applied in order. If any edit fails, the operation stops
    and the file is left in its state after the last successful edit.

    Args:
        path: Path to the file to edit
        edits: List of edit operations, each with:
            - old_string: Text to find and replace
            - new_string: Replacement text

    Returns:
        Summary of edits made or error message

    Example:
        multi_edit("main.py", [
            {"old_string": "foo", "new_string": "bar"},
            {"old_string": "baz", "new_string": "qux"}
        ])
    """
    try:
        valid_path = validate_path(path)
        if not os.path.exists(valid_path):
            return "Error: File not found."

        with open(valid_path, encoding="utf-8") as f:
            content = f.read()

        original_content = content
        successful_edits = 0

        for i, edit in enumerate(edits):
            old_string = edit.get("old_string", "")
            new_string = edit.get("new_string", "")

            if not old_string:
                return f"Error: Edit {i} missing 'old_string'"

            if old_string not in content:
                # Rollback and report error
                with open(valid_path, "w", encoding="utf-8") as f:
                    f.write(original_content)
                return f"Error: Edit {i} failed - 'old_string' not found in file. Rolled back {successful_edits} edits."

            # Check for uniqueness
            count = content.count(old_string)
            if count > 1:
                with open(valid_path, "w", encoding="utf-8") as f:
                    f.write(original_content)
                return f"Error: Edit {i} failed - 'old_string' appears {count} times (must be unique). Rolled back."

            content = content.replace(old_string, new_string, 1)
            successful_edits += 1

        # Write final result
        with open(valid_path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"Successfully applied {successful_edits} edits to {path}"

    except Exception as e:
        return f"Error in multi_edit: {str(e)}"


# ========================================
# Server Entry Point
# ========================================

if __name__ == "__main__":
    mcp.run()
