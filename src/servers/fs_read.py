import fnmatch
import glob as glob_module
import itertools
import logging
import os
import re
from datetime import datetime

from mcp.server.fastmcp import FastMCP

from src.core.security import validate_path
from src.services import code_nav, document, outline, vision

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("PolymathFileSystemReader")


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
                # This avoids loading the entire file into memory
                iterator = itertools.islice(f, offset, offset + limit if limit else None)
                selected_lines = list(iterator)

                # We can't easily know total_lines without reading the whole file,
                # but for large files we strictly want to avoid that.
                # So we return what we have.

                if not selected_lines:
                    return f"No lines found at offset {offset}."

                result = f"[Lines {offset + 1}-{offset + len(selected_lines)}]\n\n"
                result += "".join(selected_lines)

                if limit and len(selected_lines) == limit:
                    # Peek if there is more (optimization: simplistic check)
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
        # Find matching files
        matches = glob_module.glob(pattern, recursive=True)

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
        depth: Maximum recursion depth
    """
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


if __name__ == "__main__":
    mcp.run()
