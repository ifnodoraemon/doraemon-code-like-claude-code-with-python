"""
Async File Operations

Provides non-blocking file operations using asyncio.to_thread.
This module wraps synchronous file operations to prevent blocking
the event loop, improving performance for I/O-heavy operations.
"""

import asyncio
import fnmatch
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Callable

from src.core.security.security import validate_path


async def async_read_path(
    path: str,
    offset: int = 0,
    limit: int | None = None,
    encoding: str = "utf-8",
) -> str:
    """
    Asynchronously read a file.

    Args:
        path: File path
        offset: Line offset to start from
        limit: Maximum number of lines to read
        encoding: File encoding

    Returns:
        File content as string
    """
    valid_path = validate_path(path)

    def _read() -> str:
        if not os.path.exists(valid_path):
            return "Error: File not found."

        try:
            if offset == 0 and limit is None:
                with open(valid_path, encoding=encoding, errors="replace") as f:
                    return f.read()

            with open(valid_path, encoding=encoding, errors="replace") as f:
                import itertools

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

    return await asyncio.to_thread(_read)


async def async_write_path(
    path: str,
    content: str,
    encoding: str = "utf-8",
    mode: str = "w",
) -> str:
    """
    Asynchronously write content to a file.

    Args:
        path: File path
        content: Content to write
        encoding: File encoding
        mode: Write mode ('w' for write, 'a' for append)

    Returns:
        Success message or error
    """
    valid_path = validate_path(path)

    def _write() -> str:
        try:
            Path(valid_path).parent.mkdir(parents=True, exist_ok=True)
            with open(valid_path, mode, encoding=encoding) as f:
                f.write(content)
            return f"Successfully wrote {len(content)} characters to {path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"

    return await asyncio.to_thread(_write)


async def async_list_path(
    path: str = ".",
    show_hidden: bool = False,
    pattern: str | None = None,
) -> list[str]:
    """
    Asynchronously list directory contents.

    Args:
        path: Directory path
        show_hidden: Whether to show hidden files
        pattern: Optional glob pattern to filter

    Returns:
        List of file/directory names
    """
    valid_path = validate_path(path)

    def _list() -> list[str]:
        if not os.path.exists(valid_path):
            return []

        try:
            items = os.listdir(valid_path)
            if not show_hidden:
                items = [i for i in items if not i.startswith(".")]
            if pattern:
                items = [i for i in items if fnmatch.fnmatch(i, pattern)]
            return sorted(items)
        except Exception:
            return []

    return await asyncio.to_thread(_list)


async def async_file_exists(path: str) -> bool:
    """Asynchronously check if a file exists."""
    valid_path = validate_path(path)
    return await asyncio.to_thread(os.path.exists, valid_path)


async def async_get_file_size(path: str) -> int:
    """Asynchronously get file size in bytes."""
    valid_path = validate_path(path)

    def _get_size() -> int:
        try:
            return os.path.getsize(valid_path)
        except Exception:
            return -1

    return await asyncio.to_thread(_get_size)


async def async_copy_path(src: str, dst: str) -> str:
    """Asynchronously copy a file."""
    import shutil

    valid_src = validate_path(src)
    valid_dst = validate_path(dst)

    def _copy() -> str:
        try:
            Path(valid_dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(valid_src, valid_dst)
            return f"Successfully copied {src} to {dst}"
        except Exception as e:
            return f"Error copying file: {str(e)}"

    return await asyncio.to_thread(_copy)


async def async_move_path(src: str, dst: str) -> str:
    """Asynchronously move a file."""
    import shutil

    valid_src = validate_path(src)
    valid_dst = validate_path(dst)

    def _move() -> str:
        try:
            Path(valid_dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(valid_src, valid_dst)
            return f"Successfully moved {src} to {dst}"
        except Exception as e:
            return f"Error moving file: {str(e)}"

    return await asyncio.to_thread(_move)


async def async_delete_path(path: str) -> str:
    """Asynchronously delete a file."""
    valid_path = validate_path(path)

    def _delete() -> str:
        try:
            if os.path.isfile(valid_path):
                os.remove(valid_path)
            elif os.path.isdir(valid_path):
                import shutil

                shutil.rmtree(valid_path)
            return f"Successfully deleted {path}"
        except Exception as e:
            return f"Error deleting: {str(e)}"

    return await asyncio.to_thread(_delete)


async def async_glob_search(
    pattern: str,
    path: str = ".",
    recursive: bool = True,
) -> list[str]:
    """
    Asynchronously search for files matching a glob pattern.

    Args:
        pattern: Glob pattern (e.g., "**/*.py")
        path: Base directory
        recursive: Whether to search recursively

    Returns:
        List of matching file paths
    """
    import glob as glob_module

    valid_path = validate_path(path)

    def _search() -> list[str]:
        try:
            if recursive:
                full_pattern = os.path.join(valid_path, pattern)
            else:
                full_pattern = os.path.join(valid_path, pattern)

            matches = glob_module.glob(full_pattern, recursive=recursive)
            # Return relative paths
            return [os.path.relpath(m, valid_path) for m in matches]
        except Exception:
            return []

    return await asyncio.to_thread(_search)


async def async_grep_search(
    pattern: str,
    path: str = ".",
    file_pattern: str = "*",
    max_results: int = 100,
    ignore_case: bool = True,
) -> list[dict[str, Any]]:
    """
    Asynchronously search for pattern in files.

    Args:
        pattern: Regex pattern to search
        file_pattern: Glob pattern for files to search
        path: Base directory
        max_results: Maximum number of results
        ignore_case: Case-insensitive search

    Returns:
        List of {file, line_number, line_content} dicts
    """
    import re
    import glob as glob_module

    valid_path = validate_path(path)

    def _search() -> list[dict[str, Any]]:
        results = []
        flags = re.IGNORECASE if ignore_case else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error:
            return [{"error": f"Invalid regex pattern: {pattern}"}]

        try:
            file_glob = os.path.join(valid_path, "**", file_pattern)
            files = glob_module.glob(file_glob, recursive=True)

            for filepath in files:
                if not os.path.isfile(filepath):
                    continue

                try:
                    with open(filepath, encoding="utf-8", errors="replace") as f:
                        for line_num, line in enumerate(f, 1):
                            if regex.search(line):
                                rel_path = os.path.relpath(filepath, valid_path)
                                results.append(
                                    {
                                        "file": rel_path,
                                        "line_number": line_num,
                                        "line_content": line.rstrip()[:200],
                                    }
                                )
                                if len(results) >= max_results:
                                    return results
                except Exception:
                    continue

        except Exception as e:
            return [{"error": str(e)}]

        return results

    return await asyncio.to_thread(_search)


async def run_io(func: Callable, *args, **kwargs) -> Any:
    """
    Run a synchronous I/O function in a thread pool.

    This is a utility for running any blocking I/O operation
    without blocking the event loop.

    Args:
        func: Synchronous function to run
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Function result
    """
    return await asyncio.to_thread(func, *args, **kwargs)
