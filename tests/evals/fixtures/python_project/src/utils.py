"""Utility functions for the test project."""

import json
from pathlib import Path
from typing import Any, Dict, List


def read_json(file_path: str) -> Dict[str, Any]:
    """
    Read and parse a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Parsed JSON data as dictionary
    """
    with open(file_path, "r") as f:
        return json.load(f)


def write_json(file_path: str, data: Dict[str, Any], indent: int = 2) -> None:
    """
    Write data to a JSON file.

    Args:
        file_path: Path to the output file
        data: Data to write
        indent: JSON indentation level
    """
    with open(file_path, "w") as f:
        json.dump(data, f, indent=indent)


def ensure_directory(path: str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def list_files(
    directory: str,
    pattern: str = "*",
    recursive: bool = False,
) -> List[Path]:
    """
    List files in a directory matching a pattern.

    Args:
        directory: Directory to search
        pattern: Glob pattern to match
        recursive: Whether to search recursively

    Returns:
        List of matching file paths
    """
    dir_path = Path(directory)
    if recursive:
        return list(dir_path.rglob(pattern))
    return list(dir_path.glob(pattern))


def format_size(size_bytes: int) -> str:
    """
    Format a byte size as a human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def truncate_string(s: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate a string to a maximum length.

    Args:
        s: String to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated string
    """
    if len(s) <= max_length:
        return s
    return s[: max_length - len(suffix)] + suffix
