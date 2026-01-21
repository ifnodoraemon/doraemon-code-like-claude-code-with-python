import difflib

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()


def generate_diff(file_path: str, new_content: str) -> str:
    """
    Generate a colored diff between existing file content and new content.
    """
    import os

    # Handle new file case
    if not os.path.exists(file_path):
        return f"[new file] {file_path}\n" + new_content

    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            old_content = f.read()
    except Exception:
        return "[Binary or unreadable file - Cannot show diff]"

    diff_lines = difflib.unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm="",
    )

    diff_text = "".join(diff_lines)
    if not diff_text:
        return "[No changes]"

    return diff_text


def print_diff(file_path: str, new_content: str):
    """
    Print a syntax-highlighted diff to the console.
    """
    diff_text = generate_diff(file_path, new_content)

    if diff_text.startswith("["):
        console.print(f"[yellow]{diff_text}[/yellow]")
    else:
        syntax = Syntax(diff_text, "diff", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title=f"Diff: {file_path}", expand=False))
