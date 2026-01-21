"""
Polymath Agent Definition for ADK Web Debugging.

This module provides a standalone agent definition that can be used
with Google ADK Web UI for debugging and development.

Note: This is for development/debugging only. The main CLI uses
the MCP client architecture defined in src/host/cli.py.
"""

import json
import os
from typing import Any

from rich.console import Console

from src.servers.computer import execute_python
from src.servers.fs_read import list_directory, read_file
from src.servers.fs_write import write_file

# Import tool implementations from servers (for in-process debugging)
from src.servers.memory import get_user_persona, save_note, search_notes, update_user_persona
from src.servers.web import fetch_page, search_internet

console = Console()


def load_config() -> dict[str, Any]:
    """Load configuration from .polymath/config.json."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, ".polymath", "config.json")

    if not os.path.exists(config_path):
        return {"persona": {}}
    with open(config_path) as f:
        return json.load(f)


config = load_config()
persona = config.get("persona", {})

# Agent configuration for reference (used by external tools like ADK)
AGENT_CONFIG = {
    "name": persona.get("name", "Polymath"),
    "model": os.getenv("MODEL_NAME", "gemini-2.0-flash"),
    "description": f"A {persona.get('role', 'Generalist AI Assistant')} with access to memory, files, web, and code execution tools.",
    "tools": [
        # Memory Tools
        {"name": "save_note", "function": save_note},
        {"name": "search_notes", "function": search_notes},
        {"name": "update_user_persona", "function": update_user_persona},
        {"name": "get_user_persona", "function": get_user_persona},
        # Web Tools
        {"name": "search_internet", "function": search_internet},
        {"name": "fetch_page", "function": fetch_page},
        # Computer Tools
        {"name": "execute_python", "function": execute_python},
        # Filesystem Tools
        {"name": "read_file", "function": read_file},
        {"name": "list_directory", "function": list_directory},
        {"name": "write_file", "function": write_file},
    ],
    "instructions": f"""
    You are {persona.get("name", "Polymath")}, a {persona.get("role", "Generalist AI Assistant")}.

    You have access to a suite of powerful tools:
    1. [Memory]: You can save notes and recall user preferences.
    2. [Files]: You can read files (PDFs, Images, Code) and list directories.
       Note: 'read_file' handles OCR automatically for images.
    3. [Web]: You can search and fetch internet content.
    4. [Computer]: Execute Python code.

    Always use the appropriate tool for the task.
    If asked to write something, check your memory for style guides first.
    """,
}


def get_tool_functions() -> dict[str, callable]:
    """Return a dictionary of tool name to function mappings."""
    return {tool["name"]: tool["function"] for tool in AGENT_CONFIG["tools"]}


if __name__ == "__main__":
    console.print("[bold green]Polymath Agent Configuration Loaded.[/bold green]")
    console.print(f"[dim]Agent Name: {AGENT_CONFIG['name']}[/dim]")
    console.print(f"[dim]Available Tools: {len(AGENT_CONFIG['tools'])}[/dim]")
    for tool in AGENT_CONFIG["tools"]:
        console.print(f"  - {tool['name']}")
