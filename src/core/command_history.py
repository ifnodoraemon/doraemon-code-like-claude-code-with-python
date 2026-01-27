"""
Command History System

Provides command history with persistence and search.

Features:
- Persistent history across sessions
- Reverse search (Ctrl+R style)
- History per project
- Deduplication
"""

import json
import logging
import readline
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HistoryEntry:
    """A history entry."""

    command: str
    timestamp: float = field(default_factory=time.time)
    project: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "timestamp": self.timestamp,
            "project": self.project,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HistoryEntry":
        return cls(
            command=data["command"],
            timestamp=data.get("timestamp", time.time()),
            project=data.get("project", ""),
        )


class CommandHistory:
    """
    Manages command history with persistence.

    Usage:
        history = CommandHistory(project="myproject")

        # Add command
        history.add("explain this code")

        # Search history
        results = history.search("explain")

        # Get recent commands
        recent = history.get_recent(10)

        # Setup readline integration
        history.setup_readline()
    """

    def __init__(
        self,
        project: str = "default",
        history_file: str | Path | None = None,
        max_entries: int = 1000,
    ):
        """
        Initialize command history.

        Args:
            project: Project name
            history_file: Path to history file (None = auto)
            max_entries: Maximum entries to keep
        """
        self.project = project
        self.max_entries = max_entries

        if history_file:
            self._history_file = Path(history_file)
        else:
            self._history_file = Path.home() / ".doraemon" / "history" / f"{project}.json"

        self._history_file.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[HistoryEntry] = []
        self._search_results: list[HistoryEntry] = []
        self._search_index: int = 0

        self._load()

    def _load(self):
        """Load history from file."""
        if not self._history_file.exists():
            return

        try:
            data = json.loads(self._history_file.read_text(encoding="utf-8"))
            self._entries = [
                HistoryEntry.from_dict(e) for e in data.get("entries", [])
            ]
            logger.debug(f"Loaded {len(self._entries)} history entries")
        except Exception as e:
            logger.error(f"Failed to load history: {e}")

    def _save(self):
        """Save history to file."""
        try:
            # Trim to max entries
            if len(self._entries) > self.max_entries:
                self._entries = self._entries[-self.max_entries :]

            data = {
                "version": 1,
                "project": self.project,
                "entries": [e.to_dict() for e in self._entries],
            }
            self._history_file.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as e:
            logger.error(f"Failed to save history: {e}")

    def add(self, command: str):
        """
        Add a command to history.

        Args:
            command: Command to add
        """
        command = command.strip()
        if not command:
            return

        # Skip duplicates of the last entry
        if self._entries and self._entries[-1].command == command:
            return

        entry = HistoryEntry(command=command, project=self.project)
        self._entries.append(entry)
        self._save()

    def get_recent(self, limit: int = 20) -> list[str]:
        """
        Get recent commands.

        Args:
            limit: Maximum number to return

        Returns:
            List of recent commands
        """
        return [e.command for e in self._entries[-limit:]][::-1]

    def search(self, query: str, limit: int = 20) -> list[str]:
        """
        Search history for matching commands.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching commands
        """
        query_lower = query.lower()
        results = []

        for entry in reversed(self._entries):
            if query_lower in entry.command.lower():
                if entry.command not in results:
                    results.append(entry.command)
                if len(results) >= limit:
                    break

        return results

    def search_reverse(self, query: str) -> str | None:
        """
        Start or continue reverse search (Ctrl+R style).

        Args:
            query: Search query

        Returns:
            Matching command or None
        """
        if not query:
            self._search_results = []
            self._search_index = 0
            return None

        # Start new search
        self._search_results = [
            e for e in reversed(self._entries) if query.lower() in e.command.lower()
        ]
        self._search_index = 0

        if self._search_results:
            return self._search_results[0].command
        return None

    def search_next(self) -> str | None:
        """Get next result in reverse search."""
        if not self._search_results:
            return None

        self._search_index += 1
        if self._search_index >= len(self._search_results):
            self._search_index = 0

        return self._search_results[self._search_index].command

    def search_prev(self) -> str | None:
        """Get previous result in reverse search."""
        if not self._search_results:
            return None

        self._search_index -= 1
        if self._search_index < 0:
            self._search_index = len(self._search_results) - 1

        return self._search_results[self._search_index].command

    def clear(self):
        """Clear all history."""
        self._entries = []
        self._save()

    def setup_readline(self):
        """Setup readline integration for arrow key history."""
        try:
            # Load history into readline
            for entry in self._entries[-100:]:  # Load last 100
                readline.add_history(entry.command)

            # Configure readline
            readline.set_history_length(self.max_entries)

            logger.debug("Readline history setup complete")

        except Exception as e:
            logger.warning(f"Failed to setup readline: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get history statistics."""
        return {
            "total_entries": len(self._entries),
            "project": self.project,
            "file": str(self._history_file),
        }


class BashModeExecutor:
    """
    Executes shell commands directly (Bash mode with ! prefix).

    Usage:
        executor = BashModeExecutor()

        # Execute command
        output = executor.execute("ls -la")

        # Execute and capture for context
        result = executor.execute_for_context("git status")
    """

    def __init__(self, cwd: str | Path | None = None):
        """
        Initialize executor.

        Args:
            cwd: Working directory
        """
        self.cwd = Path(cwd) if cwd else Path.cwd()

    def execute(
        self, command: str, timeout: float = 30, capture: bool = True
    ) -> dict[str, Any]:
        """
        Execute a shell command.

        Args:
            command: Command to execute
            timeout: Timeout in seconds
            capture: Capture output

        Returns:
            Dict with:
            - success: bool
            - output: str
            - error: str
            - exit_code: int
        """
        import subprocess

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.cwd,
                capture_output=capture,
                text=True,
                timeout=timeout,
            )

            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
                "exit_code": result.returncode,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": f"Command timed out after {timeout}s",
                "exit_code": -1,
            }

        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "exit_code": -1,
            }

    def execute_for_context(self, command: str, timeout: float = 30) -> str:
        """
        Execute command and format output for conversation context.

        Args:
            command: Command to execute
            timeout: Timeout in seconds

        Returns:
            Formatted output string for context
        """
        result = self.execute(command, timeout)

        lines = [f"$ {command}"]

        if result["output"]:
            lines.append(result["output"].rstrip())

        if result["error"]:
            lines.append(f"[stderr] {result['error'].rstrip()}")

        if result["exit_code"] != 0:
            lines.append(f"[exit code: {result['exit_code']}]")

        return "\n".join(lines)
