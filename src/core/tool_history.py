"""
Tool Execution History

Records and allows replay of tool executions.

Features:
- Full execution recording
- Replay capability
- Filtering and search
- Export/Import
- Debug support
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Execution status."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ToolExecution:
    """A single tool execution record."""

    id: str
    tool: str
    arguments: dict[str, Any]
    result: Any
    status: ExecutionStatus
    started_at: float
    completed_at: float
    error_message: str | None = None
    session_id: str | None = None
    parent_id: str | None = None  # For nested calls
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Get execution duration in seconds."""
        return self.completed_at - self.started_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tool": self.tool,
            "arguments": self.arguments,
            "result": self._serialize_result(self.result),
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration": self.duration,
            "error_message": self.error_message,
            "session_id": self.session_id,
            "parent_id": self.parent_id,
            "metadata": self.metadata,
        }

    def _serialize_result(self, result: Any) -> Any:
        """Serialize result for storage."""
        if isinstance(result, str | int | float | bool | type(None)):
            return result
        if isinstance(result, list | dict):
            try:
                json.dumps(result)
                return result
            except (TypeError, ValueError):
                return str(result)
        return str(result)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolExecution":
        return cls(
            id=data["id"],
            tool=data["tool"],
            arguments=data["arguments"],
            result=data["result"],
            status=ExecutionStatus(data["status"]),
            started_at=data["started_at"],
            completed_at=data["completed_at"],
            error_message=data.get("error_message"),
            session_id=data.get("session_id"),
            parent_id=data.get("parent_id"),
            metadata=data.get("metadata", {}),
        )


class ToolHistoryManager:
    """
    Manages tool execution history.

    Usage:
        history = ToolHistoryManager()

        # Record execution
        exec_id = history.start("file_read", {"path": "/file"})
        result = do_tool_call()
        history.complete(exec_id, result)

        # Or use context manager
        with history.record("file_read", {"path": "/file"}) as recorder:
            result = do_tool_call()
            recorder.set_result(result)

        # Query history
        recent = history.get_recent(10)
        by_tool = history.filter(tool="file_read")

        # Replay
        await history.replay(exec_id, executor)
    """

    def __init__(
        self,
        max_entries: int = 10000,
        persist_path: Path | None = None,
        session_id: str | None = None,
    ):
        """
        Initialize history manager.

        Args:
            max_entries: Maximum history entries to keep
            persist_path: Path to persist history
            session_id: Current session ID
        """
        self._entries: list[ToolExecution] = []
        self._max_entries = max_entries
        self._persist_path = persist_path
        self._session_id = session_id or str(uuid.uuid4())[:8]
        self._pending: dict[str, dict] = {}  # In-progress executions

        # Load persisted history
        if persist_path and persist_path.exists():
            self._load_history()

    def start(
        self,
        tool: str,
        arguments: dict[str, Any],
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Start recording a tool execution.

        Args:
            tool: Tool name
            arguments: Tool arguments
            parent_id: Parent execution ID (for nested calls)
            metadata: Additional metadata

        Returns:
            Execution ID
        """
        exec_id = str(uuid.uuid4())

        self._pending[exec_id] = {
            "id": exec_id,
            "tool": tool,
            "arguments": arguments,
            "started_at": time.time(),
            "session_id": self._session_id,
            "parent_id": parent_id,
            "metadata": metadata or {},
        }

        return exec_id

    def complete(
        self,
        exec_id: str,
        result: Any,
        status: ExecutionStatus = ExecutionStatus.SUCCESS,
        error_message: str | None = None,
    ):
        """
        Complete a tool execution recording.

        Args:
            exec_id: Execution ID from start()
            result: Execution result
            status: Execution status
            error_message: Error message if failed
        """
        if exec_id not in self._pending:
            logger.warning("Unknown execution ID: %s", exec_id)
            return

        pending = self._pending.pop(exec_id)

        execution = ToolExecution(
            id=pending["id"],
            tool=pending["tool"],
            arguments=pending["arguments"],
            result=result,
            status=status,
            started_at=pending["started_at"],
            completed_at=time.time(),
            error_message=error_message,
            session_id=pending["session_id"],
            parent_id=pending["parent_id"],
            metadata=pending["metadata"],
        )

        self._add_entry(execution)

    def record(
        self,
        tool: str,
        arguments: dict[str, Any],
        parent_id: str | None = None,
    ) -> "ExecutionRecorder":
        """
        Context manager for recording execution.

        Args:
            tool: Tool name
            arguments: Tool arguments
            parent_id: Parent execution ID

        Returns:
            ExecutionRecorder context manager
        """
        return ExecutionRecorder(self, tool, arguments, parent_id)

    def _add_entry(self, execution: ToolExecution):
        """Add an entry to history."""
        self._entries.append(execution)

        # Trim if too large
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries // 2 :]

        # Persist
        if self._persist_path:
            self._save_history()

    def get_recent(self, limit: int = 50) -> list[ToolExecution]:
        """Get recent executions."""
        return self._entries[-limit:]

    def get_by_id(self, exec_id: str) -> ToolExecution | None:
        """Get execution by ID."""
        for entry in reversed(self._entries):
            if entry.id == exec_id:
                return entry
        return None

    def filter(
        self,
        tool: str | None = None,
        status: ExecutionStatus | None = None,
        session_id: str | None = None,
        since: float | None = None,
        until: float | None = None,
    ) -> list[ToolExecution]:
        """
        Filter executions.

        Args:
            tool: Filter by tool name
            status: Filter by status
            session_id: Filter by session
            since: Filter by start time (timestamp)
            until: Filter by end time (timestamp)

        Returns:
            Filtered list of executions
        """
        results = []

        for entry in self._entries:
            if tool and entry.tool != tool:
                continue
            if status and entry.status != status:
                continue
            if session_id and entry.session_id != session_id:
                continue
            if since and entry.started_at < since:
                continue
            if until and entry.completed_at > until:
                continue
            results.append(entry)

        return results

    def search(self, query: str) -> list[ToolExecution]:
        """
        Search executions.

        Args:
            query: Search query (searches tool names and arguments)

        Returns:
            Matching executions
        """
        query = query.lower()
        results = []

        for entry in self._entries:
            if query in entry.tool.lower():
                results.append(entry)
                continue
            if query in str(entry.arguments).lower():
                results.append(entry)
                continue

        return results

    async def replay(
        self,
        exec_id: str,
        executor: Any,  # Function(tool, args) -> result
    ) -> Any:
        """
        Replay an execution.

        Args:
            exec_id: Execution ID to replay
            executor: Tool executor function

        Returns:
            New execution result
        """
        execution = self.get_by_id(exec_id)
        if not execution:
            raise ValueError(f"Execution not found: {exec_id}")

        return await executor(execution.tool, execution.arguments)

    def get_stats(self) -> dict[str, Any]:
        """Get execution statistics."""
        if not self._entries:
            return {
                "total": 0,
                "by_tool": {},
                "by_status": {},
                "avg_duration": 0,
            }

        by_tool: dict[str, int] = {}
        by_status: dict[str, int] = {}
        total_duration = 0

        for entry in self._entries:
            by_tool[entry.tool] = by_tool.get(entry.tool, 0) + 1
            status_key = entry.status.value
            by_status[status_key] = by_status.get(status_key, 0) + 1
            total_duration += entry.duration

        return {
            "total": len(self._entries),
            "by_tool": by_tool,
            "by_status": by_status,
            "avg_duration": total_duration / len(self._entries),
            "session_id": self._session_id,
        }

    def clear(self):
        """Clear all history."""
        self._entries.clear()
        self._pending.clear()

        if self._persist_path and self._persist_path.exists():
            self._persist_path.unlink()

    def export(self, path: Path | None = None) -> str:
        """
        Export history to JSON.

        Args:
            path: Optional path to write to

        Returns:
            JSON string
        """
        data = {
            "exported_at": time.time(),
            "session_id": self._session_id,
            "total_entries": len(self._entries),
            "entries": [e.to_dict() for e in self._entries],
        }

        json_str = json.dumps(data, indent=2)

        if path:
            path.write_text(json_str, encoding="utf-8")

        return json_str

    def import_history(self, path: Path):
        """
        Import history from JSON file.

        Args:
            path: Path to JSON file
        """
        data = json.loads(path.read_text(encoding="utf-8"))

        for entry_data in data.get("entries", []):
            try:
                execution = ToolExecution.from_dict(entry_data)
                self._entries.append(execution)
            except Exception as e:
                logger.warning("Failed to import entry: %s", e)

        logger.info("Imported %s entries", len(data.get('entries', [])))

    def _save_history(self):
        """Save history to disk."""
        if not self._persist_path:
            return

        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "entries": [e.to_dict() for e in self._entries[-1000:]],
            }
            self._persist_path.write_text(json.dumps(data), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to save history: %s", e)

    def _load_history(self):
        """Load history from disk."""
        try:
            data = json.loads(self._persist_path.read_text(encoding="utf-8"))
            for entry_data in data.get("entries", []):
                try:
                    self._entries.append(ToolExecution.from_dict(entry_data))
                except Exception:
                    pass
            logger.info("Loaded %s history entries", len(self._entries))
        except Exception as e:
            logger.warning("Failed to load history: %s", e)


class ExecutionRecorder:
    """Context manager for recording tool execution."""

    def __init__(
        self,
        manager: ToolHistoryManager,
        tool: str,
        arguments: dict[str, Any],
        parent_id: str | None = None,
    ):
        self._manager = manager
        self._tool = tool
        self._arguments = arguments
        self._parent_id = parent_id
        self._exec_id: str | None = None
        self._result: Any = None
        self._status = ExecutionStatus.SUCCESS
        self._error: str | None = None

    def __enter__(self) -> "ExecutionRecorder":
        self._exec_id = self._manager.start(self._tool, self._arguments, self._parent_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self._status = ExecutionStatus.ERROR
            self._error = str(exc_val)

        self._manager.complete(
            self._exec_id,
            self._result,
            self._status,
            self._error,
        )

    def set_result(self, result: Any):
        """Set the execution result."""
        self._result = result

    def set_error(self, error: str):
        """Set error status."""
        self._status = ExecutionStatus.ERROR
        self._error = error

    @property
    def exec_id(self) -> str | None:
        """Get execution ID."""
        return self._exec_id


# Global history instance
_tool_history: ToolHistoryManager | None = None


def get_tool_history() -> ToolHistoryManager:
    """Get the global tool history manager."""
    global _tool_history
    if _tool_history is None:
        _tool_history = ToolHistoryManager()
    return _tool_history
