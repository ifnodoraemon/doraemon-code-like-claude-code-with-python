"""
Doraemon Directory Management - Two-Level Storage

User Level (~/.doraemon/):
    Global settings and shared resources across all projects.
    ├── settings.json     # User preferences (model, theme, language)
    ├── skills/           # User-defined skills (shared across projects)
    ├── cache/            # Global cache
    └── usage-data/       # Usage statistics

Project Level (.agent/):
    Project-specific data, stored in the project directory.
    ├── config.json       # Project model and runtime configuration
    ├── conversations/    # Session history
    ├── checkpoints/      # File snapshots
    ├── traces/           # Execution traces
    └── memory/           # Project memory

Design Principles:
    1. Lazy creation - directories created only when needed
    2. Clear separation - user preferences vs project data
    3. Portability - .agent/ can be committed to version control
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

USER_HOME = Path.home() / ".doraemon"

_project_dir: Path | None = None


def set_project_dir(path: Path | None) -> None:
    """Set the current project directory."""
    global _project_dir
    _project_dir = path


def get_project_dir() -> Path:
    """Get current project directory (defaults to cwd)."""
    if _project_dir:
        return _project_dir
    return Path.cwd()


def get_agent_dir() -> Path:
    """Get .agent directory path (project level)."""
    return get_project_dir() / ".agent"


# ========================================
# User Level (~/.doraemon/)
# ========================================


def _ensure_user_home() -> Path:
    """Ensure user home directory exists (lazy creation)."""
    USER_HOME.mkdir(parents=True, exist_ok=True)
    for subdir in ["skills", "cache", "usage-data"]:
        (USER_HOME / subdir).mkdir(exist_ok=True)
    return USER_HOME


def get_user_settings_path() -> Path:
    """Get user-level settings.json path."""
    return USER_HOME / "settings.json"


def get_user_skills_dir() -> Path:
    """Get user-level skills directory."""
    _ensure_user_home()
    return USER_HOME / "skills"


def get_user_cache_dir() -> Path:
    """Get user-level cache directory."""
    _ensure_user_home()
    return USER_HOME / "cache"


def get_usage_data_dir() -> Path:
    """Get usage data directory."""
    _ensure_user_home()
    return USER_HOME / "usage-data"


DEFAULT_USER_SETTINGS = {
    "mode": "build",
    "max_turns": 100,
    "max_context_tokens": 128000,
    "theme": "dark",
    "telemetry": True,
    "language": "auto",
}


def load_user_settings() -> dict[str, Any]:
    """Load user-level settings."""
    settings_path = get_user_settings_path()
    if settings_path.exists():
        try:
            return json.loads(settings_path.read_text())
        except Exception:
            pass
    return DEFAULT_USER_SETTINGS.copy()


def save_user_settings(settings: dict[str, Any]) -> None:
    """Save user-level settings."""
    _ensure_user_home()
    settings_path = get_user_settings_path()
    settings_path.write_text(json.dumps(settings, indent=2))


# ========================================
# Project Level (.agent/)
# ========================================


def _ensure_agent_dir() -> Path:
    """Ensure .agent directory exists (lazy creation)."""
    agent_dir = get_agent_dir()
    agent_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["conversations", "checkpoints", "traces", "memory", "logs"]:
        (agent_dir / subdir).mkdir(exist_ok=True)
    return agent_dir


def get_project_config_path() -> Path:
    """Get project config.json path."""
    return get_agent_dir() / "config.json"


def get_conversations_dir() -> Path:
    """Get conversations directory."""
    _ensure_agent_dir()
    return get_agent_dir() / "conversations"


def get_checkpoints_dir() -> Path:
    """Get checkpoints directory."""
    _ensure_agent_dir()
    return get_agent_dir() / "checkpoints"


def get_traces_dir() -> Path:
    """Get traces directory."""
    _ensure_agent_dir()
    return get_agent_dir() / "traces"


def get_memory_dir() -> Path:
    """Get memory directory."""
    _ensure_agent_dir()
    return get_agent_dir() / "memory"


def get_project_skills_dir() -> Path:
    """Get project-level skills directory."""
    _ensure_agent_dir()
    return get_agent_dir() / "skills"


# ========================================
# History Management
# ========================================


def get_history_path() -> Path:
    """Get command history path (user level)."""
    return USER_HOME / "history.jsonl"


def append_history(entry: dict[str, Any]) -> None:
    """Append entry to command history."""
    _ensure_user_home()
    history_path = get_history_path()
    entry["timestamp"] = time.time()
    entry["datetime"] = datetime.now().isoformat()
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_history(limit: int = 100) -> list[dict]:
    """Read recent command history."""
    history_path = get_history_path()
    if not history_path.exists():
        return []
    entries = []
    with open(history_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries[-limit:]


# ========================================
# Trace System
# ========================================


@dataclass
class TraceEvent:
    """A single trace event."""

    type: str
    name: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "name": self.name,
            "data": self.data,
            "timestamp": self.timestamp,
            "duration": self.duration,
        }


class Trace:
    """
    Execution trace recorder with hierarchical IDs.

    Traces are saved to .agent/traces/ (project level).

    ID Hierarchy:
        session_id  - Entire conversation session
          └── turn_id - Single agent run() call
                └── span_id - Single operation (tool_call, llm_call)

    Usage:
        trace = Trace("session_abc123")
        trace.start_turn("What files exist?")
        trace.tool_call("read", {"path": "/src"}, "result...", 0.05)
        trace.llm_call("gemini", messages, response, 100, 50, 1.2)
        trace.end_turn()
        trace.save()
    """

    def __init__(self, session_id: str, project_dir: Path | None = None):
        if not session_id:
            import uuid

            session_id = str(uuid.uuid4())
        self.session_id = session_id
        self.events: list[TraceEvent] = []
        self._start_time = time.time()
        self._project_dir = project_dir
        self._turn_count = 0
        self._current_turn_id: str | None = None
        self._span_count = 0

    def start_turn(self, user_input: str, metadata: dict[str, Any] | None = None) -> str:
        """Start a new turn, returns turn_id."""
        self._turn_count += 1
        self._span_count = 0
        self._current_turn_id = f"{self.session_id[:8]}-t{self._turn_count}"
        data = {
            "user_input": user_input[:500],
            "input": {
                "role": "user",
                "content": user_input,
            },
        }
        if metadata:
            data.update(metadata)
        self.events.append(
            TraceEvent(
                type="turn_start",
                name=self._current_turn_id,
                data=data,
            )
        )
        return self._current_turn_id

    def end_turn(self, success: bool = True, error: str | None = None) -> None:
        """End current turn."""
        if self._current_turn_id:
            self.events.append(
                TraceEvent(
                    type="turn_end",
                    name=self._current_turn_id,
                    data={"success": success, "error": error},
                )
            )
        self._current_turn_id = None

    def _next_span_id(self) -> str:
        """Generate next span ID."""
        self._span_count += 1
        return f"{self._current_turn_id or 'unknown'}-s{self._span_count}"

    def event(
        self,
        type: str,
        name: str,
        data: dict[str, Any] | None = None,
    ) -> str:
        """Record an event, returns event_id."""
        span_id = self._next_span_id()
        self.events.append(
            TraceEvent(
                type=type,
                name=name,
                data={"span_id": span_id, **(data or {})},
            )
        )
        return span_id

    def tool_call(
        self,
        tool_name: str,
        args: dict,
        result: str,
        duration: float,
        error: str | None = None,
        source: str = "built_in",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Record a tool call, returns span_id."""
        span_id = self._next_span_id()
        extra_metadata = metadata or {}
        self.events.append(
            TraceEvent(
                type="tool_call",
                name=tool_name,
                data={
                    "span_id": span_id,
                    "session_id": self.session_id,
                    "turn_id": self._current_turn_id,
                    "tool_name": tool_name,
                    "tool_source": source,
                    "input": {
                        "arguments": args,
                    },
                    "output": {
                        "content": result,
                        "error": error,
                    },
                    "success": error is None,
                    "args": args,
                    "result": result[:2000] if result else None,
                    "error": error,
                    **extra_metadata,
                },
                duration=duration,
            )
        )
        return span_id

    def llm_call(
        self,
        model: str,
        messages: list[dict] | None = None,
        response: dict | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        duration: float = 0.0,
    ) -> str:
        """Record an LLM call with full input/output, returns span_id."""
        span_id = self._next_span_id()
        self.events.append(
            TraceEvent(
                type="llm_call",
                name=model,
                data={
                    "span_id": span_id,
                    "session_id": self.session_id,
                    "turn_id": self._current_turn_id,
                    "model": model,
                    "input": {
                        "messages": messages or [],
                    },
                    "output": response or {},
                    "input_messages": [
                        {"role": m.get("role"), "content": str(m.get("content", ""))[:1000]}
                        for m in (messages or [])
                    ],
                    "response": {
                        "content": str(response.get("content", ""))[:2000] if response else None,
                        "tool_calls": response.get("tool_calls") if response else None,
                    },
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
                duration=duration,
            )
        )
        return span_id

    def error(self, message: str, details: dict | None = None, span_id: str | None = None) -> str:
        """Record an error with optional span context."""
        event_id = self._next_span_id()
        self.events.append(
            TraceEvent(
                type="error",
                name="error",
                data={
                    "span_id": span_id or event_id,
                    "session_id": self.session_id,
                    "turn_id": self._current_turn_id,
                    "message": message,
                    "details": details or {},
                },
            )
        )
        return event_id

    def save(self) -> Path:
        """Save trace to project traces directory."""
        if self._project_dir:
            traces_dir = self._project_dir / ".agent" / "traces"
            traces_dir.mkdir(parents=True, exist_ok=True)
        else:
            traces_dir = get_traces_dir()

        self._cleanup_old_traces(traces_dir, max_files=200, max_age_days=30)

        trace_file = traces_dir / f"{self.session_id}.json"

        data = {
            "schema_version": 2,
            "session_id": self.session_id,
            "start_time": self._start_time,
            "end_time": time.time(),
            "events": [e.to_dict() for e in self.events],
            "summary": {
                "total_events": len(self.events),
                "tool_calls": len([e for e in self.events if e.type == "tool_call"]),
                "llm_calls": len([e for e in self.events if e.type == "llm_call"]),
                "errors": len([e for e in self.events if e.type == "error"]),
            },
        }

        trace_file.write_text(json.dumps(data, indent=2))
        return trace_file

    @staticmethod
    def _cleanup_old_traces(traces_dir: Path, max_files: int = 200, max_age_days: int = 30) -> None:
        """Remove oldest trace files when count exceeds max_files or age exceeds max_age_days."""
        try:
            trace_files = sorted(
                traces_dir.glob("*.json"),
                key=lambda p: p.stat().st_mtime,
            )
            now = time.time()
            cutoff = now - (max_age_days * 86400)
            removed = 0
            for f in trace_files:
                should_remove = False
                try:
                    if f.stat().st_mtime < cutoff:
                        should_remove = True
                except OSError:
                    should_remove = True
                if should_remove:
                    f.unlink(missing_ok=True)
                    removed += 1
            if len(trace_files) - removed > max_files:
                for f in trace_files[removed : len(trace_files) - max_files + removed]:
                    f.unlink(missing_ok=True)
                    removed += 1
            if removed:
                logger.info("Cleaned up %d old trace files from %s", removed, traces_dir)
        except OSError:
            pass

    @classmethod
    def load(cls, session_id: str, project_dir: Path | None = None) -> "Trace | None":
        """Load a trace from file."""
        if project_dir:
            traces_dir = project_dir / ".agent" / "traces"
        else:
            traces_dir = get_traces_dir()

        trace_file = traces_dir / f"{session_id}.json"

        if not trace_file.exists():
            return None

        try:
            data = json.loads(trace_file.read_text())
            trace = cls(session_id, project_dir)
            trace._start_time = data.get("start_time", time.time())
            trace.events = [TraceEvent(**e) for e in data.get("events", [])]
            return trace
        except Exception:
            return None


# ========================================
# Backup System
# ========================================


def backup_file(path: Path, reason: str = "edit") -> Path | None:
    """Backup a file before modification (project level)."""
    if not path.exists():
        return None

    backups_dir = get_checkpoints_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{path.name}.{timestamp}.{reason}.bak"
    backup_path = backups_dir / backup_name

    import shutil

    shutil.copy2(path, backup_path)

    return backup_path


def list_backups(path: Path | None = None) -> list[Path]:
    """List available backups."""
    backups_dir = get_checkpoints_dir()

    if path:
        pattern = f"{path.name}.*"
        return sorted(backups_dir.glob(pattern))

    return sorted(backups_dir.glob("*"))


def restore_backup(backup_path: Path, target: Path) -> bool:
    """Restore a file from backup."""
    if not backup_path.exists():
        return False

    import shutil

    shutil.copy2(backup_path, target)
    return True


# ========================================
# Session Management
# ========================================


def save_session(session_id: str, data: dict[str, Any]) -> Path:
    """Save session data to project conversations."""
    sessions_dir = get_conversations_dir()
    session_file = sessions_dir / f"{session_id}.json"

    data["saved_at"] = time.time()
    session_file.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    return session_file


def load_session(session_id: str) -> dict[str, Any] | None:
    """Load session data from project conversations."""
    sessions_dir = get_conversations_dir()
    session_file = sessions_dir / f"{session_id}.json"

    if not session_file.exists():
        return None

    try:
        return json.loads(session_file.read_text())
    except Exception:
        return None


def list_sessions() -> list[dict[str, Any]]:
    """List all saved sessions for current project."""
    sessions_dir = get_conversations_dir()

    sessions = []
    for session_file in sessions_dir.glob("*.json"):
        try:
            data = json.loads(session_file.read_text())
            data["id"] = session_file.stem
            sessions.append(data)
        except (json.JSONDecodeError, OSError):
            pass

    return sorted(sessions, key=lambda s: s.get("saved_at", 0), reverse=True)
