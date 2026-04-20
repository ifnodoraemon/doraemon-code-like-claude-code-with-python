"""
Session Management System

Enhanced session management with support for:
- Session persistence and restoration
- Session forking (branching)
- Session export (JSON, Markdown)
- Session naming and search
- Multi-session management
"""

import json
import logging
import re
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
_SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


@dataclass
class SessionMetadata:
    """Metadata for a session."""

    id: str
    name: str | None = None
    project: str = "default"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    message_count: int = 0
    total_tokens: int = 0
    mode: str = "build"
    parent_id: str | None = None  # For forked sessions
    tags: list[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "project": self.project,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": self.message_count,
            "total_tokens": self.total_tokens,
            "mode": self.mode,
            "parent_id": self.parent_id,
            "tags": self.tags,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionMetadata":
        return cls(
            id=data["id"],
            name=data.get("name"),
            project=data.get("project", "default"),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            message_count=data.get("message_count", 0),
            total_tokens=data.get("total_tokens", 0),
            mode=data.get("mode", "build"),
            parent_id=data.get("parent_id"),
            tags=data.get("tags", []),
            description=data.get("description", ""),
        )

    def get_display_name(self) -> str:
        """Get display name for session."""
        if self.name:
            return self.name
        dt = datetime.fromtimestamp(self.created_at)
        return f"Session {self.id[:8]} ({dt.strftime('%Y-%m-%d %H:%M')})"


@dataclass
class SessionData:
    """Complete session data."""

    metadata: SessionMetadata
    messages: list[dict[str, Any]] = field(default_factory=list)
    summaries: list[dict[str, Any]] = field(default_factory=list)
    checkpoints: list[str] = field(default_factory=list)  # Checkpoint IDs
    orchestration_state: dict[str, Any] = field(default_factory=dict)
    orchestration_runs: list[dict[str, Any]] = field(default_factory=list)
    active_orchestration_run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "metadata": self.metadata.to_dict(),
            "messages": self.messages,
            "summaries": self.summaries,
            "checkpoints": self.checkpoints,
            "orchestration_state": self.orchestration_state,
            "orchestration_runs": self.orchestration_runs,
            "active_orchestration_run_id": self.active_orchestration_run_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionData":
        orchestration_state = data.get("orchestration_state", {})
        orchestration_runs = data.get("orchestration_runs")
        if orchestration_runs is None:
            orchestration_runs = [orchestration_state] if orchestration_state else []

        active_run_id = data.get("active_orchestration_run_id")
        if active_run_id is None and orchestration_state:
            active_run_id = orchestration_state.get("run_id")

        return cls(
            metadata=SessionMetadata.from_dict(data.get("metadata", data)),
            messages=data.get("messages", []),
            summaries=data.get("summaries", []),
            checkpoints=data.get("checkpoints", []),
            orchestration_state=orchestration_state,
            orchestration_runs=orchestration_runs,
            active_orchestration_run_id=active_run_id,
        )


class SessionManager:
    """
    Manages multiple sessions with persistence.

    Usage:
        mgr = SessionManager(base_dir=".agent/sessions")

        # Create new session
        session = mgr.create_session(project="myproject", name="auth-refactor")

        # List sessions
        sessions = mgr.list_sessions(project="myproject")

        # Resume session
        session = mgr.resume_session("abc123")
        session = mgr.resume_session("auth-refactor")  # By name

        # Fork session
        forked = mgr.fork_session("abc123", name="auth-v2")

        # Export session
        mgr.export_session("abc123", format="markdown", path="session.md")
    """

    def __init__(self, base_dir: str | Path = ".agent/sessions"):
        """
        Initialize session manager.

        Args:
            base_dir: Base directory for session storage
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, SessionMetadata] = {}
        self._load_index()

    def _generate_id(self) -> str:
        """Generate a unique session ID."""
        return uuid.uuid4().hex[:20]

    def _is_safe_session_id(self, session_id: str) -> bool:
        """Return whether a session identifier is safe to use as a filename."""
        return bool(_SESSION_ID_RE.fullmatch(session_id))

    def _get_session_path(self, session_id: str) -> Path:
        """Get path for session data file."""
        if not self._is_safe_session_id(session_id):
            raise ValueError(f"Invalid session ID: {session_id}")
        return self.base_dir / f"{session_id}.json"

    def _load_index(self):
        """Load session index."""
        index_path = self.base_dir / "index.json"
        if index_path.exists():
            try:
                data = json.loads(index_path.read_text(encoding="utf-8"))
                for entry in data.get("sessions", []):
                    meta = SessionMetadata.from_dict(entry)
                    self._index[meta.id] = meta
            except Exception as e:
                logger.error("Failed to load session index: %s", e)

    def _save_index(self):
        """Save session index."""
        index_path = self.base_dir / "index.json"
        data = {
            "version": 1,
            "sessions": [m.to_dict() for m in self._index.values()],
        }
        index_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def create_session(
        self,
        project: str = "default",
        name: str | None = None,
        mode: str = "build",
        description: str = "",
        tags: list[str] | None = None,
    ) -> SessionData:
        """
        Create a new session.

        Args:
            project: Project name
            name: Optional session name
            mode: Initial mode
            description: Session description
            tags: Optional tags

        Returns:
            New SessionData
        """
        session_id = self._generate_id()

        metadata = SessionMetadata(
            id=session_id,
            name=name,
            project=project,
            mode=mode,
            description=description,
            tags=tags or [],
        )

        session = SessionData(metadata=metadata)

        # Save session
        self._save_session(session)
        self._index[session_id] = metadata
        self._save_index()

        logger.info("Created session %s%s", session_id, f" ({name})" if name else "")
        return session

    def _save_session(self, session: SessionData):
        """Save session to file."""
        path = self._get_session_path(session.metadata.id)
        session.metadata.updated_at = time.time()
        path.write_text(
            json.dumps(session.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def load_session(self, session_id: str) -> SessionData | None:
        """Load a session by ID."""
        if not self._is_safe_session_id(session_id):
            logger.warning("Rejected unsafe session ID: %s", session_id)
            return None

        path = self._get_session_path(session_id)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return SessionData.from_dict(data)
        except Exception as e:
            logger.error("Failed to load session %s: %s", session_id, e)
            return None

    def save_session(self, session: SessionData):
        """Save session data."""
        self._save_session(session)

        # Update index
        self._index[session.metadata.id] = session.metadata
        self._save_index()

    def resume_session(self, identifier: str) -> SessionData | None:
        """
        Resume a session by ID or name.

        Args:
            identifier: Session ID or name

        Returns:
            SessionData or None if not found
        """
        # Try as ID first
        session = self.load_session(identifier)
        if session:
            logger.info("Resumed session %s", identifier)
            return session

        # Try as name
        for meta in self._index.values():
            if meta.name == identifier:
                session = self.load_session(meta.id)
                if session:
                    logger.info("Resumed session %s (%s)", meta.id, identifier)
                    return session

        logger.warning("Session not found: %s", identifier)
        return None

    def fork_session(
        self,
        session_id: str,
        name: str | None = None,
        at_message: int | None = None,
    ) -> SessionData | None:
        """
        Fork a session, creating a new branch.

        Args:
            session_id: Session to fork
            name: Name for forked session
            at_message: Fork at specific message index (None = current state)

        Returns:
            New forked session or None if source not found
        """
        source = self.load_session(session_id)
        if not source:
            return None

        # Create new session
        new_id = self._generate_id()

        # Determine messages to include
        messages = source.messages
        if at_message is not None:
            messages = messages[:at_message]
        message_count = len(messages)

        orchestration_runs = deepcopy(source.orchestration_runs)
        if at_message is not None:
            orchestration_runs = [
                run
                for run in orchestration_runs
                if int(run.get("message_end_index", -1)) < message_count
            ]
        latest_run = deepcopy(orchestration_runs[-1]) if orchestration_runs else {}

        metadata = SessionMetadata(
            id=new_id,
            name=name or f"Fork of {source.metadata.get_display_name()}",
            project=source.metadata.project,
            mode=source.metadata.mode,
            parent_id=session_id,
            message_count=message_count,
            tags=source.metadata.tags.copy(),
            description=f"Forked from {source.metadata.get_display_name()}",
        )

        forked = SessionData(
            metadata=metadata,
            messages=messages.copy(),
            summaries=source.summaries.copy(),
            checkpoints=source.checkpoints.copy(),
            orchestration_state=latest_run,
            orchestration_runs=orchestration_runs,
            active_orchestration_run_id=latest_run.get("run_id"),
        )

        # Save forked session
        self._save_session(forked)
        self._index[new_id] = metadata
        self._save_index()

        logger.info("Forked session %s -> %s", session_id, new_id)
        return forked

    def list_sessions(
        self,
        project: str | None = None,
        limit: int = 20,
        include_forks: bool = True,
    ) -> list[SessionMetadata]:
        """
        List sessions.

        Args:
            project: Filter by project
            limit: Maximum number to return
            include_forks: Include forked sessions

        Returns:
            List of session metadata
        """
        sessions = list(self._index.values())

        # Filter
        if project:
            sessions = [s for s in sessions if s.project == project]
        if not include_forks:
            sessions = [s for s in sessions if not s.parent_id]

        # Sort by updated_at descending
        sessions.sort(key=lambda s: s.updated_at, reverse=True)

        return sessions[:limit]

    def search_sessions(self, query: str, project: str | None = None) -> list[SessionMetadata]:
        """
        Search sessions by name, description, or tags.

        Args:
            query: Search query
            project: Optional project filter

        Returns:
            Matching sessions
        """
        query_lower = query.lower()
        results = []

        for meta in self._index.values():
            if project and meta.project != project:
                continue

            # Search in name
            if meta.name and query_lower in meta.name.lower():
                results.append(meta)
                continue

            # Search in description
            if meta.description and query_lower in meta.description.lower():
                results.append(meta)
                continue

            # Search in tags
            if any(query_lower in tag.lower() for tag in meta.tags):
                results.append(meta)
                continue

        # Sort by relevance (name match first, then by date)
        results.sort(
            key=lambda s: (
                0 if s.name and query_lower in s.name.lower() else 1,
                -s.updated_at,
            )
        )

        return results

    def rename_session(self, session_id: str, new_name: str) -> bool:
        """Rename a session."""
        if session_id not in self._index:
            return False

        self._index[session_id].name = new_name
        self._index[session_id].updated_at = time.time()

        # Update session file
        session = self.load_session(session_id)
        if session:
            session.metadata.name = new_name
            self._save_session(session)

        self._save_index()
        logger.info("Renamed session %s to '%s'", session_id, new_name)
        return True

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id not in self._index:
            return False

        # Delete file
        path = self._get_session_path(session_id)
        if path.exists():
            path.unlink()

        # Remove from index
        del self._index[session_id]
        self._save_index()

        logger.info("Deleted session %s", session_id)
        return True

    def export_session(
        self,
        session_id: str,
        format: str = "json",
        path: str | Path | None = None,
    ) -> str:
        """
        Export a session.

        Args:
            session_id: Session to export
            format: Export format ("json", "markdown", "text")
            path: Optional output file path

        Returns:
            Exported content as string
        """
        session = self.load_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        if format == "json":
            content = json.dumps(session.to_dict(), indent=2, ensure_ascii=False)

        elif format == "markdown":
            content = self._export_markdown(session)

        elif format == "text":
            content = self._export_text(session)

        else:
            raise ValueError(f"Unknown format: {format}")

        if path:
            Path(path).write_text(content, encoding="utf-8")
            logger.info("Exported session %s to %s", session_id, path)

        return content

    def _export_markdown(self, session: SessionData) -> str:
        """Export session as Markdown."""
        lines = [
            f"# {session.metadata.get_display_name()}",
            "",
            f"**Project:** {session.metadata.project}",
            f"**Created:** {datetime.fromtimestamp(session.metadata.created_at).isoformat()}",
            f"**Messages:** {session.metadata.message_count}",
            f"**Tokens:** {session.metadata.total_tokens:,}",
            "",
        ]

        if session.metadata.description:
            lines.extend([session.metadata.description, ""])

        if session.metadata.tags:
            lines.extend([f"**Tags:** {', '.join(session.metadata.tags)}", ""])

        lines.extend(["---", "", "## Conversation", ""])

        for msg in session.messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if role == "user":
                lines.extend(["### User", "", content, ""])
            else:
                lines.extend(["### Assistant", "", content, ""])

        return "\n".join(lines)

    def _export_text(self, session: SessionData) -> str:
        """Export session as plain text."""
        lines = [
            f"Session: {session.metadata.get_display_name()}",
            f"Project: {session.metadata.project}",
            f"Created: {datetime.fromtimestamp(session.metadata.created_at).isoformat()}",
            "-" * 50,
            "",
        ]

        for msg in session.messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            lines.extend([f"[{role}]", content, ""])

        return "\n".join(lines)

    def get_recent_sessions(self, limit: int = 5) -> list[dict[str, Any]]:
        """Get recent sessions for quick access."""
        sessions = self.list_sessions(limit=limit)
        return [
            {
                "id": s.id,
                "name": s.get_display_name(),
                "project": s.project,
                "updated_at": datetime.fromtimestamp(s.updated_at).isoformat(),
                "message_count": s.message_count,
            }
            for s in sessions
        ]
