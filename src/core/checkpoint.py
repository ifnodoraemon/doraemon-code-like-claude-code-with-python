"""
Checkpoint System - File Edit History and Rollback

Provides automatic checkpointing before file edits with support for
reverting changes to specific points in time.

Features:
- Automatic snapshots before each file edit
- Checkpoint per user prompt
- 30-day retention (configurable)
- Rewind to any checkpoint (code only, conversation only, or both)
"""

import gzip
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)


@dataclass
class FileSnapshot:
    """Snapshot of a single file."""

    path: str
    content: str | None
    exists: bool
    size: int
    mtime: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "content": self.content,
            "exists": self.exists,
            "size": self.size,
            "mtime": self.mtime,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileSnapshot":
        return cls(
            path=data["path"],
            content=data.get("content"),
            exists=data.get("exists", True),
            size=data.get("size", 0),
            mtime=data.get("mtime"),
        )


@dataclass
class Checkpoint:
    """A checkpoint containing file snapshots and conversation state."""

    id: str
    created_at: float
    prompt: str
    files: list[FileSnapshot] = field(default_factory=list)
    message_count: int = 0
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "prompt": self.prompt,
            "files": [f.to_dict() for f in self.files],
            "message_count": self.message_count,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        return cls(
            id=data["id"],
            created_at=data["created_at"],
            prompt=data.get("prompt", ""),
            files=[FileSnapshot.from_dict(f) for f in data.get("files", [])],
            message_count=data.get("message_count", 0),
            description=data.get("description", ""),
        )


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint system."""

    enabled: bool = True
    save_directory: str = ".agent/checkpoints"
    max_file_size: int = 1024 * 1024
    retention_days: int = 30
    compress: bool = True


class CheckpointManager:
    """
    Manages file checkpoints for rollback capability.

    Usage:
        mgr = CheckpointManager(project="myproject")
        mgr.begin_checkpoint("user prompt", message_count=5)
        mgr.snapshot_file("/path/to/file.py")
        mgr.finalize_checkpoint()
        mgr.rewind(checkpoint_id, mode="both")
    """

    def __init__(
        self,
        project: str = "default",
        config: CheckpointConfig | None = None,
    ):
        self.project = project
        self.config = config or CheckpointConfig()
        self.checkpoints: list[Checkpoint] = []
        self._current: Checkpoint | None = None
        self._pending_files: set[str] = set()

        self._base_dir = Path(self.config.save_directory) / project
        self._base_dir.mkdir(parents=True, exist_ok=True)

        if not self._load_index():
            return
        self._cleanup_old_checkpoints()

    def begin_checkpoint(self, prompt: str, message_count: int = 0) -> str:
        """Begin a new checkpoint before processing a user prompt."""
        if not self.config.enabled:
            return ""

        checkpoint_id = self._generate_id()
        self._current = Checkpoint(
            id=checkpoint_id,
            created_at=time.time(),
            prompt=prompt[:500],
            message_count=message_count,
        )
        self._pending_files.clear()

        logger.debug("Started checkpoint %s", checkpoint_id)
        return checkpoint_id

    def snapshot_file(self, path: str) -> bool:
        """Take a snapshot of a file before it's modified."""
        if not self.config.enabled or not self._current:
            return False

        abs_path = str(Path(path).resolve())
        if abs_path in self._pending_files:
            return False

        try:
            file_path = Path(path)

            if file_path.exists():
                stat_result = file_path.stat()
                size = stat_result.st_size
                mtime = stat_result.st_mtime
                if size > self.config.max_file_size:
                    logger.warning("File too large for snapshot: %s (%s bytes)", path, size)
                    snapshot = FileSnapshot(
                        path=abs_path,
                        content=None,
                        exists=True,
                        size=size,
                        mtime=mtime,
                    )
                else:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                    snapshot = FileSnapshot(
                        path=abs_path,
                        content=content,
                        exists=True,
                        size=size,
                        mtime=mtime,
                    )
            else:
                snapshot = FileSnapshot(
                    path=abs_path,
                    content=None,
                    exists=False,
                    size=0,
                    mtime=None,
                )

            self._current.files.append(snapshot)
            self._pending_files.add(abs_path)
            logger.debug("Snapshot taken: %s", path)
            return True

        except Exception as e:
            logger.error("Failed to snapshot file %s: %s", path, e)
            return False

    def finalize_checkpoint(self, description: str = "") -> str | None:
        """Finalize the current checkpoint."""
        if not self._current:
            return None

        if not self._current.files:
            logger.debug("No files changed, discarding checkpoint")
            self._current = None
            return None

        self._current.description = description
        self.checkpoints.append(self._current)

        checkpoint_id = self._current.id
        self._save_checkpoint(self._current)
        self._save_index()

        logger.info("Checkpoint %s saved with %s file(s)", checkpoint_id, len(self._current.files))

        self._current = None
        self._pending_files.clear()

        return checkpoint_id

    def discard_checkpoint(self):
        """Discard the current checkpoint without saving."""
        self._current = None
        self._pending_files.clear()

    def snapshot(self, path: str, reason: str = "") -> bool:
        """Convenience: take a single-file snapshot in its own checkpoint."""
        self.begin_checkpoint(reason)
        result = self.snapshot_file(path)
        if result:
            self.finalize_checkpoint(description=reason)
        else:
            self.discard_checkpoint()
        return result

    def list_checkpoints(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent checkpoints."""
        result = []
        for cp in reversed(self.checkpoints[-limit:]):
            dt = datetime.fromtimestamp(cp.created_at)
            result.append(
                {
                    "id": cp.id,
                    "created_at": dt.isoformat(),
                    "prompt": cp.prompt[:100] + ("..." if len(cp.prompt) > 100 else ""),
                    "files_count": len(cp.files),
                    "message_count": cp.message_count,
                    "description": cp.description,
                }
            )
        return result

    def get_checkpoint(self, checkpoint_id: str) -> Checkpoint | None:
        """Get a checkpoint by ID."""
        for cp in self.checkpoints:
            if cp.id == checkpoint_id:
                return cp
        return None

    def rewind(
        self,
        checkpoint_id: str,
        mode: Literal["code", "conversation", "both"] = "both",
    ) -> dict[str, Any]:
        """Rewind to a specific checkpoint."""
        checkpoint = self.get_checkpoint(checkpoint_id)
        if not checkpoint:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")

        result = {
            "restored_files": [],
            "failed_files": [],
            "message_count": None,
            "checkpoints_removed": 0,
        }

        if mode in ("code", "both"):
            for snapshot in checkpoint.files:
                try:
                    self._restore_file(snapshot)
                    result["restored_files"].append(snapshot.path)
                except Exception as e:
                    logger.error("Failed to restore %s: %s", snapshot.path, e)
                    result["failed_files"].append({"path": snapshot.path, "error": str(e)})

        if mode in ("conversation", "both"):
            result["message_count"] = checkpoint.message_count

        cp_index = self.checkpoints.index(checkpoint)
        if cp_index < len(self.checkpoints) - 1:
            removed = self.checkpoints[cp_index + 1 :]
            result["checkpoints_removed"] = len(removed)

            for cp in removed:
                self._delete_checkpoint_file(cp.id)

            self.checkpoints = self.checkpoints[: cp_index + 1]
            self._save_index()

        logger.info(
            f"Rewound to checkpoint {checkpoint_id}: {len(result['restored_files'])} files restored"
        )

        return result

    def rewind_last(
        self, mode: Literal["code", "conversation", "both"] = "code"
    ) -> dict[str, Any] | None:
        """Rewind to the previous checkpoint."""
        if len(self.checkpoints) < 2:
            return None

        target = self.checkpoints[-2]
        return self.rewind(target.id, mode)

    def _generate_id(self) -> str:
        """Generate a unique checkpoint ID."""
        return uuid.uuid4().hex[:12]

    def _restore_file(self, snapshot: FileSnapshot):
        """Restore a file from snapshot."""
        file_path = Path(snapshot.path)

        if not snapshot.exists:
            if file_path.exists():
                file_path.unlink()
                logger.debug("Deleted file: %s", snapshot.path)
            return

        if snapshot.content is None:
            raise ValueError(f"No content available for {snapshot.path}")

        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(snapshot.content, encoding="utf-8")
        logger.debug("Restored file: %s", snapshot.path)

    def _get_checkpoint_path(self, checkpoint_id: str) -> Path:
        """Get path for checkpoint data file."""
        if self.config.compress:
            return self._base_dir / f"{checkpoint_id}.json.gz"
        return self._base_dir / f"{checkpoint_id}.json"

    def _save_checkpoint(self, checkpoint: Checkpoint):
        """Save checkpoint data to file."""
        path = self._get_checkpoint_path(checkpoint.id)
        data = json.dumps(checkpoint.to_dict(), ensure_ascii=False)

        if self.config.compress:
            with gzip.open(path, "wt", encoding="utf-8") as f:
                f.write(data)
        else:
            path.write_text(data, encoding="utf-8")

    def _load_checkpoint(self, checkpoint_id: str) -> Checkpoint | None:
        """Load checkpoint data from file."""
        path = self._get_checkpoint_path(checkpoint_id)
        if not path.exists():
            path = self._base_dir / f"{checkpoint_id}.json"
            if not path.exists():
                return None

        try:
            if path.suffix == ".gz":
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = json.loads(path.read_text(encoding="utf-8"))

            return Checkpoint.from_dict(data)
        except Exception as e:
            logger.error("Failed to load checkpoint %s: %s", checkpoint_id, e)
            return None

    def _delete_checkpoint_file(self, checkpoint_id: str):
        """Delete checkpoint data file."""
        for ext in [".json.gz", ".json"]:
            path = self._base_dir / f"{checkpoint_id}{ext}"
            if path.exists():
                path.unlink()

    def _save_index(self):
        """Save checkpoint index (metadata only)."""
        index_path = self._base_dir / "index.json"
        data = {
            "version": 1,
            "project": self.project,
            "checkpoints": [
                {
                    "id": cp.id,
                    "created_at": cp.created_at,
                    "prompt": cp.prompt[:100],
                    "files_count": len(cp.files),
                    "message_count": cp.message_count,
                }
                for cp in self.checkpoints
            ],
        }
        index_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load_index(self) -> bool:
        """Load checkpoint index. Returns True on success."""
        index_path = self._base_dir / "index.json"
        if not index_path.exists():
            return True

        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))

            for entry in data.get("checkpoints", []):
                checkpoint = self._load_checkpoint(entry["id"])
                if checkpoint:
                    self.checkpoints.append(checkpoint)
                else:
                    self.checkpoints.append(
                        Checkpoint(
                            id=entry["id"],
                            created_at=entry["created_at"],
                            prompt=entry.get("prompt", ""),
                            message_count=entry.get("message_count", 0),
                        )
                    )

            logger.info("Loaded %s checkpoints", len(self.checkpoints))
            return True

        except Exception as e:
            logger.error("Failed to load checkpoint index: %s", e)
            return False

    def _cleanup_old_checkpoints(self):
        """Remove checkpoints older than retention period."""
        if not self.config.retention_days:
            return

        cutoff = time.time() - (self.config.retention_days * 24 * 60 * 60)
        removed = 0

        new_checkpoints = []
        for cp in self.checkpoints:
            if cp.created_at < cutoff:
                self._delete_checkpoint_file(cp.id)
                removed += 1
            else:
                new_checkpoints.append(cp)

        if removed > 0:
            self.checkpoints = new_checkpoints
            self._save_index()
            logger.info("Cleaned up %s old checkpoints", removed)
