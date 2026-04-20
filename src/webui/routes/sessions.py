"""
Sessions API Routes

Manages chat sessions, diffs, and undo.
"""

import os
import re
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.core.session import SessionManager

router = APIRouter()

_ACTIVE_STREAMS: dict[str, float] = {}


def mark_stream_active(session_id: str) -> None:
    _ACTIVE_STREAMS[session_id] = time.time()


def mark_stream_finished(session_id: str) -> None:
    _ACTIVE_STREAMS.pop(session_id, None)


_RUN_SUMMARY_OMIT_FIELDS = {
    "task_graph",
    "task_summaries",
    "worker_assignments",
    "executed_task_ids",
    "completed_task_ids",
    "failed_task_ids",
}
_RUN_SUMMARY_COUNT_FIELDS = {
    "task_graph": "task_graph_count",
    "task_summaries": "task_summary_count",
    "worker_assignments": "worker_assignment_count",
    "executed_task_ids": "executed_task_count",
    "completed_task_ids": "completed_task_count",
    "failed_task_ids": "failed_task_count",
}


class SessionResponse(BaseModel):
    id: str
    name: str | None
    message_count: int
    updated_at: float


def _serialize_run(run: dict, *, include_details: bool) -> dict:
    payload = dict(run)
    if include_details:
        return payload
    for source_field, count_field in _RUN_SUMMARY_COUNT_FIELDS.items():
        value = payload.get(source_field)
        if isinstance(value, dict | list):
            payload[count_field] = len(value)
    for key in _RUN_SUMMARY_OMIT_FIELDS:
        payload.pop(key, None)
    return payload


@router.get("", response_model=list[SessionResponse], include_in_schema=False)
@router.get("/", response_model=list[SessionResponse])
async def list_sessions(project: str = "default", limit: int = 50):
    """List recent sessions."""
    limit = min(max(1, limit), 100)
    mgr = SessionManager(base_dir=Path.cwd() / ".agent" / "sessions")
    sessions = mgr.list_sessions(project=project, limit=limit)

    return [
        SessionResponse(
            id=s.id, name=s.name, message_count=s.message_count, updated_at=s.updated_at
        )
        for s in sessions
    ]


@router.get("/{session_id}")
async def get_session(
    session_id: str,
    include_messages: bool = True,
    message_limit: int | None = None,
    message_offset: int | None = None,
):
    """Get session details."""
    # Validate session_id format to prevent IDOR
    if not re.match(r"^[a-zA-Z0-9_-]+$", session_id):
        raise HTTPException(status_code=400, detail="Invalid session ID format")

    mgr = SessionManager(base_dir=Path.cwd() / ".agent" / "sessions")
    session = mgr.resume_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    total_messages = len(session.messages)
    if message_limit is not None:
        message_limit = min(max(1, message_limit), 500)

    if not include_messages:
        message_offset = 0
        messages = []
    elif message_offset is not None:
        message_offset = min(max(0, message_offset), total_messages)
        end = (
            total_messages
            if message_limit is None
            else min(total_messages, message_offset + message_limit)
        )
        messages = list(session.messages[message_offset:end])
    elif message_limit is not None:
        message_offset = max(0, total_messages - message_limit)
        messages = list(session.messages[message_offset:])
    else:
        message_offset = 0
        messages = list(session.messages)

    return {
        "id": session.metadata.id,
        "name": session.metadata.name,
        "messages": messages,
        "message_count": total_messages,
        "message_offset": message_offset,
        "has_more_messages": bool(include_messages and message_offset > 0),
        "orchestration_state": _serialize_run(session.orchestration_state, include_details=False),
        "orchestration_runs": [
            _serialize_run(run, include_details=False) for run in session.orchestration_runs
        ],
        "active_orchestration_run_id": session.active_orchestration_run_id,
    }


@router.get("/{session_id}/runs/{run_id}")
async def get_session_run(session_id: str, run_id: str):
    """Get full details for a single orchestration run."""
    if not re.match(r"^[a-zA-Z0-9_-]+$", session_id):
        raise HTTPException(status_code=400, detail="Invalid session ID format")
    if not re.match(r"^[a-zA-Z0-9_-]+$", run_id):
        raise HTTPException(status_code=400, detail="Invalid run ID format")

    mgr = SessionManager(base_dir=Path.cwd() / ".agent" / "sessions")
    session = mgr.resume_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    for run in reversed(session.orchestration_runs):
        if run.get("run_id") == run_id:
            return {"run": _serialize_run(run, include_details=True)}

    if session.orchestration_state.get("run_id") == run_id:
        return {"run": _serialize_run(session.orchestration_state, include_details=True)}

    raise HTTPException(status_code=404, detail="Run not found")


@router.get("/{session_id}/diff")
async def get_session_diff(
    session_id: str,
    checkpoint_id: str | None = None,
    include_content: bool = True,
):
    """Return file changes made in a session, derived from checkpoints."""
    if not re.match(r"^[a-zA-Z0-9_-]+$", session_id):
        raise HTTPException(status_code=400, detail="Invalid session ID format")

    from src.core.checkpoint import CheckpointManager

    mgr = SessionManager(base_dir=Path.cwd() / ".agent" / "sessions")
    session = mgr.resume_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    cp_mgr = CheckpointManager(project=session.metadata.project or "default")
    session_cp_ids = getattr(session.metadata, "checkpoints", None) or []

    checkpoints_data: list[dict[str, Any]] = []

    target_ids = [checkpoint_id] if checkpoint_id else session_cp_ids
    if not target_ids:
        created_at = getattr(session.metadata, "created_at", 0) or 0
        target_ids = [
            cp.id for cp in cp_mgr.checkpoints if getattr(cp, "created_at", 0) or 0 >= created_at
        ]

    for cp_id in target_ids:
        cp = cp_mgr.get_checkpoint(cp_id) if hasattr(cp_mgr, "get_checkpoint") else None
        if cp is None:
            continue
        files: list[dict[str, Any]] = []
        for snap in cp.files:
            file_entry: dict[str, Any] = {"path": snap.path}
            before: dict[str, Any] = {"exists": snap.exists}
            if include_content:
                before["content"] = snap.content
            before["size"] = snap.size
            before["mtime"] = snap.mtime

            after: dict[str, Any] = {"exists": False, "content": None, "size": None, "mtime": None}
            try:
                snap_path = snap.path
                if not os.path.isabs(snap_path):
                    continue
                resolved = os.path.realpath(snap_path)
                current = Path(resolved)
                if current.exists():
                    after["exists"] = True
                    after["mtime"] = current.stat().st_mtime
                    after["size"] = current.stat().st_size
                    if include_content:
                        after["content"] = current.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass

            file_entry["before"] = before
            file_entry["after"] = after
            files.append(file_entry)

        checkpoints_data.append(
            {
                "id": cp.id,
                "created_at": getattr(cp, "created_at", None),
                "prompt": getattr(cp, "prompt", None),
                "description": getattr(cp, "description", ""),
                "files": files,
            }
        )

    return {
        "session_id": session_id,
        "checkpoint_count": len(checkpoints_data),
        "checkpoints": checkpoints_data,
    }


class UndoRequest(BaseModel):
    checkpoint_id: str | None = None
    mode: str = "code"
    dry_run: bool = False


@router.post("/{session_id}/undo")
async def undo_session(session_id: str, request: UndoRequest):
    """Revert the most recent write operation by rewinding to a checkpoint."""
    if not re.match(r"^[a-zA-Z0-9_-]+$", session_id):
        raise HTTPException(status_code=400, detail="Invalid session ID format")

    if session_id in _ACTIVE_STREAMS:
        raise HTTPException(
            status_code=409, detail="Cannot undo while session has an active agent stream"
        )

    from src.core.checkpoint import CheckpointManager

    mgr = SessionManager(base_dir=Path.cwd() / ".agent" / "sessions")
    session = mgr.resume_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    cp_mgr = CheckpointManager(project=session.metadata.project or "default")
    session_cp_ids = getattr(session.metadata, "checkpoints", None) or []

    target_id = request.checkpoint_id or (session_cp_ids[-1] if session_cp_ids else None)
    if not target_id:
        try:
            target_id = cp_mgr.checkpoints[-1].id
        except (IndexError, AttributeError):
            raise HTTPException(
                status_code=404, detail="No checkpoints available for undo"
            ) from None

    if request.dry_run:
        cp = cp_mgr.get_checkpoint(target_id) if hasattr(cp_mgr, "get_checkpoint") else None
        restored = [f.path for f in (cp.files if cp else [])] if cp else []
        return {
            "checkpoint_id": target_id,
            "mode": request.mode,
            "restored_files": restored,
            "deleted_files": [],
            "failed_files": [],
            "dry_run": True,
        }

    try:
        result = cp_mgr.rewind(target_id, mode=request.mode)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Undo failed: {e}") from e

    restored = getattr(result, "restored_files", [])
    deleted = getattr(result, "deleted_files", [])
    failed = getattr(result, "failed_files", [])

    return {
        "checkpoint_id": target_id,
        "mode": request.mode,
        "restored_files": restored,
        "deleted_files": deleted,
        "failed_files": failed,
        "dry_run": False,
    }
