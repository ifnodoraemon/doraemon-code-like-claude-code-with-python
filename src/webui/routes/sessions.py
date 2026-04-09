"""
Sessions API Routes

Manages chat sessions.
"""

import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.core.session import SessionManager

router = APIRouter()

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
        end = total_messages if message_limit is None else min(total_messages, message_offset + message_limit)
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
