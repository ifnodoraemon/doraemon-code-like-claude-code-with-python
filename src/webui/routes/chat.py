"""
Chat API Routes

Handles chat interactions with SSE streaming.
"""

import json
import logging
import re
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.agent.adapter import AgentSession

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    project: str = "default"
    model: str | None = None
    execution_mode: Literal["turn", "orchestrate"] = "turn"
    max_workers: int = 2
    context: dict[str, Any] | None = None

    @property
    def validated_message(self) -> str:
        return self.message


@router.post("/")
async def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint with SSE streaming.
    """
    try:
        message = request.validated_message
        if request.session_id and not re.match(r"^[a-zA-Z0-9_-]+$", request.session_id):
            raise HTTPException(status_code=400, detail="Invalid session ID format")
        session = AgentSession(
            model_client=None,
            registry=None,
            project=request.project,
            mode="build",
            project_dir=Path.cwd(),
            enable_trace=False,
            session_id=request.session_id,
            model_name=request.model,
        )

        async def event_generator() -> AsyncGenerator[str, None]:
            try:
                if request.execution_mode == "orchestrate":
                    result = await session.orchestrate(
                        message,
                        context=request.context,
                        max_workers=max(1, request.max_workers),
                    )
                    task_manager = session.get_task_manager()
                    task_tree = task_manager.get_task_tree() if task_manager is not None else []
                    yield f"data: {json.dumps({'type': 'orchestration', 'session_id': session.session_id, 'content': result.summary, 'result': result.to_dict(), 'task_graph': task_tree})}\n\n"
                else:
                    async for event in session.turn_stream(message):
                        event_type = event.get("type", "event")
                        data = {"type": event_type, "session_id": session.session_id}

                        if event_type == "response":
                            data.update(
                                {
                                    "content": event.get("content"),
                                    "tool_calls": None,
                                    "finish_reason": "stop",
                                }
                            )
                        elif event_type == "tool_call":
                            data.update(
                                {
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "name": event.get("name"),
                                            "arguments": event.get("args", {}),
                                        }
                                    ],
                                    "finish_reason": None,
                                }
                            )
                        elif event_type == "error":
                            data["error"] = event.get("error")
                        elif event_type == "done":
                            data.update(
                                {
                                    "content": None,
                                    "tool_calls": None,
                                    "finish_reason": "stop",
                                }
                            )
                        else:
                            data["content"] = event.get("content")

                        yield f"data: {json.dumps(data)}\n\n"

                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                if request.execution_mode == "orchestrate":
                    task_manager = session.get_task_manager()
                    task_tree = task_manager.get_task_tree() if task_manager is not None else []
                    error_summary = f"Orchestration failed: {e}"
                    err_data = {
                        "type": "orchestration",
                        "session_id": session.session_id,
                        "content": error_summary,
                        "result": {
                            "success": False,
                            "summary": error_summary,
                            "worker_assignments": {},
                        },
                        "task_graph": task_tree,
                    }
                else:
                    err_data = {"error": str(e), "session_id": session.session_id}
                yield f"data: {json.dumps(err_data)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                await session.aclose()

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e
