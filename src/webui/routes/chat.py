"""
Chat API Routes

Handles chat interactions with SSE streaming.
"""

import json
import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.core.context_manager import ContextConfig, ContextManager
from src.core.llm.model_client import ModelClient
from src.core.model_utils import Message, ToolDefinition
from src.core.tool_selector import ToolSelector
from src.host.tools import get_default_registry

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    project: str = "default"
    model: str | None = None

    @property
    def validated_message(self) -> str:
        if len(self.message) > 200_000:
            raise ValueError("Message too long")
        return self.message


# Initializer helpers (simplified for now)
async def get_model_client():
    return await ModelClient.create()


def get_tools_for_mode(mode: str = "build"):
    selector = ToolSelector()
    tool_names = selector.get_tools_for_mode(mode)
    registry = get_default_registry(tool_names)
    genai_tools = registry.get_genai_tools(tool_names)

    # Convert to ToolDefinition
    definitions = []
    for tool in genai_tools:
        if hasattr(tool, "name"):
            definitions.append(
                ToolDefinition(
                    name=tool.name,
                    description=getattr(tool, "description", ""),
                    parameters=getattr(tool, "parameters", {}) or {},
                )
            )
        elif isinstance(tool, dict):
            definitions.append(
                ToolDefinition(
                    name=tool.get("name", ""),
                    description=tool.get("description", ""),
                    parameters=tool.get("parameters", {}),
                )
            )
    return definitions


@router.post("/")
async def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint with SSE streaming.
    """
    try:
        client = await get_model_client()
        tools = get_tools_for_mode("build")

        # Initialize context (simplified)
        ctx_config = ContextConfig(auto_save=True)
        ctx = ContextManager(project=request.project, config=ctx_config)

        # Add user message
        ctx.add_user_message(request.message)

        # Prepare messages
        messages = [
            Message(role="system", content="You are Doraemon Code, an intelligent AI assistant."),
            Message(role="user", content=request.message),
        ]

        # Generator for SSE
        async def event_generator() -> AsyncGenerator[str, None]:
            try:
                # Stream response
                async for chunk in client.chat_stream(messages, tools=tools):
                    data = {
                        "content": chunk.content,
                        "tool_calls": chunk.tool_calls,
                        "finish_reason": chunk.finish_reason,
                    }
                    yield f"data: {json.dumps(data)}\n\n"

                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                err_data = {"error": str(e)}
                yield f"data: {json.dumps(err_data)}\n\n"
            finally:
                await client.close()

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Chat error: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e
