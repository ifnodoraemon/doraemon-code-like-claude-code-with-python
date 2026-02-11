"""
Model Gateway Server

FastAPI server that provides a unified API for all model providers.

Usage:
    # Start server
    python -m src.gateway.server

    # Or with uvicorn
    uvicorn src.gateway.server:app --host 0.0.0.0 --port 8000
"""

import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .router import ModelRouter
from .schema import ChatMessage, ChatRequest, ErrorResponse, ToolDefinition

logger = logging.getLogger(__name__)

# Global router instance
router: ModelRouter | None = None


def load_config() -> dict[str, Any]:
    """Load gateway configuration from environment."""
    return {
        "google": {
            "enabled": bool(os.getenv("GOOGLE_API_KEY")),
            "api_key": os.getenv("GOOGLE_API_KEY"),
        },
        "openai": {
            "enabled": bool(os.getenv("OPENAI_API_KEY")),
            "api_key": os.getenv("OPENAI_API_KEY"),
            "api_base": os.getenv("OPENAI_API_BASE"),
        },
        "anthropic": {
            "enabled": bool(os.getenv("ANTHROPIC_API_KEY")),
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
        },
        "ollama": {
            "enabled": os.getenv("OLLAMA_ENABLED", "true").lower() == "true",
            "api_base": os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
        },
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize router on startup."""
    global router
    config = load_config()
    router = ModelRouter(config)
    await router.initialize()
    logger.info("Model Gateway started")
    yield
    logger.info("Model Gateway stopped")


# Create FastAPI app
app = FastAPI(
    title="Doraemon Model Gateway",
    description="Unified API for multiple AI model providers",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware - restrict to configured origins
ALLOWED_ORIGINS = os.getenv(
    "DORAEMON_CORS_ORIGINS",
    "http://localhost:5173,http://localhost:8000,http://127.0.0.1:5173,http://127.0.0.1:8000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# Gateway API key (optional)
GATEWAY_API_KEY = os.getenv("DORAEMON_GATEWAY_KEY")


def verify_api_key(authorization: str | None) -> bool:
    """Verify API key if configured."""
    if not GATEWAY_API_KEY:
        return True  # No key configured, allow all

    if not authorization:
        return False

    # Support "Bearer <key>" or just "<key>"
    if authorization.startswith("Bearer "):
        key = authorization[7:]
    else:
        key = authorization

    return secrets.compare_digest(key, GATEWAY_API_KEY)


# Pydantic models for API
class ChatCompletionMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatCompletionTool(BaseModel):
    type: str = "function"
    function: dict


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatCompletionMessage]
    tools: list[ChatCompletionTool] | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    stream: bool = False
    stop: list[str] | None = None
    top_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None


@app.get("/health")
async def health():
    """Health check endpoint."""
    if router:
        provider_health = await router.health_check()
        return {
            "status": "healthy",
            "providers": provider_health,
        }
    return {"status": "initializing"}


@app.get("/v1/models")
async def list_models(
    provider: str | None = None,
    authorization: str | None = Header(None),
):
    """List available models."""
    if not verify_api_key(authorization):
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not router:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    models = router.list_models(provider)
    return {
        "object": "list",
        "data": [
            {
                "id": m.id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": m.provider,
                "permission": [],
                "root": m.id,
                "parent": None,
                # Extended info
                "name": m.name,
                "description": m.description,
                "context_window": m.context_window,
                "max_output": m.max_output,
                "input_price": m.input_price,
                "output_price": m.output_price,
                "capabilities": m.capabilities,
                "aliases": m.aliases,
            }
            for m in models
        ],
    }


@app.get("/v1/models/{model_id}")
async def get_model(
    model_id: str,
    authorization: str | None = Header(None),
):
    """Get information about a specific model."""
    if not verify_api_key(authorization):
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not router:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    model = router.get_model_info(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    return {
        "id": model.id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": model.provider,
        "name": model.name,
        "description": model.description,
        "context_window": model.context_window,
        "max_output": model.max_output,
        "input_price": model.input_price,
        "output_price": model.output_price,
        "capabilities": model.capabilities,
        "aliases": model.aliases,
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: str | None = Header(None),
):
    """Create a chat completion."""
    if not verify_api_key(authorization):
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not router:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    # Convert to internal format
    messages = []
    for m in request.messages:
        tool_calls = None
        if m.tool_calls:
            from .schema import ToolCall
            tool_calls = [ToolCall.from_dict(tc) for tc in m.tool_calls]

        messages.append(ChatMessage(
            role=m.role,
            content=m.content,
            tool_calls=tool_calls,
            tool_call_id=m.tool_call_id,
            name=m.name,
        ))

    tools = None
    if request.tools:
        tools = [
            ToolDefinition(
                name=t.function["name"],
                description=t.function.get("description", ""),
                parameters=t.function.get("parameters", {}),
            )
            for t in request.tools
        ]

    chat_request = ChatRequest(
        model=request.model,
        messages=messages,
        tools=tools,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stream=request.stream,
        stop=request.stop,
        top_p=request.top_p,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
    )

    if request.stream:
        return StreamingResponse(
            stream_response(chat_request),
            media_type="text/event-stream",
        )

    response = await router.chat(chat_request)

    if isinstance(response, ErrorResponse):
        raise HTTPException(
            status_code=500,
            detail=response.to_dict(),
        )

    return response.to_dict()


async def stream_response(request: ChatRequest):
    """Generate SSE stream for chat completion."""
    if router is None:
        yield f"data: {json.dumps({'error': 'Gateway not initialized'})}\n\n"
        return
    try:
        async for chunk in router.chat_stream(request):
            if isinstance(chunk, ErrorResponse):
                yield f"data: {json.dumps(chunk.to_dict())}\n\n"
                break

            yield f"data: {json.dumps(chunk.to_dict())}\n\n"
    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield f"data: {json.dumps({'error': 'Internal stream error'})}\n\n"

    yield "data: [DONE]\n\n"


@app.get("/v1/providers")
async def list_providers(authorization: str | None = Header(None)):
    """List enabled providers."""
    if not verify_api_key(authorization):
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not router:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    return {
        "providers": router.get_providers(),
    }


def main():
    """Run the gateway server."""
    import uvicorn

    host = os.getenv("GATEWAY_HOST", "0.0.0.0")
    port = int(os.getenv("GATEWAY_PORT", "8000"))

    print(f"Starting Doraemon Model Gateway on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
