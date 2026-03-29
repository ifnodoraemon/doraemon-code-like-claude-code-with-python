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

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .router import ModelRouter
from .schema import ChatMessage, ChatRequest, ErrorResponse, ToolDefinition

logger = logging.getLogger(__name__)


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
            "api_base": os.getenv("ANTHROPIC_API_BASE"),
        },
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize router on startup, clean up on shutdown."""
    config = load_config()
    app.state.router = ModelRouter(config)
    await app.state.router.initialize()
    logger.info("Model Gateway started")
    yield
    if app.state.router:
        await app.state.router.close()
        app.state.router = None
    logger.info("Model Gateway stopped")


# Create FastAPI app
app = FastAPI(
    title="Model Gateway",
    description="Unified API for multiple AI model providers",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware - restrict to configured origins
ALLOWED_ORIGINS = os.getenv(
    "AGENT_CORS_ORIGINS",
    "http://localhost:5173,http://localhost:8000,http://127.0.0.1:5173,http://127.0.0.1:8000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# Request body size limit (default 10 MB)
MAX_REQUEST_BODY_BYTES = int(os.getenv("AGENT_MAX_REQUEST_BYTES", str(10 * 1024 * 1024)))


@app.middleware("http")
async def limit_request_body(request: Request, call_next):
    """Reject requests whose body exceeds the configured size limit."""
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_REQUEST_BODY_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request body too large"},
                )
            # Content-Length is present and within limits — skip full body read
            return await call_next(request)
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid Content-Length header"},
            )

    # No Content-Length (e.g. chunked transfer) — must read body to check size
    body = await request.body()
    if len(body) > MAX_REQUEST_BODY_BYTES:
        return JSONResponse(
            status_code=413,
            content={"detail": "Request body too large"},
        )

    return await call_next(request)


# Gateway API key (optional)
GATEWAY_API_KEY = os.getenv("AGENT_API_KEY")


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
class FunctionDefinition(BaseModel):
    """Typed function definition for tool calls."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = {}


class ToolCallInfo(BaseModel):
    """Typed tool call information."""

    id: str
    type: str = "function"
    function: "ToolCallFunction"


class ToolCallFunction(BaseModel):
    """Typed function call payload for assistant tool calls."""

    name: str
    arguments: str | dict[str, Any] = "{}"


class ChatCompletionMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[ToolCallInfo] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatCompletionTool(BaseModel):
    type: str = "function"
    function: FunctionDefinition


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


class AnthropicToolDefinition(BaseModel):
    name: str
    description: str = ""
    input_schema: dict[str, Any] = {}


class AnthropicMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]]


class AnthropicMessagesRequest(BaseModel):
    model: str
    messages: list[AnthropicMessage]
    system: str | None = None
    tools: list[AnthropicToolDefinition] | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    stream: bool = False
    stop_sequences: list[str] | None = None
    top_p: float | None = None


def _parse_tool_call_arguments(arguments: str | dict[str, Any] | None) -> dict[str, Any]:
    """Normalize tool call arguments from either JSON string or dict form."""
    if isinstance(arguments, dict):
        return arguments
    if not arguments:
        return {}
    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        logger.warning("Failed to parse tool call arguments: %s", str(arguments)[:200])
        return {}


def _build_openai_messages(request: ChatCompletionRequest) -> list[ChatMessage]:
    """Convert OpenAI-format messages to the unified gateway schema."""
    messages = []
    for m in request.messages:
        tool_calls = None
        if m.tool_calls:
            from .schema import ToolCall

            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=_parse_tool_call_arguments(tc.function.arguments),
                )
                for tc in m.tool_calls
            ]

        messages.append(
            ChatMessage(
                role=m.role,
                content=m.content,
                tool_calls=tool_calls,
                tool_call_id=m.tool_call_id,
                name=m.name,
            )
        )
    return messages


def _build_openai_tools(
    tools: list[ChatCompletionTool] | None,
) -> list[ToolDefinition] | None:
    """Convert OpenAI-format tools to the unified gateway schema."""
    if not tools:
        return None
    return [
        ToolDefinition(
            name=t.function.name,
            description=t.function.description,
            parameters=t.function.parameters,
        )
        for t in tools
    ]


def _build_chat_request_from_openai(request: ChatCompletionRequest) -> ChatRequest:
    """Translate an OpenAI-compatible chat request to the unified gateway schema."""
    return ChatRequest(
        model=request.model,
        messages=_build_openai_messages(request),
        tools=_build_openai_tools(request.tools),
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stream=request.stream,
        stop=request.stop,
        top_p=request.top_p,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
    )


def _build_chat_request_from_anthropic(request: AnthropicMessagesRequest) -> ChatRequest:
    """Translate an Anthropic-compatible messages request to the unified gateway schema."""
    from .schema import ToolCall

    messages: list[ChatMessage] = []

    if request.system:
        messages.append(ChatMessage(role="system", content=request.system))

    for message in request.messages:
        content = message.content
        if isinstance(content, str):
            messages.append(ChatMessage(role=message.role, content=content))
            continue

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in content:
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text")
                if text:
                    text_parts.append(text)
            elif block_type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.get("id", ""),
                        name=block.get("name", ""),
                        arguments=block.get("input") or {},
                    )
                )
            elif block_type == "tool_result":
                result_content = block.get("content")
                if isinstance(result_content, list):
                    result_content = "".join(
                        part.get("text", "")
                        for part in result_content
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
                messages.append(
                    ChatMessage(
                        role="tool",
                        content=result_content if isinstance(result_content, str) else "",
                        tool_call_id=block.get("tool_use_id"),
                    )
                )

        if text_parts or tool_calls or message.role != "user":
            messages.append(
                ChatMessage(
                    role=message.role,
                    content="".join(text_parts) or None,
                    tool_calls=tool_calls or None,
                )
            )

    tools = None
    if request.tools:
        tools = [
            ToolDefinition(
                name=tool.name,
                description=tool.description,
                parameters=tool.input_schema,
            )
            for tool in request.tools
        ]

    return ChatRequest(
        model=request.model,
        messages=messages,
        tools=tools,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stream=request.stream,
        stop=request.stop_sequences,
        top_p=request.top_p,
    )


def _anthropic_stop_reason(finish_reason: str | None) -> str:
    """Map unified finish reasons to Anthropic stop reasons."""
    if finish_reason == "tool_calls":
        return "tool_use"
    if finish_reason == "length":
        return "max_tokens"
    return "end_turn"


def _convert_chat_response_to_anthropic(response) -> dict[str, Any]:
    """Convert a unified gateway response to Anthropic's message format."""
    choice = response.choices[0] if response.choices else None
    if isinstance(choice, dict):
        message = choice.get("message")
        finish_reason = choice.get("finish_reason")
    else:
        message = choice.message if choice else None
        finish_reason = choice.finish_reason if choice else None

    if isinstance(message, dict):
        text_content = message.get("content")
        raw_tool_calls = message.get("tool_calls") or []
        tool_calls = [
            {
                "id": tc.get("id", ""),
                "name": (tc.get("function") or {}).get("name", ""),
                "arguments": (tc.get("function") or {}).get("arguments", {}),
            }
            for tc in raw_tool_calls
        ]
    else:
        text_content = message.content if message else None
        tool_calls = (message.tool_calls or []) if message else []

    content: list[dict[str, Any]] = []
    if text_content:
        content.append({"type": "text", "text": text_content})
    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            content.append(
                {
                    "type": "tool_use",
                    "id": tool_call.get("id", ""),
                    "name": tool_call.get("name", ""),
                    "input": tool_call.get("arguments") or {},
                }
            )
        else:
            content.append(
                {
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "input": tool_call.arguments,
                }
            )

    usage = response.usage.to_dict() if response.usage else {}
    return {
        "id": response.id,
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": response.model,
        "stop_reason": _anthropic_stop_reason(finish_reason),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def _format_sse(event: str, data: dict[str, Any] | str) -> str:
    """Format an Anthropic-style SSE event."""
    payload = data if isinstance(data, str) else json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


@app.get("/health")
async def health(request: Request):
    """Health check endpoint."""
    router = getattr(request.app.state, "router", None)
    if router:
        provider_health = await router.health_check()
        return {
            "status": "healthy",
            "providers": provider_health,
        }
    return {"status": "initializing"}


@app.get("/v1/models")
async def list_models(
    request: Request,
    provider: str | None = None,
    authorization: str | None = Header(None),
):
    """List available models."""
    if not verify_api_key(authorization):
        raise HTTPException(status_code=401, detail="Invalid API key")

    router = getattr(request.app.state, "router", None)
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
    request: Request,
    authorization: str | None = Header(None),
):
    """Get information about a specific model."""
    if not verify_api_key(authorization):
        raise HTTPException(status_code=401, detail="Invalid API key")

    router = getattr(request.app.state, "router", None)
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
    raw_request: Request,
    authorization: str | None = Header(None),
):
    """Create a chat completion."""
    if not verify_api_key(authorization):
        raise HTTPException(status_code=401, detail="Invalid API key")

    router = getattr(raw_request.app.state, "router", None)
    if not router:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    chat_request = _build_chat_request_from_openai(request)

    if request.stream:
        return StreamingResponse(
            stream_response(chat_request, router, preferred_provider="openai"),
            media_type="text/event-stream",
        )

    response = await router.chat(chat_request, preferred_provider="openai")

    if isinstance(response, ErrorResponse):
        raise HTTPException(
            status_code=500,
            detail=response.to_dict(),
        )

    return response.to_dict()


@app.post("/v1/messages")
async def anthropic_messages(
    request: AnthropicMessagesRequest,
    raw_request: Request,
    x_api_key: str | None = Header(None),
    authorization: str | None = Header(None),
):
    """Create a message using Anthropic-compatible request/response bodies."""
    auth_header = authorization or x_api_key
    if not verify_api_key(auth_header):
        raise HTTPException(status_code=401, detail="Invalid API key")

    router = getattr(raw_request.app.state, "router", None)
    if not router:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    chat_request = _build_chat_request_from_anthropic(request)

    if request.stream:
        return StreamingResponse(
            stream_anthropic_response(chat_request, router),
            media_type="text/event-stream",
        )

    response = await router.chat(chat_request, preferred_provider="anthropic")

    if isinstance(response, ErrorResponse):
        raise HTTPException(status_code=500, detail=response.to_dict())

    return _convert_chat_response_to_anthropic(response)


async def stream_anthropic_response(request: ChatRequest, router: ModelRouter):
    """Generate Anthropic-compatible SSE events for a streaming message request."""
    message_id: str | None = None
    content_block_started = False
    tool_block_index = 0

    async for chunk in router.chat_stream(request, preferred_provider="anthropic"):
        if isinstance(chunk, ErrorResponse):
            yield _format_sse(
                "error",
                {"type": "error", "error": {"type": chunk.code, "message": chunk.error}},
            )
            break

        if not message_id:
            message_id = chunk.id
            yield _format_sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": chunk.id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": chunk.model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": chunk.usage.prompt_tokens if chunk.usage else 0,
                            "output_tokens": 0,
                        },
                    },
                },
            )

        if chunk.delta_content:
            if not content_block_started:
                yield _format_sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
                content_block_started = True
            yield _format_sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": chunk.delta_content},
                },
            )

        if chunk.delta_tool_calls:
            for tool_call in chunk.delta_tool_calls:
                yield _format_sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": tool_block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tool_call.id,
                            "name": tool_call.name,
                            "input": tool_call.arguments,
                        },
                    },
                )
                tool_block_index += 1

        if chunk.finish_reason:
            if content_block_started:
                yield _format_sse(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": 0},
                )
            yield _format_sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": _anthropic_stop_reason(chunk.finish_reason),
                        "stop_sequence": None,
                    },
                    "usage": {
                        "output_tokens": chunk.usage.completion_tokens if chunk.usage else 0,
                    },
                },
            )
            yield _format_sse("message_stop", {"type": "message_stop"})
            return


async def stream_response(
    request: ChatRequest,
    router: ModelRouter,
    preferred_provider: str | None = None,
):
    """Generate SSE stream for chat completion."""
    try:
        async for chunk in router.chat_stream(request, preferred_provider=preferred_provider):
            if isinstance(chunk, ErrorResponse):
                yield f"data: {json.dumps(chunk.to_dict())}\n\n"
                break

            yield f"data: {json.dumps(chunk.to_dict())}\n\n"
    except Exception as e:
        logger.error(f"Stream error: {e}")
        error_payload = json.dumps({"error": "Internal stream error"})
        yield f"data: {error_payload}\n\n"

    yield "data: [DONE]\n\n"


@app.get("/v1/providers")
async def list_providers(
    request: Request,
    authorization: str | None = Header(None),
):
    """List enabled providers."""
    if not verify_api_key(authorization):
        raise HTTPException(status_code=401, detail="Invalid API key")

    router = getattr(request.app.state, "router", None)
    if not router:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    return {
        "providers": router.get_providers(),
    }


def main():
    """Run the gateway server."""
    import uvicorn

    host = os.getenv("AGENT_GATEWAY_HOST", "0.0.0.0")
    port = int(os.getenv("AGENT_GATEWAY_PORT", "8000"))

    print(f"Starting Model Gateway on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
