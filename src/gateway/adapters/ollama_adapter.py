"""
Ollama Adapter

Translates unified API to Ollama format for local models.
"""

import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..schema import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Choice,
    FinishReason,
    ModelInfo,
    Role,
    StreamChunk,
    ToolCall,
    Usage,
)
from .base import AdapterConfig, BaseAdapter

logger = logging.getLogger(__name__)


# Common Ollama models (user can add more)
OLLAMA_MODELS = [
    ModelInfo(
        id="llama3.3:70b",
        name="Llama 3.3 70B",
        provider="ollama",
        description="Meta's latest large model",
        context_window=128_000,
        max_output=8_192,
        input_price=0.0,  # Local, free
        output_price=0.0,
        capabilities=["text", "code", "reasoning", "function_calling"],
        aliases=["llama3.3", "llama-3.3"],
    ),
    ModelInfo(
        id="qwen2.5:72b",
        name="Qwen 2.5 72B",
        provider="ollama",
        description="Alibaba's large model",
        context_window=128_000,
        max_output=8_192,
        input_price=0.0,
        output_price=0.0,
        capabilities=["text", "code", "reasoning", "function_calling"],
        aliases=["qwen2.5", "qwen-2.5"],
    ),
    ModelInfo(
        id="deepseek-r1:70b",
        name="DeepSeek R1 70B",
        provider="ollama",
        description="DeepSeek's reasoning model",
        context_window=64_000,
        max_output=8_192,
        input_price=0.0,
        output_price=0.0,
        capabilities=["text", "code", "reasoning"],
        aliases=["deepseek-r1", "r1"],
    ),
    ModelInfo(
        id="codellama:34b",
        name="Code Llama 34B",
        provider="ollama",
        description="Code-specialized Llama",
        context_window=16_000,
        max_output=4_096,
        input_price=0.0,
        output_price=0.0,
        capabilities=["text", "code"],
        aliases=["codellama"],
    ),
]


class OllamaAdapter(BaseAdapter):
    """Adapter for Ollama local models."""

    provider_name = "ollama"

    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self._base_url = config.api_base or "http://localhost:11434"
        self._dynamic_models: list[ModelInfo] = []

    async def initialize(self) -> None:
        """Initialize Ollama client and fetch available models."""
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self.config.timeout,
        )

        # Fetch available models from Ollama
        try:
            await self._fetch_models()
            logger.info(f"Ollama adapter initialized with {len(self._dynamic_models)} models")
        except Exception as e:
            logger.warning(f"Could not fetch Ollama models: {e}")

    async def close(self) -> None:
        """Close the httpx client to release resources."""
        if hasattr(self, "_client") and self._client:
            await self._client.aclose()
            self._client = None

    async def _fetch_models(self) -> None:
        """Fetch available models from Ollama API."""
        try:
            response = await self._client.get("/api/tags")
            if response.status_code == 200:
                data = response.json()
                self._dynamic_models = []
                for model in data.get("models", []):
                    name = model.get("name", "")
                    self._dynamic_models.append(
                        ModelInfo(
                            id=name,
                            name=name,
                            provider="ollama",
                            description=f"Local Ollama model: {name}",
                            context_window=model.get("context_length", 4096),
                            max_output=4096,
                            input_price=0.0,
                            output_price=0.0,
                            capabilities=["text", "code"],
                            aliases=[],
                        )
                    )
        except Exception as e:
            logger.warning(f"Failed to fetch Ollama models: {e}")

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send chat request to Ollama."""
        messages = [self._convert_message(m) for m in request.messages]

        payload = {
            "model": request.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": request.temperature,
            },
        }

        if request.max_tokens:
            payload["options"]["num_predict"] = request.max_tokens

        if request.tools:
            payload["tools"] = [self._convert_tool(t) for t in request.tools]

        response = await self._client.post("/api/chat", json=payload)
        response.raise_for_status()

        data = response.json()
        return self._convert_response(data, request.model)

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """Stream chat response from Ollama."""
        messages = [self._convert_message(m) for m in request.messages]

        payload = {
            "model": request.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": request.temperature,
            },
        }

        if request.max_tokens:
            payload["options"]["num_predict"] = request.max_tokens

        if request.tools:
            payload["tools"] = [self._convert_tool(t) for t in request.tools]

        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        total_tokens = 0

        async with self._client.stream("POST", "/api/chat", json=payload) as response:
            async for line in response.aiter_lines():
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                delta_content = None
                delta_tool_calls = None
                finish_reason = None
                usage = None

                # Text content
                if "message" in data and "content" in data["message"]:
                    delta_content = data["message"]["content"]

                # Tool calls
                if "message" in data and "tool_calls" in data["message"]:
                    delta_tool_calls = []
                    for tc in data["message"]["tool_calls"]:
                        delta_tool_calls.append(
                            ToolCall(
                                id=f"call_{uuid.uuid4().hex[:8]}",
                                name=tc["function"]["name"],
                                arguments=tc["function"].get("arguments", {}),
                            )
                        )

                # Check if done
                if data.get("done", False):
                    finish_reason = FinishReason.STOP.value
                    if "eval_count" in data:
                        total_tokens = data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                        usage = Usage(
                            prompt_tokens=data.get("prompt_eval_count", 0),
                            completion_tokens=data.get("eval_count", 0),
                            total_tokens=total_tokens,
                        )

                yield StreamChunk(
                    id=response_id,
                    model=request.model,
                    delta_content=delta_content,
                    delta_tool_calls=delta_tool_calls,
                    finish_reason=finish_reason,
                    usage=usage,
                )

    def get_models(self) -> list[ModelInfo]:
        """Get available Ollama models."""
        # Return dynamic models if available, otherwise return defaults
        if self._dynamic_models:
            return self._dynamic_models
        return OLLAMA_MODELS

    def _convert_message(self, msg: ChatMessage) -> dict:
        """Convert unified message to Ollama format."""
        role = msg.role if isinstance(msg.role, str) else msg.role.value

        result = {
            "role": role,
            "content": msg.content or "",
        }

        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
                }
                for tc in msg.tool_calls
            ]

        return result

    def _convert_tool(self, tool: Any) -> dict:
        """Convert unified tool to Ollama format."""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }

    def _convert_response(self, data: dict, model: str) -> ChatResponse:
        """Convert Ollama response to unified format."""
        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        content = None
        tool_calls = None

        if "message" in data:
            message_data = data["message"]
            content = message_data.get("content")

            if "tool_calls" in message_data:
                tool_calls = []
                for tc in message_data["tool_calls"]:
                    tool_calls.append(
                        ToolCall(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            name=tc["function"]["name"],
                            arguments=tc["function"].get("arguments", {}),
                        )
                    )

        message = ChatMessage(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
        )

        finish_reason = FinishReason.TOOL_CALLS if tool_calls else FinishReason.STOP

        usage = Usage(
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        )

        return ChatResponse(
            id=response_id,
            model=model,
            choices=[Choice(index=0, message=message, finish_reason=finish_reason)],
            usage=usage,
        )

    async def health_check(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = await self._client.get("/api/tags")
            return response.status_code == 200
        except Exception:
            return False
