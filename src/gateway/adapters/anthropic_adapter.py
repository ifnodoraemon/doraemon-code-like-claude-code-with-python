"""
Anthropic Adapter

Translates unified API to Anthropic Claude format.
"""

import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from src.core.llm.model_utils import normalize_anthropic_base_url

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
from .base import BaseAdapter

logger = logging.getLogger(__name__)


# Anthropic models
ANTHROPIC_MODELS = [
    ModelInfo(
        id="claude-sonnet-4-20250514",
        name="Claude Sonnet 4",
        provider="anthropic",
        description="Best balance of intelligence and speed",
        context_window=200_000,
        max_output=64_000,
        input_price=3.00,
        output_price=15.00,
        capabilities=["text", "vision", "code", "reasoning", "function_calling"],
        aliases=["claude-sonnet-4", "sonnet-4", "sonnet"],
    ),
    ModelInfo(
        id="claude-opus-4-20250514",
        name="Claude Opus 4",
        provider="anthropic",
        description="Most powerful model",
        context_window=200_000,
        max_output=32_000,
        input_price=15.00,
        output_price=75.00,
        capabilities=["text", "vision", "code", "reasoning", "function_calling"],
        aliases=["claude-opus-4", "opus-4", "opus"],
    ),
    ModelInfo(
        id="claude-3-5-sonnet-20241022",
        name="Claude 3.5 Sonnet",
        provider="anthropic",
        description="Previous generation balanced model",
        context_window=200_000,
        max_output=8_192,
        input_price=3.00,
        output_price=15.00,
        capabilities=["text", "vision", "code", "reasoning", "function_calling"],
        aliases=["claude-3.5-sonnet", "claude-35-sonnet"],
    ),
    ModelInfo(
        id="claude-3-5-haiku-20241022",
        name="Claude 3.5 Haiku",
        provider="anthropic",
        description="Fast and affordable",
        context_window=200_000,
        max_output=8_192,
        input_price=0.80,
        output_price=4.00,
        capabilities=["text", "vision", "code", "function_calling"],
        aliases=["claude-3.5-haiku", "claude-35-haiku", "haiku"],
    ),
]


class AnthropicAdapter(BaseAdapter):
    """Adapter for Anthropic Claude models."""

    provider_name = "anthropic"

    async def initialize(self) -> None:
        """Initialize Anthropic client."""
        try:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(
                api_key=self.config.api_key,
                base_url=normalize_anthropic_base_url(self.config.api_base),
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
            logger.info("Anthropic adapter initialized")
        except ImportError as e:
            raise ImportError("anthropic package required: pip install anthropic") from e

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send chat request to Claude."""
        system = None
        messages = []

        for msg in request.messages:
            role = msg.role if isinstance(msg.role, str) else msg.role.value
            if role == "system":
                system = msg.content
            else:
                messages.append(self._convert_message(msg))

        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens or 4096,
        }

        if system:
            kwargs["system"] = system

        if request.temperature != 0.7:
            kwargs["temperature"] = request.temperature

        if request.tools:
            kwargs["tools"] = [self._convert_tool(t) for t in request.tools]

        if request.stop:
            kwargs["stop_sequences"] = request.stop

        if request.top_p is not None:
            kwargs["top_p"] = request.top_p

        response = await self._client.messages.create(**kwargs)
        return self._convert_response(response, request.model)

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """Stream chat response from Claude."""
        system = None
        messages = []

        for msg in request.messages:
            role = msg.role if isinstance(msg.role, str) else msg.role.value
            if role == "system":
                system = msg.content
            else:
                messages.append(self._convert_message(msg))

        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens or 4096,
        }

        if system:
            kwargs["system"] = system

        if request.temperature != 0.7:
            kwargs["temperature"] = request.temperature

        if request.tools:
            kwargs["tools"] = [self._convert_tool(t) for t in request.tools]

        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        input_tokens = 0
        output_tokens = 0

        async with self._client.messages.stream(**kwargs) as stream:
            async for event in stream:
                delta_content = None
                delta_tool_calls = None
                finish_reason = None

                if event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        delta_content = event.delta.text

                elif event.type == "content_block_start":
                    if hasattr(event.content_block, "type"):
                        if event.content_block.type == "tool_use":
                            delta_tool_calls = [
                                ToolCall(
                                    id=event.content_block.id,
                                    name=event.content_block.name,
                                    arguments={},
                                )
                            ]

                elif event.type == "message_delta":
                    if hasattr(event, "usage"):
                        output_tokens = event.usage.output_tokens
                    if event.delta.stop_reason:
                        finish_reason = self._convert_finish_reason(event.delta.stop_reason)

                elif event.type == "message_start":
                    if hasattr(event.message, "usage"):
                        input_tokens = event.message.usage.input_tokens

                usage = None
                if finish_reason:
                    usage = Usage(
                        prompt_tokens=input_tokens,
                        completion_tokens=output_tokens,
                        total_tokens=input_tokens + output_tokens,
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
        """Get available Anthropic models."""
        return ANTHROPIC_MODELS

    def _convert_message(self, msg: ChatMessage) -> dict:
        """Convert unified message to Anthropic format."""
        role = msg.role if isinstance(msg.role, str) else msg.role.value

        if role == "tool":
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content,
                    }
                ],
            }

        content: list[dict[str, Any]] = []

        if msg.content:
            content.append({"type": "text", "text": msg.content})

        if msg.tool_calls:
            for tc in msg.tool_calls:
                content.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }
                )

        return {
            "role": "assistant" if role == "assistant" else "user",
            "content": content if content else msg.content,
        }

    def _convert_tool(self, tool: Any) -> dict:
        """Convert unified tool to Anthropic format."""
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        }

    def _convert_response(self, response: Any, model: str) -> ChatResponse:
        """Convert Anthropic response to unified format."""
        content = None
        tool_calls = None

        for block in response.content:
            if block.type == "text":
                content = (content or "") + block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )
                )

        message = ChatMessage(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
        )

        finish_reason = self._convert_finish_reason(response.stop_reason)

        usage = Usage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

        return ChatResponse(
            id=response.id,
            model=model,
            choices=[Choice(index=0, message=message, finish_reason=finish_reason)],
            usage=usage,
        )

    def _convert_finish_reason(self, reason: str | None) -> str:
        """Convert Anthropic stop reason to unified format."""
        if not reason:
            return FinishReason.STOP.value

        reason = reason.lower()
        if reason == "end_turn":
            return FinishReason.STOP.value
        elif reason == "tool_use":
            return FinishReason.TOOL_CALLS.value
        elif reason == "max_tokens":
            return FinishReason.LENGTH.value
        return FinishReason.STOP.value
