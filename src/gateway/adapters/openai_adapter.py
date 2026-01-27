"""
OpenAI Adapter

Translates unified API to OpenAI format.
Since our unified format is based on OpenAI's, this is mostly pass-through.
"""

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from ..schema import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Choice,
    ModelInfo,
    Role,
    StreamChunk,
    ToolCall,
    Usage,
)
from .base import BaseAdapter

logger = logging.getLogger(__name__)


# OpenAI models
OPENAI_MODELS = [
    ModelInfo(
        id="gpt-4o",
        name="GPT-4o",
        provider="openai",
        description="OpenAI's flagship multimodal model",
        context_window=128_000,
        max_output=16_384,
        input_price=2.50,
        output_price=10.00,
        capabilities=["text", "vision", "code", "reasoning", "function_calling"],
        aliases=["gpt4o", "4o"],
    ),
    ModelInfo(
        id="gpt-4o-mini",
        name="GPT-4o Mini",
        provider="openai",
        description="Small, fast, and affordable",
        context_window=128_000,
        max_output=16_384,
        input_price=0.15,
        output_price=0.60,
        capabilities=["text", "vision", "code", "function_calling"],
        aliases=["gpt4o-mini", "4o-mini", "mini"],
    ),
    ModelInfo(
        id="o1",
        name="o1",
        provider="openai",
        description="OpenAI's reasoning model",
        context_window=200_000,
        max_output=100_000,
        input_price=15.00,
        output_price=60.00,
        capabilities=["text", "code", "reasoning"],
        aliases=["o1-preview"],
    ),
    ModelInfo(
        id="o3-mini",
        name="o3-mini",
        provider="openai",
        description="Fast reasoning model",
        context_window=200_000,
        max_output=100_000,
        input_price=1.10,
        output_price=4.40,
        capabilities=["text", "code", "reasoning"],
        aliases=["o3mini"],
    ),
]


class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI models."""

    provider_name = "openai"

    async def initialize(self) -> None:
        """Initialize OpenAI client."""
        try:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
            logger.info("OpenAI adapter initialized")
        except ImportError as e:
            raise ImportError("openai package required: pip install openai") from e

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send chat request to OpenAI."""
        # Convert to OpenAI format (mostly pass-through)
        messages = [self._convert_message(m) for m in request.messages]

        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
        }

        if request.max_tokens:
            kwargs["max_tokens"] = request.max_tokens

        if request.tools:
            kwargs["tools"] = [t.to_dict() for t in request.tools]

        if request.stop:
            kwargs["stop"] = request.stop

        if request.top_p is not None:
            kwargs["top_p"] = request.top_p

        if request.presence_penalty is not None:
            kwargs["presence_penalty"] = request.presence_penalty

        if request.frequency_penalty is not None:
            kwargs["frequency_penalty"] = request.frequency_penalty

        response = await self._client.chat.completions.create(**kwargs)
        return self._convert_response(response)

    async def chat_stream(
        self, request: ChatRequest
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat response from OpenAI."""
        messages = [self._convert_message(m) for m in request.messages]

        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if request.max_tokens:
            kwargs["max_tokens"] = request.max_tokens

        if request.tools:
            kwargs["tools"] = [t.to_dict() for t in request.tools]

        if request.stop:
            kwargs["stop"] = request.stop

        response = await self._client.chat.completions.create(**kwargs)

        async for chunk in response:
            delta_content = None
            delta_tool_calls = None
            finish_reason = None
            usage = None

            if chunk.choices:
                choice = chunk.choices[0]
                delta = choice.delta

                if delta.content:
                    delta_content = delta.content

                if delta.tool_calls:
                    delta_tool_calls = []
                    for tc in delta.tool_calls:
                        delta_tool_calls.append(
                            ToolCall(
                                id=tc.id or "",
                                name=tc.function.name if tc.function else "",
                                arguments=json.loads(tc.function.arguments)
                                if tc.function and tc.function.arguments
                                else {},
                            )
                        )

                if choice.finish_reason:
                    finish_reason = choice.finish_reason

            if chunk.usage:
                usage = Usage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                )

            yield StreamChunk(
                id=chunk.id,
                model=chunk.model,
                delta_content=delta_content,
                delta_tool_calls=delta_tool_calls,
                finish_reason=finish_reason,
                usage=usage,
            )

    def get_models(self) -> list[ModelInfo]:
        """Get available OpenAI models."""
        return OPENAI_MODELS

    def _convert_message(self, msg: ChatMessage) -> dict:
        """Convert unified message to OpenAI format."""
        result = {"role": msg.role if isinstance(msg.role, str) else msg.role.value}

        if msg.content is not None:
            result["content"] = msg.content

        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in msg.tool_calls
            ]

        if msg.tool_call_id:
            result["tool_call_id"] = msg.tool_call_id

        if msg.name:
            result["name"] = msg.name

        return result

    def _convert_response(self, response: Any) -> ChatResponse:
        """Convert OpenAI response to unified format."""
        choices = []

        for choice in response.choices:
            tool_calls = None
            if choice.message.tool_calls:
                tool_calls = []
                for tc in choice.message.tool_calls:
                    tool_calls.append(
                        ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=json.loads(tc.function.arguments)
                            if tc.function.arguments
                            else {},
                        )
                    )

            message = ChatMessage(
                role=Role.ASSISTANT,
                content=choice.message.content,
                tool_calls=tool_calls,
            )

            choices.append(
                Choice(
                    index=choice.index,
                    message=message,
                    finish_reason=choice.finish_reason,
                )
            )

        usage = None
        if response.usage:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return ChatResponse(
            id=response.id,
            model=response.model,
            choices=choices,
            usage=usage,
            created=response.created,
        )
