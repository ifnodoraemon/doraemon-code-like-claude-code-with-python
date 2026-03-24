"""
Direct Model Client

Client that connects directly to provider APIs (Google, OpenAI, Anthropic, Ollama).
"""

import asyncio
import json
import logging
import random
from collections.abc import AsyncIterator, Sequence
from typing import Any

from src.core.model_client_base import BaseModelClient
from src.core.model_utils import (
    ChatResponse,
    ClientConfig,
    Message,
    Provider,
    StreamChunk,
    ToolDefinition,
    get_content_text,
)
from src.core.provider_adapters import (
    AnthropicAdapter,
    GoogleAdapter,
    OpenAIAdapter,
    build_anthropic_content_parts,
    build_google_content_parts,
    build_openai_content_parts,
)

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
INITIAL_DELAY = 1.0
MAX_DELAY = 60.0


# Re-export for backward compatibility
_build_google_content_parts = build_google_content_parts
_build_openai_content_parts = build_openai_content_parts
_build_anthropic_content_parts = build_anthropic_content_parts


def _is_retryable(exc: Exception) -> bool:
    """Check if an exception is retryable (rate limit or server error)."""
    msg = str(exc).lower()
    # Rate limit errors
    if "rate" in msg and "limit" in msg:
        return True
    if "429" in msg or "too many requests" in msg:
        return True
    # Server errors (5xx)
    if "500" in msg or "502" in msg or "503" in msg or "504" in msg:
        return True
    if "server error" in msg or "internal error" in msg:
        return True
    # Transient network errors
    if "timeout" in msg or "timed out" in msg:
        return True
    if "connection" in msg and ("reset" in msg or "refused" in msg or "error" in msg):
        return True
    # Resource exhausted (Google)
    if "resource" in msg and "exhausted" in msg:
        return True
    return False


async def _retry_async(coro_fn, *args, **kwargs):
    """Execute an async function with retry and exponential backoff."""
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            return await coro_fn(*args, **kwargs)
        except Exception as e:
            if not _is_retryable(e) or attempt >= MAX_RETRIES - 1:
                raise
            last_exc = e
            delay = min(INITIAL_DELAY * (2**attempt), MAX_DELAY)
            delay *= 0.5 + random.random()  # jitter
            logger.warning(
                f"Retryable error (attempt {attempt + 1}/{MAX_RETRIES}): {e}. "
                f"Retrying in {delay:.1f}s..."
            )
            await asyncio.sleep(delay)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("_retry_async failed: no attempts were made")


class DirectModelClient(BaseModelClient):
    """Client that connects directly to provider APIs."""

    # Provider detection patterns
    PROVIDER_PATTERNS = {
        Provider.GOOGLE: ["gemini-", "palm-"],
        Provider.OPENAI: ["gpt-", "o1", "o3"],
        Provider.ANTHROPIC: ["claude-"],
        Provider.OLLAMA: [],
    }

    def __init__(self, config: ClientConfig):
        self.config = config
        self._providers: dict[Provider, Any] = {}

    async def __aenter__(self):
        """Context manager entry - ensure providers are connected."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure all providers are closed."""
        await self.close()
        return False

    async def connect(self) -> None:
        """Initialize provider clients."""
        # Google Gemini
        if self.config.google_api_key:
            try:
                from google import genai

                self._providers[Provider.GOOGLE] = genai.Client(api_key=self.config.google_api_key)
                logger.info("Google Gemini client initialized")
            except ImportError:
                logger.warning("google-genai not installed")

        # OpenAI
        if self.config.openai_api_key:
            try:
                from openai import AsyncOpenAI

                self._providers[Provider.OPENAI] = AsyncOpenAI(api_key=self.config.openai_api_key)
                logger.info("OpenAI client initialized")
            except ImportError:
                logger.warning("openai not installed")

        # Anthropic
        if self.config.anthropic_api_key:
            try:
                from anthropic import AsyncAnthropic

                self._providers[Provider.ANTHROPIC] = AsyncAnthropic(
                    api_key=self.config.anthropic_api_key
                )
                logger.info("Anthropic client initialized")
            except ImportError:
                logger.warning("anthropic not installed")

        # Ollama (always available if running)
        try:
            import httpx

            self._providers[Provider.OLLAMA] = httpx.AsyncClient(
                base_url=self.config.ollama_base_url,
                timeout=120.0,
            )
            logger.info("Ollama client initialized")
        except Exception:
            pass

        if not self._providers:
            raise RuntimeError("No providers available. Check your API keys.")

    def _detect_provider(self, model: str) -> Provider:
        """Detect provider from model name."""
        if not self._providers:
            raise RuntimeError("No providers available. Check your API keys.")

        for provider, patterns in self.PROVIDER_PATTERNS.items():
            for pattern in patterns:
                if model.startswith(pattern):
                    if provider in self._providers:
                        return provider
                    logger.warning(
                        "Model '%s' matched provider '%s' but it is not configured; "
                        "falling back to the first available provider.",
                        model,
                        provider.value,
                    )
                    return next(iter(self._providers.keys()))

        # Default to first available
        return next(iter(self._providers.keys()))

    async def chat(
        self,
        messages: Sequence[Message | dict],
        tools: Sequence[ToolDefinition | dict] | None = None,
        **kwargs,
    ) -> ChatResponse:
        if not self._providers:
            await self.connect()

        model = kwargs.get("model", self.config.model)
        provider = self._detect_provider(model)

        if provider == Provider.GOOGLE:
            return await _retry_async(self._chat_google, messages, tools, **kwargs)
        elif provider == Provider.OPENAI:
            return await _retry_async(self._chat_openai, messages, tools, **kwargs)
        elif provider == Provider.ANTHROPIC:
            return await _retry_async(self._chat_anthropic, messages, tools, **kwargs)
        else:
            return await _retry_async(self._chat_ollama, messages, tools, **kwargs)

    async def _chat_google(
        self,
        messages: Sequence[Message | dict],
        tools: Sequence[ToolDefinition | dict] | None = None,
        **kwargs,
    ) -> ChatResponse:
        """Chat with Google Gemini."""
        from google.genai import types

        client = self._providers[Provider.GOOGLE]
        model = kwargs.get("model", self.config.model)
        temperature = kwargs.get("temperature", self.config.temperature)

        # Convert messages and build config via adapter
        system_instruction, contents = GoogleAdapter.convert_messages(
            messages, self.config.system_prompt, types
        )
        gen_config = GoogleAdapter.build_config(
            tools, system_instruction, temperature, self.config.max_tokens, types
        )

        response = await client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=gen_config,
        )

        # Parse response via adapter
        if not response.candidates:
            logger.warning("No candidates in Gemini response")
            return ChatResponse(
                content=None,
                tool_calls=None,
                finish_reason="error",
                usage=None,
                raw=response,
            )

        candidate = response.candidates[0]
        content, thought, tool_calls = GoogleAdapter.parse_candidate(candidate)
        finish_reason = "tool_calls" if tool_calls else "stop"
        usage = GoogleAdapter.parse_usage(response)

        return ChatResponse(
            content=content,
            thought=thought,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            raw=response,
        )

    async def _chat_openai(
        self,
        messages: Sequence[Message | dict],
        tools: Sequence[ToolDefinition | dict] | None = None,
        **kwargs,
    ) -> ChatResponse:
        """Chat with OpenAI."""
        client = self._providers[Provider.OPENAI]
        model = kwargs.get("model", self.config.model)
        temperature = kwargs.get("temperature", self.config.temperature)

        msg_list = OpenAIAdapter.convert_messages(messages)
        params = OpenAIAdapter.build_params(
            model, msg_list, tools, temperature, self.config.max_tokens
        )

        response = await client.chat.completions.create(**params)

        if not response.choices:
            logger.warning("No choices in OpenAI response")
            return ChatResponse(
                content=None,
                tool_calls=None,
                finish_reason="error",
                usage=None,
                raw=response,
            )
        choice = response.choices[0]

        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in choice.message.tool_calls
            ]

        return ChatResponse(
            content=choice.message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            if response.usage
            else None,
            raw=response,
        )

    async def _chat_anthropic(
        self,
        messages: Sequence[Message | dict],
        tools: Sequence[ToolDefinition | dict] | None = None,
        **kwargs,
    ) -> ChatResponse:
        """Chat with Anthropic Claude."""
        client = self._providers[Provider.ANTHROPIC]
        model = kwargs.get("model", self.config.model)

        system, msg_list = AnthropicAdapter.convert_messages(messages, self.config.system_prompt)
        params = AnthropicAdapter.build_params(
            model, msg_list, tools, system, self.config.max_tokens
        )

        response = await client.messages.create(**params)

        content = None
        tool_calls = None

        for block in response.content:
            if block.type == "text":
                content = (content or "") + block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input),
                        },
                    }
                )

        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            raw=response,
        )

    async def _chat_ollama(
        self,
        messages: Sequence[Message | dict],
        tools: Sequence[ToolDefinition | dict] | None = None,
        **kwargs,
    ) -> ChatResponse:
        """Chat with Ollama."""
        client = self._providers[Provider.OLLAMA]
        model = kwargs.get("model", self.config.model)

        msg_list = []
        for m in messages:
            msg = m if isinstance(m, dict) else m.to_dict()
            # Ollama doesn't support multimodal - extract text only
            content = get_content_text(msg.get("content", ""))
            msg_list.append(
                {
                    "role": msg.get("role", "user"),
                    "content": content,
                }
            )

        payload = {
            "model": model,
            "messages": msg_list,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
            },
        }

        response = await client.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

        return ChatResponse(
            content=data.get("message", {}).get("content"),
            tool_calls=None,
            finish_reason="stop",
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            },
            raw=data,
        )

    async def chat_stream(
        self,
        messages: Sequence[Message | dict],
        tools: Sequence[ToolDefinition | dict] | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Streaming chat with provider-specific streaming APIs."""
        if not self._providers:
            await self.connect()

        model = kwargs.get("model", self.config.model)
        provider = self._detect_provider(model)

        if provider == Provider.GOOGLE:
            async for chunk in self._stream_google(messages, tools, **kwargs):
                yield chunk
        elif provider == Provider.OPENAI:
            async for chunk in self._stream_openai(messages, tools, **kwargs):
                yield chunk
        elif provider == Provider.ANTHROPIC:
            async for chunk in self._stream_anthropic(messages, tools, **kwargs):
                yield chunk
        else:
            # Ollama: fallback to non-streaming
            response = await self.chat(messages, tools, **kwargs)
            yield StreamChunk(
                content=response.content,
                tool_calls=response.tool_calls,
                finish_reason=response.finish_reason,
                usage=response.usage,
            )

    async def _stream_google(
        self,
        messages: Sequence[Message | dict],
        tools: Sequence[ToolDefinition | dict] | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Stream with Google Gemini using generate_content_stream."""
        from google.genai import types

        client = self._providers[Provider.GOOGLE]
        model = kwargs.get("model", self.config.model)
        temperature = kwargs.get("temperature", self.config.temperature)

        # Reuse adapter for message conversion and config building
        system_instruction, contents = GoogleAdapter.convert_messages(
            messages, self.config.system_prompt, types
        )
        gen_config = GoogleAdapter.build_config(
            tools, system_instruction, temperature, self.config.max_tokens, types
        )

        async for response in client.aio.models.generate_content_stream(
            model=model,
            contents=contents,
            config=gen_config,
        ):
            if not response.candidates:
                continue

            candidate = response.candidates[0]
            text, thought, tool_calls = GoogleAdapter.parse_candidate(candidate)
            usage = GoogleAdapter.parse_usage(response)

            finish_reason = None
            if hasattr(candidate, "finish_reason") and candidate.finish_reason:
                finish_reason = "tool_calls" if tool_calls else "stop"

            yield StreamChunk(
                content=text,
                thought=thought,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                usage=usage,
            )

    async def _stream_openai(
        self,
        messages: Sequence[Message | dict],
        tools: Sequence[ToolDefinition | dict] | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Stream with OpenAI API."""
        client = self._providers[Provider.OPENAI]
        model = kwargs.get("model", self.config.model)
        temperature = kwargs.get("temperature", self.config.temperature)

        msg_list = OpenAIAdapter.convert_messages(messages)
        params = OpenAIAdapter.build_params(
            model, msg_list, tools, temperature, self.config.max_tokens, stream=True
        )

        stream = await client.chat.completions.create(**params)

        async for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            content = delta.content if hasattr(delta, "content") else None
            tool_calls = None

            if hasattr(delta, "tool_calls") and delta.tool_calls:
                tool_calls = []
                for tc in delta.tool_calls:
                    tc_dict = {"index": tc.index}
                    if tc.id:
                        tc_dict["id"] = tc.id
                    if tc.function:
                        tc_dict["function"] = {}
                        if tc.function.name:
                            tc_dict["function"]["name"] = tc.function.name
                        if tc.function.arguments:
                            tc_dict["function"]["arguments"] = tc.function.arguments
                    tool_calls.append(tc_dict)

            finish_reason = chunk.choices[0].finish_reason

            usage = None
            if hasattr(chunk, "usage") and chunk.usage:
                usage = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                }

            yield StreamChunk(
                content=content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                usage=usage,
            )

    async def _stream_anthropic(
        self,
        messages: Sequence[Message | dict],
        tools: Sequence[ToolDefinition | dict] | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Stream with Anthropic API."""
        client = self._providers[Provider.ANTHROPIC]
        model = kwargs.get("model", self.config.model)

        system, msg_list = AnthropicAdapter.convert_messages(messages, self.config.system_prompt)
        params = AnthropicAdapter.build_params(
            model, msg_list, tools, system, self.config.max_tokens
        )

        async with client.messages.stream(**params) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        yield StreamChunk(content=event.delta.text)
                    elif hasattr(event.delta, "partial_json"):
                        yield StreamChunk(
                            tool_calls=[
                                {
                                    "index": event.index,
                                    "function": {"arguments": event.delta.partial_json},
                                }
                            ]
                        )
                elif event.type == "content_block_start":
                    if (
                        hasattr(event.content_block, "type")
                        and event.content_block.type == "tool_use"
                    ):
                        yield StreamChunk(
                            tool_calls=[
                                {
                                    "index": event.index,
                                    "id": event.content_block.id,
                                    "type": "function",
                                    "function": {
                                        "name": event.content_block.name,
                                        "arguments": "",
                                    },
                                }
                            ]
                        )
                elif event.type == "message_delta":
                    usage = None
                    if hasattr(event, "usage") and event.usage:
                        usage = {
                            "prompt_tokens": getattr(event.usage, "input_tokens", 0) or 0,
                            "completion_tokens": getattr(event.usage, "output_tokens", 0) or 0,
                            "total_tokens": (getattr(event.usage, "input_tokens", 0) or 0)
                            + (getattr(event.usage, "output_tokens", 0) or 0),
                        }
                    yield StreamChunk(
                        finish_reason=getattr(event.delta, "stop_reason", None),
                        usage=usage,
                    )

    async def list_models(self) -> list[dict]:
        """List available models from all providers."""
        models = []

        if Provider.GOOGLE in self._providers:
            models.extend(
                [
                    {"id": "gemini-2.5-flash-preview", "provider": "google"},
                    {"id": "gemini-2.5-pro-preview", "provider": "google"},
                ]
            )

        if Provider.OPENAI in self._providers:
            models.extend(
                [
                    {"id": "gpt-4o", "provider": "openai"},
                    {"id": "gpt-4o-mini", "provider": "openai"},
                ]
            )

        if Provider.ANTHROPIC in self._providers:
            models.extend(
                [
                    {"id": "claude-sonnet-4-20250514", "provider": "anthropic"},
                    {"id": "claude-3-5-haiku-20241022", "provider": "anthropic"},
                ]
            )

        return models

    async def close(self) -> None:
        """Close all provider clients."""
        for _provider, client in self._providers.items():
            if hasattr(client, "aclose"):
                await client.aclose()
        self._providers.clear()
