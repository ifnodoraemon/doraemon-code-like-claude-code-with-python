"""
Direct Model Client

Client that connects directly to provider APIs (Google, OpenAI, Anthropic).
"""

import asyncio
import json
import logging
import random
from collections.abc import AsyncIterator, Sequence
from typing import Any

from src.core.errors import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerOpenError
from src.core.llm.model_client_base import BaseModelClient
from src.core.llm.model_utils import (
    ChatResponse,
    ClientConfig,
    Message,
    Provider,
    StreamChunk,
    ToolDefinition,
    normalize_anthropic_base_url,
)
from src.core.llm.providers import (
    AnthropicAdapter,
    GoogleAdapter,
    OpenAIAdapter,
    _serialize_gemini_thought_signature,
    build_anthropic_content_parts,
    build_google_content_parts,
    build_openai_content_parts,
)

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
INITIAL_DELAY = 1.0
MAX_DELAY = 60.0

# CircuitBreaker defaults
BREAKER_FAILURE_THRESHOLD = 5
BREAKER_TIMEOUT = 60.0
_OPENAI_PROTOCOL_CACHE: dict[tuple[str, str], str] = {}


# Re-export for backward compatibility
_build_google_content_parts = build_google_content_parts
_build_openai_content_parts = build_openai_content_parts
_build_anthropic_content_parts = build_anthropic_content_parts


def _is_google_openai_compatible_base(base_url: str | None) -> bool:
    if not base_url:
        return False
    normalized = base_url.rstrip("/").lower()
    return "generativelanguage.googleapis.com" in normalized and normalized.endswith("/openai")


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
    }

    def __init__(self, config: ClientConfig):
        self.config = config
        self.model = config.model
        self._providers: dict[Provider, Any] = {}
        self._circuit_breakers: dict[Provider, CircuitBreaker] = {}
        self._openai_protocol: str | None = (
            config.openai_protocol
            if config.openai_protocol != "auto"
            else _OPENAI_PROTOCOL_CACHE.get(self._get_openai_protocol_cache_key())
        )

        breaker_config = CircuitBreakerConfig(
            failure_threshold=BREAKER_FAILURE_THRESHOLD,
            timeout=BREAKER_TIMEOUT,
        )
        for provider in Provider:
            self._circuit_breakers[provider] = CircuitBreaker(breaker_config)

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
        effective_google_api_key = self.config.google_api_key
        if not effective_google_api_key and _is_google_openai_compatible_base(
            self.config.openai_api_base
        ):
            effective_google_api_key = self.config.openai_api_key

        if effective_google_api_key:
            try:
                from google import genai

                self._providers[Provider.GOOGLE] = genai.Client(api_key=effective_google_api_key)
                logger.info("Google Gemini client initialized")
            except ImportError:
                logger.warning("google-genai not installed")

        # OpenAI
        if self.config.openai_api_key:
            try:
                from openai import AsyncOpenAI

                self._providers[Provider.OPENAI] = AsyncOpenAI(
                    api_key=self.config.openai_api_key,
                    base_url=self.config.openai_api_base,
                )
                logger.info("OpenAI client initialized")
            except ImportError:
                logger.warning("openai not installed")

        # Anthropic
        if self.config.anthropic_api_key:
            try:
                from anthropic import AsyncAnthropic

                self._providers[Provider.ANTHROPIC] = AsyncAnthropic(
                    api_key=self.config.anthropic_api_key,
                    base_url=normalize_anthropic_base_url(self.config.anthropic_api_base),
                )
                logger.info("Anthropic client initialized")
            except ImportError:
                logger.warning("anthropic not installed")

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
        breaker = self._circuit_breakers.get(provider)

        async def _chat_with_provider():
            if provider == Provider.GOOGLE:
                return await self._chat_google(messages, tools, **kwargs)
            elif provider == Provider.OPENAI:
                return await self._chat_openai(messages, tools, **kwargs)
            else:
                return await self._chat_anthropic(messages, tools, **kwargs)

        try:
            if breaker:
                return await breaker.call_async(_chat_with_provider)
            return await _chat_with_provider()
        except CircuitBreakerOpenError as e:
            logger.error("Circuit breaker open for %s: %s", provider.value, e)
            raise RuntimeError(
                f"Provider {provider.value} is temporarily unavailable. "
                f"Please try again in a moment."
            ) from e

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
            messages, self.config.system_prompt, types_module=types
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
        """Chat with OpenAI.

        Prefer the Responses API for newer OpenAI-compatible models and fall back
        to Chat Completions when the upstream does not support `/responses`.
        """
        model = kwargs.get("model", self.config.model)
        temperature = kwargs.get("temperature", self.config.temperature)

        msg_list = OpenAIAdapter.convert_messages(messages)
        client = self._providers[Provider.OPENAI]
        tools = self._prepare_tools_for_provider(Provider.OPENAI, tools)
        response_tools = self._build_openai_responses_tools(tools)
        protocol_mode = self.config.openai_protocol
        allow_fallback = protocol_mode == "auto"
        use_responses = False
        if protocol_mode == "responses":
            use_responses = True
        elif protocol_mode == "chat_completions":
            use_responses = False
        else:
            use_responses = self._openai_protocol == "responses" or (
                self._openai_protocol is None and self._should_use_openai_responses_api(msg_list)
            )

        if use_responses:
            try:
                params = self._build_openai_responses_params(
                    model=model,
                    msg_list=msg_list,
                    tools=response_tools,
                    temperature=temperature,
                    max_output_tokens=self.config.max_tokens,
                    include=self.config.openai_responses_include,
                    stream=False,
                )
                logger.info(
                    "OpenAI Responses request summary: %s",
                    self._summarize_openai_responses_request_params(params),
                )
                response = await client.responses.create(**params)
                if isinstance(response, str):
                    raise RuntimeError(
                        "Provider returned non-OpenAI responses payload (string body)"
                    )
                logger.info(
                    "OpenAI Responses response summary: %s",
                    self._summarize_openai_responses_payload(response),
                )
                response_error = getattr(response, "error", None)
                response_status = getattr(response, "status", None)
                response_code = getattr(response, "code", None)
                response_message = getattr(response, "message", None)
                response_output = getattr(response, "output", None)
                if (
                    response_error
                    or response_status == "failed"
                    or response_code
                    or response_message
                    or (response_status is None and response_output is None)
                ):
                    error_msg = getattr(response_error, "message", None) if response_error else None
                    raise RuntimeError(
                        "OpenAI Responses API failed: "
                        f"{error_msg or response_message or response_code or response_status or 'empty response'}"
                    )
                self._openai_protocol = "responses"
                if self.config.openai_protocol == "auto":
                    _OPENAI_PROTOCOL_CACHE[self._get_openai_protocol_cache_key()] = "responses"
                parsed = self._parse_openai_responses_response(response)
                if not parsed.content and not parsed.tool_calls:
                    logger.warning(
                        "OpenAI Responses parsed an empty result: %s",
                        self._summarize_openai_responses_payload(response, include_items=True),
                    )
                return parsed
            except Exception as e:
                if not allow_fallback or not self._should_fallback_to_chat_completions(e):
                    raise
                logger.warning(
                    "OpenAI Responses API unavailable, falling back to chat.completions: %s", e
                )

        params = OpenAIAdapter.build_params(
            model, msg_list, tools, temperature, self.config.max_tokens
        )
        response = await client.chat.completions.create(**params)
        self._openai_protocol = "chat_completions"
        if self.config.openai_protocol == "auto":
            _OPENAI_PROTOCOL_CACHE[self._get_openai_protocol_cache_key()] = "chat_completions"

        if isinstance(response, str):
            raise RuntimeError("Provider returned non-OpenAI chat response (string body)")

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
            tool_calls = []
            for tc in choice.message.tool_calls:
                tc_dict = {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                thought_signature = getattr(tc, "thought_signature", None) or getattr(
                    tc.function, "thought_signature", None
                )
                if thought_signature is not None:
                    serialized_signature = _serialize_gemini_thought_signature(thought_signature)
                    tc_dict["thought_signature"] = serialized_signature
                    tc_dict["function"]["thought_signature"] = serialized_signature
                tool_calls.append(tc_dict)

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

    @staticmethod
    def _build_openai_responses_tools(
        tools: Sequence[ToolDefinition | dict] | None,
    ) -> list[dict] | None:
        """Convert chat-completions style tools into Responses API tools."""
        if not tools:
            return None

        response_tools: list[dict] = []
        for tool in tools:
            formatted = tool.to_openai_format() if isinstance(tool, ToolDefinition) else tool
            if formatted.get("type") != "function":
                response_tools.append(formatted)
                continue

            function = formatted.get("function", {})
            response_tools.append(
                {
                    "type": "function",
                    "name": function.get("name"),
                    "description": function.get("description", ""),
                    "parameters": function.get("parameters", {}),
                    "strict": False,
                }
            )
        return response_tools

    @staticmethod
    def _build_openai_responses_params(
        *,
        model: str,
        msg_list: list[dict],
        tools: list[dict] | None,
        temperature: float,
        max_output_tokens: int | None,
        include: Sequence[str] | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Build request params for the OpenAI Responses API."""
        input_items: list[dict[str, Any]] = []
        for msg in msg_list:
            role = msg.get("role", "user")
            content = msg.get("content")
            tool_calls = msg.get("tool_calls") or []
            provider_items = msg.get("provider_items") or []

            def _build_content_parts(
                content_value: Any, *, assistant: bool = False
            ) -> list[dict[str, Any]]:
                text_type = "output_text" if assistant else "input_text"
                if isinstance(content_value, str):
                    return [{"type": text_type, "text": content_value}]
                if isinstance(content_value, list):
                    content_parts: list[dict[str, Any]] = []
                    for part in content_value:
                        part_type = part.get("type")
                        if part_type == "text":
                            content_parts.append({"type": text_type, "text": part.get("text", "")})
                        elif part_type == "image_url":
                            content_parts.append(
                                {
                                    "type": "input_image",
                                    "image_url": part.get("image_url", {}).get("url", ""),
                                }
                            )
                    return content_parts
                return [{"type": text_type, "text": str(content_value or "")}]

            if role == "assistant":
                if provider_items:
                    input_items.extend(provider_items)
                    continue
                if content:
                    input_items.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": _build_content_parts(content, assistant=True),
                        }
                    )
                for tool_call in tool_calls:
                    function = tool_call.get("function", {})
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": tool_call.get("id") or "",
                            "name": function.get("name", ""),
                            "arguments": function.get("arguments", "{}"),
                        }
                    )
                continue

            if role == "tool":
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.get("tool_call_id") or "",
                        "output": str(content or ""),
                    }
                )
                continue

            input_items.append(
                {
                    "role": role,
                    "content": _build_content_parts(content),
                }
            )

        params: dict[str, Any] = {
            "model": model,
            "input": input_items,
            "store": False,
        }
        if tools:
            params["tools"] = tools
        if max_output_tokens:
            params["max_output_tokens"] = max_output_tokens
        if include:
            params["include"] = list(include)
        if stream:
            params["stream"] = True
        return params

    @staticmethod
    def _should_fallback_to_chat_completions(exc: Exception) -> bool:
        """Detect when an upstream likely does not support `/responses`."""
        message = str(exc).lower()
        if "non-openai responses payload" in message:
            return False
        fallback_markers = [
            "/responses",
            "responses api",
            "responses endpoint",
            "not found",
            "404",
            "bad_response_status_code",
        ]
        return any(marker in message for marker in fallback_markers)

    def _should_use_openai_responses_api(self, msg_list: Sequence[dict[str, Any]]) -> bool:
        """Use Responses API only for official OpenAI-style first turns we can encode correctly."""
        base_url = (self.config.openai_api_base or "").lower().rstrip("/")
        allowed_domains = {"api.openai.com", "openai.azure.com"}
        if base_url and not any(domain in base_url for domain in allowed_domains):
            return False

        return True

    def _get_openai_protocol_cache_key(self) -> tuple[str, str]:
        return ((self.config.openai_api_base or "").rstrip("/").lower(), self.model or "")

    @staticmethod
    def _parse_openai_responses_response(response: Any) -> ChatResponse:
        """Parse an OpenAI Responses API object into the unified ChatResponse."""
        if isinstance(response, str):
            raise RuntimeError("Provider returned non-OpenAI responses payload (string body)")

        try:
            output_text = getattr(response, "output_text", None)
        except TypeError:
            output_text = None
        content = output_text or None
        tool_calls: list[dict] | None = None
        provider_items: list[dict[str, Any]] = []

        for item in getattr(response, "output", None) or []:
            item_type = getattr(item, "type", None)
            serialized_item = DirectModelClient._serialize_openai_response_output_item(item)
            if serialized_item is not None:
                provider_items.append(serialized_item)
            if item_type == "message" and not content:
                for part in getattr(item, "content", []) or []:
                    if getattr(part, "type", None) in {"output_text", "text"}:
                        text = getattr(part, "text", None)
                        if text:
                            content = (content or "") + text
            elif item_type in {"function_call", "tool_call"}:
                if tool_calls is None:
                    tool_calls = []
                arguments = getattr(item, "arguments", None)
                tool_calls.append(
                    {
                        "id": getattr(item, "call_id", None) or getattr(item, "id", ""),
                        "type": "function",
                        "function": {
                            "name": getattr(item, "name", ""),
                            "arguments": arguments
                            if isinstance(arguments, str)
                            else json.dumps(arguments or {}),
                        },
                    }
                )

        usage = None
        if getattr(response, "usage", None):
            usage = {
                "prompt_tokens": getattr(response.usage, "input_tokens", 0),
                "completion_tokens": getattr(response.usage, "output_tokens", 0),
                "total_tokens": (
                    getattr(response.usage, "input_tokens", 0)
                    + getattr(response.usage, "output_tokens", 0)
                ),
            }

        return ChatResponse(
            content=content,
            provider_items=provider_items or None,
            tool_calls=tool_calls,
            finish_reason=getattr(response, "status", None),
            usage=usage,
            raw=response,
        )

    @staticmethod
    def _summarize_openai_responses_request_params(params: dict[str, Any]) -> dict[str, Any]:
        """Build a safe request summary for Responses API diagnostics."""
        input_items = params.get("input") or []
        return {
            "model": params.get("model"),
            "stream": params.get("stream", False),
            "store": params.get("store"),
            "include": params.get("include", []),
            "tool_count": len(params.get("tools") or []),
            "tool_names": [tool.get("name") for tool in (params.get("tools") or [])],
            "input_count": len(input_items),
            "input_types": [item.get("type", "message") for item in input_items[:8]],
            "input_roles": [item.get("role") for item in input_items[:8] if item.get("role")],
        }

    @classmethod
    def _summarize_openai_responses_payload(
        cls,
        response: Any,
        *,
        include_items: bool = False,
    ) -> dict[str, Any]:
        """Build a safe summary for Responses API payloads."""
        output_items = []
        for item in getattr(response, "output", None) or []:
            serialized = cls._serialize_openai_response_output_item(item)
            if serialized is not None:
                output_items.append(serialized)

        summary: dict[str, Any] = {
            "id": getattr(response, "id", None),
            "status": getattr(response, "status", None),
            "output_text_present": bool(getattr(response, "output_text", None)),
            "output_count": len(output_items),
            "output_types": [item.get("type") for item in output_items],
            "usage": (
                {
                    "input_tokens": getattr(getattr(response, "usage", None), "input_tokens", None),
                    "output_tokens": getattr(
                        getattr(response, "usage", None), "output_tokens", None
                    ),
                }
                if getattr(response, "usage", None)
                else None
            ),
        }
        if include_items:
            summary["output_items"] = output_items
        return summary

    @staticmethod
    def _serialize_openai_response_output_item(item: Any) -> dict[str, Any] | None:
        """Convert an OpenAI Responses output item into a JSON-safe dict."""
        if hasattr(item, "model_dump"):
            return item.model_dump(exclude_none=True)
        if isinstance(item, dict):
            return item

        item_type = getattr(item, "type", None)
        if item_type == "function_call":
            arguments = getattr(item, "arguments", None)
            return {
                "type": "function_call",
                "call_id": getattr(item, "call_id", None) or getattr(item, "id", ""),
                "name": getattr(item, "name", ""),
                "arguments": arguments
                if isinstance(arguments, str)
                else json.dumps(arguments or {}),
            }
        if item_type == "reasoning":
            serialized = {"type": "reasoning"}
            if summary := getattr(item, "summary", None):
                serialized["summary"] = summary
            if encrypted := getattr(item, "encrypted_content", None):
                serialized["encrypted_content"] = encrypted
            if item_id := getattr(item, "id", None):
                serialized["id"] = item_id
            return serialized
        if item_type == "message":
            content_parts = []
            for part in getattr(item, "content", []) or []:
                part_type = getattr(part, "type", None)
                if part_type in {"output_text", "text"}:
                    content_parts.append({"type": part_type, "text": getattr(part, "text", "")})
            return {
                "type": "message",
                "role": getattr(item, "role", "assistant"),
                "content": content_parts,
            }
        return None

    async def _chat_anthropic(
        self,
        messages: Sequence[Message | dict],
        tools: Sequence[ToolDefinition | dict] | None = None,
        **kwargs,
    ) -> ChatResponse:
        """Chat with Anthropic Claude."""
        protocol_mode = self.config.anthropic_protocol
        if protocol_mode not in {"auto", "messages"}:
            raise RuntimeError(f"Unsupported Anthropic protocol: {protocol_mode}")

        client = self._providers[Provider.ANTHROPIC]
        model = kwargs.get("model", self.config.model)
        tools = self._prepare_tools_for_provider(Provider.ANTHROPIC, tools)

        system, msg_list = AnthropicAdapter.convert_messages(messages, self.config.system_prompt)
        params = AnthropicAdapter.build_params(
            model, msg_list, tools=tools, max_tokens=self.config.max_tokens, system=system
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

        if not self._provider_supports_streaming(provider):
            logger.info(
                "Streaming disabled for %s provider; falling back to one-shot chat", provider
            )
            response = await self.chat(messages, tools=tools, **kwargs)
            yield StreamChunk(
                content=response.content,
                thought=response.thought,
                tool_calls=response.tool_calls,
                finish_reason=response.finish_reason,
                usage=response.usage,
            )
            return

        if provider == Provider.GOOGLE:
            async for chunk in self._stream_google(messages, tools, **kwargs):
                yield chunk
        elif provider == Provider.OPENAI:
            async for chunk in self._stream_openai(messages, tools, **kwargs):
                yield chunk
        else:
            async for chunk in self._stream_anthropic(messages, tools, **kwargs):
                yield chunk

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
            messages, self.config.system_prompt, types_module=types
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
        tools = self._prepare_tools_for_provider(Provider.OPENAI, tools)
        if self.config.openai_protocol == "responses":
            response_tools = self._build_openai_responses_tools(tools)
            stream = await client.responses.create(
                **self._build_openai_responses_params(
                    model=model,
                    msg_list=msg_list,
                    tools=response_tools,
                    temperature=temperature,
                    max_output_tokens=self.config.max_tokens,
                    include=self.config.openai_responses_include,
                    stream=True,
                )
            )

            pending_tool_calls: dict[str, dict[str, Any]] = {}

            async for event in stream:
                event_type = getattr(event, "type", None) or (
                    event.get("type") if isinstance(event, dict) else None
                )
                if event_type == "response.output_text.delta":
                    delta = getattr(event, "delta", None)
                    if delta is None and isinstance(event, dict):
                        delta = event.get("delta")
                    if delta:
                        yield StreamChunk(content=delta)
                elif event_type == "response.output_item.added":
                    item = getattr(event, "item", None)
                    if item is None and isinstance(event, dict):
                        item = event.get("item")
                    serialized = self._serialize_openai_response_output_item(item) if item else None
                    if serialized and serialized.get("type") == "function_call":
                        call_id = serialized.get("call_id", "")
                        pending_tool_calls[call_id] = {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": serialized.get("name", ""),
                                "arguments": "",
                            },
                        }
                elif event_type == "response.function_call_arguments.delta":
                    delta = getattr(event, "delta", None)
                    if delta is None and isinstance(event, dict):
                        delta = event.get("delta")
                    if delta:
                        call_id = getattr(event, "call_id", None) or (
                            event.get("call_id") if isinstance(event, dict) else None
                        )
                        if call_id and call_id in pending_tool_calls:
                            pending_tool_calls[call_id]["function"]["arguments"] += delta
                elif event_type == "response.completed":
                    for tc in pending_tool_calls.values():
                        yield StreamChunk(tool_calls=[tc])
                    pending_tool_calls.clear()
                    yield StreamChunk(finish_reason="stop")
                elif event_type == "response.failed":
                    response = getattr(event, "response", None)
                    if response is None and isinstance(event, dict):
                        response = event.get("response")
                    raise RuntimeError(f"OpenAI Responses API stream failed: {response}")
            return

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
                        thought_signature = getattr(tc.function, "thought_signature", None)
                        if thought_signature is None:
                            thought_signature = getattr(tc, "thought_signature", None)
                        if thought_signature is not None:
                            serialized_signature = _serialize_gemini_thought_signature(
                                thought_signature
                            )
                            tc_dict["thought_signature"] = serialized_signature
                            tc_dict["function"]["thought_signature"] = serialized_signature
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
        protocol_mode = self.config.anthropic_protocol
        if protocol_mode not in {"auto", "messages"}:
            raise RuntimeError(f"Unsupported Anthropic protocol: {protocol_mode}")

        client = self._providers[Provider.ANTHROPIC]
        model = kwargs.get("model", self.config.model)
        tools = self._prepare_tools_for_provider(Provider.ANTHROPIC, tools)

        system, msg_list = AnthropicAdapter.convert_messages(messages, self.config.system_prompt)
        params = AnthropicAdapter.build_params(
            model, msg_list, tools=tools, max_tokens=self.config.max_tokens, system=system
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
        self._openai_protocol = (
            self.config.openai_protocol if self.config.openai_protocol != "auto" else None
        )

        for breaker in self._circuit_breakers.values():
            breaker.reset()

    def _prepare_tools_for_provider(
        self,
        provider: Provider,
        tools: Sequence[ToolDefinition | dict] | None,
    ) -> Sequence[ToolDefinition | dict] | None:
        """Drop tool definitions when the configured upstream does not support them."""
        if tools and not self._provider_supports_tools(provider):
            logger.info("Tool calling disabled for %s provider by configuration", provider.value)
            return None
        return tools

    def _provider_supports_tools(self, provider: Provider) -> bool:
        if provider == Provider.OPENAI:
            return self.config.openai_capabilities.tools
        if provider == Provider.ANTHROPIC:
            return self.config.anthropic_capabilities.tools
        return True

    def _provider_supports_streaming(self, provider: Provider) -> bool:
        if provider == Provider.OPENAI:
            return self.config.openai_capabilities.streaming
        if provider == Provider.ANTHROPIC:
            return self.config.anthropic_capabilities.streaming
        return True
