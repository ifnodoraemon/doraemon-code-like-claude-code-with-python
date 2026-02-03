"""
Model Adapters

Provider-specific adapters for Gateway and Direct modes.
Handles communication with different LLM providers.
"""

import json
import logging
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import Any

from src.core.model_utils import (
    ChatResponse,
    ClientConfig,
    ClientMode,
    Message,
    Provider,
    StreamChunk,
    ToolDefinition,
)

logger = logging.getLogger(__name__)


class BaseModelClient(ABC):
    """Base class for model clients."""

    @abstractmethod
    async def chat(
        self,
        messages: Sequence[Message | dict],
        tools: Sequence[ToolDefinition | dict] | None = None,
        **kwargs,
    ) -> ChatResponse:
        """Send a chat request."""
        pass

    @abstractmethod
    def chat_stream(
        self,
        messages: Sequence[Message | dict],
        tools: Sequence[ToolDefinition | dict] | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Send a streaming chat request."""
        pass

    @abstractmethod
    async def list_models(self) -> list[dict]:
        """List available models."""
        pass

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the model provider/gateway."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the client."""
        pass


class GatewayModelClient(BaseModelClient):
    """Client that connects to the Model Gateway."""

    def __init__(self, config: ClientConfig):
        self.config = config
        from httpx import AsyncClient
        self._client: AsyncClient | None = None

    async def __aenter__(self):
        """Context manager entry - ensure client is connected."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure client is closed."""
        await self.close()
        return False  # Don't suppress exceptions

    async def connect(self) -> None:
        """Initialize HTTP client."""
        import httpx

        headers = {}
        if self.config.gateway_key:
            headers["Authorization"] = f"Bearer {self.config.gateway_key}"

        if not self.config.gateway_url:
            raise ValueError("Gateway URL must be set for Gateway mode")

        self._client = httpx.AsyncClient(
            base_url=self.config.gateway_url,
            headers=headers,
            timeout=httpx.Timeout(120.0),
        )
        logger.info(f"Connected to gateway: {self.config.gateway_url}")

    async def _make_api_call(self, endpoint: str, payload: dict | None = None, method: str = "POST") -> dict:
        """
        Make API call with automatic retry on transient errors.

        Args:
            endpoint: API endpoint path
            payload: Request payload (for POST) or None (for GET)
            method: HTTP method (POST or GET)

        Returns:
            Response JSON data

        Raises:
            RateLimitError: When rate limited
            TransientError: When server error occurs
            DoraemonException: For other API errors
        """
        from src.core.errors import (
            DoraemonException,
            ErrorCategory,
            RateLimitError,
            TransientError,
            retry,
        )
        import httpx

        @retry(
            max_attempts=3,
            initial_delay=1.0,
            max_delay=60.0,
            exceptions=(TransientError, RateLimitError),
        )
        async def _call():
            if self._client is None:
                await self.connect()
            if self._client is None:
                from src.core.errors import ConfigurationError
                raise ConfigurationError("Failed to initialize HTTP client")

            try:
                if method == "GET":
                    response = await self._client.get(endpoint)
                else:
                    response = await self._client.post(endpoint, json=payload)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    # Rate limit - extract retry-after header
                    retry_after = int(e.response.headers.get("Retry-After", 60))
                    raise RateLimitError(
                        "Rate limit exceeded",
                        retry_after=retry_after,
                        context={"endpoint": endpoint, "status": 429}
                    )
                elif e.response.status_code >= 500:
                    # Server error - transient, can retry
                    raise TransientError(
                        f"Server error: {e.response.status_code}",
                        retry_after=2.0,
                        context={"endpoint": endpoint, "status": e.response.status_code}
                    )
                else:
                    # Client error - permanent
                    raise DoraemonException(
                        f"API error: {e.response.status_code} - {e.response.text}",
                        category=ErrorCategory.PERMANENT,
                        context={"endpoint": endpoint, "status": e.response.status_code}
                    )
            except httpx.RequestError as e:
                # Network error - transient
                raise TransientError(
                    f"Network error: {str(e)}",
                    retry_after=2.0,
                    context={"endpoint": endpoint, "error": str(e)}
                )

        return await _call()

    async def chat(
        self,
        messages: Sequence[Message | dict],
        tools: Sequence[ToolDefinition | dict] | None = None,
        **kwargs,
    ) -> ChatResponse:
        if not self._client:
            await self.connect()

        # Normalize messages
        msg_list = []
        for m in messages:
            if isinstance(m, Message):
                msg_list.append(m.to_dict())
            else:
                msg_list.append(m)

        # Normalize tools
        tool_list = None
        if tools:
            tool_list = []
            for t in tools:
                if isinstance(t, ToolDefinition):
                    tool_list.append(t.to_openai_format())
                else:
                    tool_list.append(t)

        payload = {
            "model": kwargs.get("model", self.config.model),
            "messages": msg_list,
            "temperature": kwargs.get("temperature", self.config.temperature),
        }
        if tool_list:
            payload["tools"] = tool_list
        if self.config.max_tokens:
            payload["max_tokens"] = self.config.max_tokens

        # Use retry-enabled API call
        data = await self._make_api_call("/v1/chat/completions", payload)

        # Extract response - safely handle empty choices
        choices = data.get("choices", [])
        if not choices:
            return ChatResponse(
                content=None,
                tool_calls=None,
                finish_reason="error",
                usage=data.get("usage"),
                raw=data,
            )
        choice = choices[0]
        message = choice.get("message", {})

        return ChatResponse(
            content=message.get("content"),
            tool_calls=message.get("tool_calls"),
            finish_reason=choice.get("finish_reason"),
            usage=data.get("usage"),
            raw=data,
        )

    async def chat_stream(
        self,
        messages: Sequence[Message | dict],
        tools: Sequence[ToolDefinition | dict] | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        if not self._client:
            await self.connect()

        # Check for client existence
        if self._client is None:
            from src.core.errors import ConfigurationError
            raise ConfigurationError("Failed to initialize HTTP client for gateway mode")

        msg_list = [m.to_dict() if isinstance(m, Message) else m for m in messages]
        tool_list = None
        if tools:
            tool_list = [
                t.to_openai_format() if isinstance(t, ToolDefinition) else t
                for t in tools
            ]

        payload = {
            "model": kwargs.get("model", self.config.model),
            "messages": msg_list,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": True,
        }
        if tool_list:
            payload["tools"] = tool_list

        async with self._client.stream("POST", "/v1/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta", {})
                    # For Gemini via Gateway, thought might be in delta if we updated Gateway?
                    # Or maybe we need to update _chat_google in DIRECT mode?
                    # The code above (lines 342-365) is inside GatewayModelClient implementation for OpenAI compatible API.
                    # If Gateway supports thought, it should be in delta.

                    yield StreamChunk(
                        content=delta.get("content"),
                        thought=delta.get("thought"), # Add thought
                        tool_calls=delta.get("tool_calls"),
                        finish_reason=choice.get("finish_reason"),
                        usage=chunk.get("usage"),
                    )
                except json.JSONDecodeError:
                    continue

    async def list_models(self) -> list[dict]:
        """List available models from gateway."""
        data = await self._make_api_call("/v1/models", method="GET")
        return data.get("data", [])

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


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
        return False  # Don't suppress exceptions

    async def connect(self) -> None:
        """Initialize provider clients."""
        # Google Gemini
        if self.config.google_api_key:
            try:
                from google import genai
                self._providers[Provider.GOOGLE] = genai.Client(
                    api_key=self.config.google_api_key
                )
                logger.info("Google Gemini client initialized")
            except ImportError:
                logger.warning("google-genai not installed")

        # OpenAI
        if self.config.openai_api_key:
            try:
                from openai import AsyncOpenAI
                self._providers[Provider.OPENAI] = AsyncOpenAI(
                    api_key=self.config.openai_api_key
                )
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
        for provider, patterns in self.PROVIDER_PATTERNS.items():
            for pattern in patterns:
                if model.startswith(pattern):
                    if provider in self._providers:
                        return provider
        # Default to first available
        if not self._providers:
            raise RuntimeError("No providers available. Check your API keys.")
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
            return await self._chat_google(messages, tools, **kwargs)
        elif provider == Provider.OPENAI:
            return await self._chat_openai(messages, tools, **kwargs)
        elif provider == Provider.ANTHROPIC:
            return await self._chat_anthropic(messages, tools, **kwargs)
        else:
            return await self._chat_ollama(messages, tools, **kwargs)

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

        # Convert messages to Gemini format
        contents = []
        system_instruction = self.config.system_prompt

        for m in messages:
            msg = m if isinstance(m, dict) else m.to_dict()
            role = msg.get("role", "user")

            if role == "system":
                system_instruction = msg.get("content", "")
                continue

            gemini_role = "user" if role == "user" else "model"
            parts = []

            if msg.get("content"):
                parts.append(types.Part(text=msg["content"]))

            if msg.get("thought"):
                parts.append(types.Part(thought=msg["thought"]))

            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    args_str = func.get("arguments", "{}")
                    try:
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse tool arguments: {args_str}")
                        args = {}

                    fc_obj = types.FunctionCall(
                        name=func.get("name"),
                        args=args,
                    )

                    # SDK Part constructor handles thought_signature
                    thought_sig = tc.get("thought_signature") or func.get("thought_signature")

                    parts.append(types.Part(
                        function_call=fc_obj,
                        thought_signature=thought_sig
                    ))

            if role == "tool" and msg.get("tool_call_id"):
                parts.append(types.Part(
                    function_response=types.FunctionResponse(
                        name=msg.get("name", "function"),
                        response={"result": msg.get("content", "")},
                    )
                ))

            if parts:
                contents.append(types.Content(role=gemini_role, parts=parts))

        # Build config
        gen_config_dict = {
            "temperature": kwargs.get("temperature", self.config.temperature),
        }

        if self.config.max_tokens:
            gen_config_dict["max_output_tokens"] = self.config.max_tokens

        if tools:
            function_declarations = []
            for t in tools:
                if isinstance(t, ToolDefinition):
                    function_declarations.append(t.to_genai_format())
                else:
                    func = t.get("function", t)
                    function_declarations.append(
                        types.FunctionDeclaration(
                            name=func.get("name"),
                            description=func.get("description", ""),
                            parameters=func.get("parameters", {}),
                        )
                    )
            gen_config_dict["tools"] = [types.Tool(function_declarations=function_declarations)]
            gen_config_dict["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(
                disable=True
            )

        if system_instruction:
            gen_config_dict["system_instruction"] = system_instruction

        gen_config = types.GenerateContentConfig(**gen_config_dict)

        response = await client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=gen_config,
        )

        # Extract response - safely handle empty candidates
        content = None
        tool_calls = None
        finish_reason = "stop"

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
        texts = []
        thoughts = []
        for part in candidate.content.parts:
            if hasattr(part, "text") and part.text:
                texts.append(part.text)
            elif hasattr(part, "thought") and part.thought:
                thoughts.append(part.thought)
            elif hasattr(part, "function_call") and part.function_call:
                if tool_calls is None:
                    tool_calls = []
                fc = part.function_call
                # Safely convert args to dict
                try:
                    args = dict(fc.args) if fc.args else {}
                except (TypeError, ValueError) as e:
                    from src.core.errors import DoraemonException, ErrorCategory
                    raise DoraemonException(
                        f"Invalid tool arguments: {fc.args}",
                        category=ErrorCategory.PERMANENT,
                        context={"tool": fc.name, "args": str(fc.args)}
                    ) from e
                tc_dict = {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": fc.name,
                        "arguments": json.dumps(args),
                    },
                }
                # Capture thought_signature if present in the PART (neighbor to function_call)
                if hasattr(part, "thought_signature") and part.thought_signature:
                    tc_dict["thought_signature"] = part.thought_signature
                elif isinstance(part, dict) and part.get("thought_signature"):
                    tc_dict["thought_signature"] = part.get("thought_signature")

                tool_calls.append(tc_dict)
        if texts:
            content = "".join(texts)

        thought = "".join(thoughts) if thoughts else None

        if tool_calls:
            finish_reason = "tool_calls"

        usage = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count or 0,
                "completion_tokens": response.usage_metadata.candidates_token_count or 0,
                "total_tokens": response.usage_metadata.total_token_count or 0,
            }

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

        msg_list = []
        for m in messages:
            msg_list.append(m if isinstance(m, dict) else m.to_dict())

        params = {
            "model": model,
            "messages": msg_list,
            "temperature": kwargs.get("temperature", self.config.temperature),
        }

        if tools:
            params["tools"] = [
                t.to_openai_format() if isinstance(t, ToolDefinition) else t
                for t in tools
            ]

        if self.config.max_tokens:
            params["max_tokens"] = self.config.max_tokens

        response = await client.chat.completions.create(**params)

        # Safely handle empty choices
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
            } if response.usage else None,
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

        system = self.config.system_prompt
        msg_list = []

        for m in messages:
            msg = m if isinstance(m, dict) else m.to_dict()
            role = msg.get("role", "user")

            if role == "system":
                system = msg.get("content", "")
                continue

            if role == "tool":
                msg_list.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id"),
                        "content": msg.get("content", ""),
                    }],
                })
            else:
                content = []
                if msg.get("content"):
                    content.append({"type": "text", "text": msg["content"]})
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        func = tc.get("function", {})
                        args_str = func.get("arguments", "{}")
                        try:
                            args = json.loads(args_str) if isinstance(args_str, str) else args_str
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse tool arguments: {args_str}")
                            args = {}
                        content.append({
                            "type": "tool_use",
                            "id": tc.get("id"),
                            "name": func.get("name"),
                            "input": args,
                        })
                msg_list.append({
                    "role": "assistant" if role == "assistant" else "user",
                    "content": content or msg.get("content"),
                })

        params = {
            "model": model,
            "messages": msg_list,
            "max_tokens": self.config.max_tokens or 4096,
        }

        if system:
            params["system"] = system

        if tools:
            params["tools"] = [
                {
                    "name": t.name if isinstance(t, ToolDefinition) else t["function"]["name"],
                    "description": t.description if isinstance(t, ToolDefinition) else t["function"].get("description", ""),
                    "input_schema": t.parameters if isinstance(t, ToolDefinition) else t["function"].get("parameters", {}),
                }
                for t in tools
            ]

        response = await client.messages.create(**params)

        content = None
        tool_calls = None

        for block in response.content:
            if block.type == "text":
                content = (content or "") + block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                })

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
            msg_list.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })

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
        """Streaming chat - currently only implemented for gateway mode."""
        # For simplicity, fall back to non-streaming for direct mode
        response = await self.chat(messages, tools, **kwargs)
        if False:
            yield StreamChunk()  # Ensure it's an async generator
        yield StreamChunk(
            content=response.content,
            tool_calls=response.tool_calls,
            finish_reason=response.finish_reason,
            usage=response.usage,
        )

    async def list_models(self) -> list[dict]:
        """List available models from all providers."""
        models = []

        if Provider.GOOGLE in self._providers:
            models.extend([
                {"id": "gemini-2.5-flash-preview", "provider": "google"},
                {"id": "gemini-2.5-pro-preview", "provider": "google"},
            ])

        if Provider.OPENAI in self._providers:
            models.extend([
                {"id": "gpt-4o", "provider": "openai"},
                {"id": "gpt-4o-mini", "provider": "openai"},
            ])

        if Provider.ANTHROPIC in self._providers:
            models.extend([
                {"id": "claude-sonnet-4-20250514", "provider": "anthropic"},
                {"id": "claude-3-5-haiku-20241022", "provider": "anthropic"},
            ])

        return models

    async def close(self) -> None:
        """Close all provider clients."""
        for provider, client in self._providers.items():
            if provider == Provider.OLLAMA and hasattr(client, "aclose"):
                await client.aclose()
        self._providers.clear()
