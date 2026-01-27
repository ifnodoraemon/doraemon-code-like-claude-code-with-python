"""
Unified Model Client

Provides a unified interface for both Gateway and Direct modes.

Modes:
1. Gateway Mode: Connect to a Model Gateway server
   - Only needs: POLYMATH_GATEWAY_URL + POLYMATH_API_KEY
   - Supports all providers through the gateway

2. Direct Mode: Connect directly to provider APIs
   - Needs individual API keys: GOOGLE_API_KEY, OPENAI_API_KEY, etc.
   - Supports provider-specific features

Usage:
    # Auto-detect mode based on environment
    client = await ModelClient.create()

    # Chat
    response = await client.chat(messages, tools)

    # Stream
    async for chunk in client.chat_stream(messages, tools):
        print(chunk.content)
"""

import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ClientMode(Enum):
    """Client connection mode."""
    GATEWAY = "gateway"
    DIRECT = "direct"


class Provider(Enum):
    """Supported providers for direct mode."""
    GOOGLE = "google"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


@dataclass
class Message:
    """Unified message format."""
    role: str
    content: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict:
        result = {"role": self.role}
        if self.content is not None:
            result["content"] = self.content
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.name:
            result["name"] = self.name
        return result


@dataclass
class ToolDefinition:
    """Tool/function definition."""
    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai_format(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_genai_format(self):
        """Convert to Google GenAI format."""
        from google.genai import types
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


@dataclass
class ChatResponse:
    """Unified chat response."""
    content: str | None = None
    tool_calls: list[dict] | None = None
    finish_reason: str | None = None
    usage: dict | None = None
    raw: Any = None  # Original response for provider-specific access

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class StreamChunk:
    """Streaming response chunk."""
    content: str | None = None
    tool_calls: list[dict] | None = None
    finish_reason: str | None = None
    usage: dict | None = None


@dataclass
class ToolCall:
    """A tool/function call from the model."""
    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ToolCall":
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            arguments=data.get("arguments", {}),
        )



@dataclass
class ClientConfig:
    """Model client configuration."""
    mode: ClientMode = ClientMode.DIRECT
    model: str = "gemini-2.5-flash-preview"
    temperature: float = 0.7
    max_tokens: int | None = None
    system_prompt: str | None = None

    # Gateway mode settings
    gateway_url: str | None = None
    gateway_key: str | None = None

    # Direct mode settings - provider API keys
    google_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"

    @classmethod
    def from_env(cls) -> "ClientConfig":
        """Load configuration from environment variables."""
        # Check if gateway mode is configured
        gateway_url = os.getenv("POLYMATH_GATEWAY_URL")

        if gateway_url:
            mode = ClientMode.GATEWAY
        else:
            mode = ClientMode.DIRECT

        return cls(
            mode=mode,
            model=os.getenv("POLYMATH_MODEL", "gemini-2.5-flash-preview"),
            # Gateway settings
            gateway_url=gateway_url,
            gateway_key=os.getenv("POLYMATH_API_KEY"),
            # Direct mode API keys
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            ollama_base_url=os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
        )


class BaseModelClient(ABC):
    """Base class for model clients."""

    @abstractmethod
    async def chat(
        self,
        messages: list[Message | dict],
        tools: list[ToolDefinition | dict] | None = None,
        **kwargs,
    ) -> ChatResponse:
        """Send a chat request."""
        pass

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[Message | dict],
        tools: list[ToolDefinition | dict] | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Send a streaming chat request."""
        pass

    @abstractmethod
    async def list_models(self) -> list[dict]:
        """List available models."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the client."""
        pass


class GatewayModelClient(BaseModelClient):
    """Client that connects to the Model Gateway."""

    def __init__(self, config: ClientConfig):
        self.config = config
        self._client = None

    async def connect(self) -> None:
        """Initialize HTTP client."""
        import httpx

        headers = {}
        if self.config.gateway_key:
            headers["Authorization"] = f"Bearer {self.config.gateway_key}"

        self._client = httpx.AsyncClient(
            base_url=self.config.gateway_url,
            headers=headers,
            timeout=httpx.Timeout(120.0),
        )
        logger.info(f"Connected to gateway: {self.config.gateway_url}")

    async def chat(
        self,
        messages: list[Message | dict],
        tools: list[ToolDefinition | dict] | None = None,
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

        response = await self._client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

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
        messages: list[Message | dict],
        tools: list[ToolDefinition | dict] | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        if not self._client:
            await self.connect()

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
                    yield StreamChunk(
                        content=delta.get("content"),
                        tool_calls=delta.get("tool_calls"),
                        finish_reason=choice.get("finish_reason"),
                        usage=chunk.get("usage"),
                    )
                except json.JSONDecodeError:
                    continue

    async def list_models(self) -> list[dict]:
        if not self._client:
            await self.connect()
        response = await self._client.get("/v1/models")
        response.raise_for_status()
        return response.json().get("data", [])

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
        messages: list[Message | dict],
        tools: list[ToolDefinition | dict] | None = None,
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
        messages: list[Message | dict],
        tools: list[ToolDefinition | dict] | None = None,
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
                parts.append({"text": msg["content"]})

            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    args_str = func.get("arguments", "{}")
                    try:
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse tool arguments: {args_str}")
                        args = {}
                    parts.append({
                        "function_call": {
                            "name": func.get("name"),
                            "args": args,
                        }
                    })

            if role == "tool" and msg.get("tool_call_id"):
                parts.append({
                    "function_response": {
                        "name": msg.get("name", "function"),
                        "response": {"result": msg.get("content", "")},
                    }
                })

            if parts:
                contents.append({"role": gemini_role, "parts": parts})

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
        for part in candidate.content.parts:
            if hasattr(part, "text") and part.text:
                texts.append(part.text)
            elif hasattr(part, "function_call") and part.function_call:
                if tool_calls is None:
                    tool_calls = []
                fc = part.function_call
                # Safely convert args to dict
                try:
                    args = dict(fc.args) if fc.args else {}
                except (TypeError, ValueError):
                    logger.warning(f"Failed to convert function call args: {fc.args}")
                    args = {}
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": fc.name,
                        "arguments": json.dumps(args),
                    },
                })
        if texts:
            content = "".join(texts)

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
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            raw=response,
        )

    async def _chat_openai(
        self,
        messages: list[Message | dict],
        tools: list[ToolDefinition | dict] | None = None,
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
        messages: list[Message | dict],
        tools: list[ToolDefinition | dict] | None = None,
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
        messages: list[Message | dict],
        tools: list[ToolDefinition | dict] | None = None,
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
        messages: list[Message | dict],
        tools: list[ToolDefinition | dict] | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Streaming chat - currently only implemented for gateway mode."""
        # For simplicity, fall back to non-streaming for direct mode
        response = await self.chat(messages, tools, **kwargs)
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


class ModelClient:
    """
    Unified model client factory.

    Auto-detects mode based on environment configuration.
    """

    @staticmethod
    async def create(config: ClientConfig | None = None) -> BaseModelClient:
        """
        Create a model client based on configuration.

        Args:
            config: Optional configuration. If not provided, loads from environment.

        Returns:
            Configured model client (Gateway or Direct mode)
        """
        if config is None:
            config = ClientConfig.from_env()

        if config.mode == ClientMode.GATEWAY:
            client = GatewayModelClient(config)
        else:
            client = DirectModelClient(config)

        await client.connect()
        return client

    @staticmethod
    def get_mode() -> ClientMode:
        """Get current mode based on environment."""
        if os.getenv("POLYMATH_GATEWAY_URL"):
            return ClientMode.GATEWAY
        return ClientMode.DIRECT

    @staticmethod
    def get_mode_info() -> dict:
        """Get information about current mode configuration."""
        mode = ModelClient.get_mode()

        if mode == ClientMode.GATEWAY:
            return {
                "mode": "gateway",
                "gateway_url": os.getenv("POLYMATH_GATEWAY_URL"),
                "has_key": bool(os.getenv("POLYMATH_API_KEY")),
            }
        else:
            return {
                "mode": "direct",
                "providers": {
                    "google": bool(os.getenv("GOOGLE_API_KEY")),
                    "openai": bool(os.getenv("OPENAI_API_KEY")),
                    "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
                    "ollama": True,
                },
            }
