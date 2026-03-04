"""
Gateway Model Client

Client that connects to the Model Gateway server.
"""

import json
import logging
from collections.abc import AsyncIterator, Sequence

from src.core.model_client_base import BaseModelClient
from src.core.model_utils import (
    ChatResponse,
    ClientConfig,
    Message,
    StreamChunk,
    ToolDefinition,
)

logger = logging.getLogger(__name__)


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
        return False

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
        import httpx

        from src.core.errors import (
            DoraemonException,
            ErrorCategory,
            RateLimitError,
            TransientError,
            retry,
        )

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
                    retry_after = int(e.response.headers.get("Retry-After", 60))
                    raise RateLimitError(
                        "Rate limit exceeded",
                        retry_after=retry_after,
                        context={"endpoint": endpoint, "status": 429}
                    ) from e
                elif e.response.status_code >= 500:
                    raise TransientError(
                        f"Server error: {e.response.status_code}",
                        retry_after=2.0,
                        context={"endpoint": endpoint, "status": e.response.status_code}
                    ) from e
                else:
                    raise DoraemonException(
                        f"API error: {e.response.status_code} - {e.response.text}",
                        category=ErrorCategory.PERMANENT,
                        context={"endpoint": endpoint, "status": e.response.status_code}
                    ) from e
            except httpx.RequestError as e:
                raise TransientError(
                    f"Network error: {str(e)}",
                    retry_after=2.0,
                    context={"endpoint": endpoint, "error": str(e)}
                ) from e

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

        data = await self._make_api_call("/v1/chat/completions", payload)

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

        try:
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
                            thought=delta.get("thought"),
                            tool_calls=delta.get("tool_calls"),
                            finish_reason=choice.get("finish_reason"),
                            usage=chunk.get("usage"),
                        )
                    except json.JSONDecodeError:
                        continue
        except (httpx.StreamError, httpx.RemoteProtocolError) as e:
            logger.error(f"Stream connection error: {e}")
            raise

    async def list_models(self) -> list[dict]:
        """List available models from gateway."""
        data = await self._make_api_call("/v1/models", method="GET")
        return data.get("data", [])

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
