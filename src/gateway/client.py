"""Gateway Client - Simple client for the Model Gateway."""

import json
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class GatewayConfig:
    """Gateway client configuration."""
    base_url: str = "http://localhost:8000"
    api_key: str | None = None
    timeout: float = 120.0

    @classmethod
    def from_env(cls) -> "GatewayConfig":
        return cls(
            base_url=os.getenv("POLYMATH_GATEWAY_URL", "http://localhost:8000"),
            api_key=os.getenv("POLYMATH_API_KEY"),
            timeout=float(os.getenv("POLYMATH_GATEWAY_TIMEOUT", "120")),
        )


class GatewayClient:
    """Client for the Polymath Model Gateway."""

    def __init__(self, config: GatewayConfig | None = None):
        self.config = config or GatewayConfig.from_env()
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "GatewayClient":
        await self.connect()
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    async def connect(self) -> None:
        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers=headers,
            timeout=httpx.Timeout(self.config.timeout),
        )

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Send a chat request."""
        if not self._client:
            raise RuntimeError("Client not connected")
        payload = {"model": model, "messages": messages, "temperature": temperature, **kwargs}
        if tools:
            payload["tools"] = tools
        if max_tokens:
            payload["max_tokens"] = max_tokens
        response = await self._client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()

    async def chat_stream(
        self, model: str, messages: list[dict[str, Any]], **kwargs
    ) -> AsyncIterator[dict[str, Any]]:
        """Send a streaming chat request."""
        if not self._client:
            raise RuntimeError("Client not connected")
        payload = {"model": model, "messages": messages, "stream": True, **kwargs}
        async with self._client.stream("POST", "/v1/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line and line.startswith("data: "):
                    data = line[6:]
                    if data != "[DONE]":
                        try:
                            yield json.loads(data)
                        except json.JSONDecodeError:
                            pass

    async def list_models(self) -> list[dict[str, Any]]:
        """List available models."""
        if not self._client:
            raise RuntimeError("Client not connected")
        response = await self._client.get("/v1/models")
        response.raise_for_status()
        return response.json().get("data", [])

    async def health(self) -> dict[str, Any]:
        """Check gateway health."""
        if not self._client:
            raise RuntimeError("Client not connected")
        response = await self._client.get("/health")
        response.raise_for_status()
        return response.json()


async def create_client() -> GatewayClient:
    """Create and connect a gateway client."""
    client = GatewayClient()
    await client.connect()
    return client
