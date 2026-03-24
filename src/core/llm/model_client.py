"""
Unified Model Client

Provides a unified interface for both Gateway and Direct modes.

Modes:
1. Gateway Mode: Connect to a Model Gateway server
   - Reads: model + gateway_url + gateway_key from .agent/config.json
   - Supports all providers through the gateway

2. Direct Mode: Connect directly to provider APIs
   - Reads provider keys from .agent/config.json
   - Supports provider-specific features

Usage:
    # Auto-detect mode based on project config
    client = await ModelClient.create()

    # Chat
    response = await client.chat(messages, tools)

    # Stream
    async for chunk in client.chat_stream(messages, tools):
        print(chunk.content)
"""

from src.core.llm.model_client_base import BaseModelClient
from src.core.llm.model_client_direct import DirectModelClient
from src.core.llm.model_client_gateway import GatewayModelClient
from src.core.llm.model_utils import ClientConfig, ClientMode

__all__ = [
    "BaseModelClient",
    "GatewayModelClient",
    "DirectModelClient",
    "ModelClient",
]


class ModelClient:
    """
    Unified model client factory.

    Auto-detects mode based on project configuration.
    """

    @staticmethod
    async def create(config: ClientConfig | None = None) -> BaseModelClient:
        """
        Create a model client based on configuration.

        Args:
            config: Optional configuration. If not provided, loads from project config.

        Returns:
            Configured model client (Gateway or Direct mode)
        """
        if config is None:
            config = ClientConfig.from_env()

        if config.mode == ClientMode.GATEWAY:
            client: BaseModelClient = GatewayModelClient(config)
        else:
            client = DirectModelClient(config)

        await client.connect()
        return client

    @staticmethod
    def get_mode() -> ClientMode:
        """Get current mode based on project configuration."""
        config = ClientConfig.from_env()
        return config.mode

    @staticmethod
    def get_mode_info() -> dict:
        """Get information about current mode configuration."""
        config = ClientConfig.from_env()
        mode = config.mode

        if mode == ClientMode.GATEWAY:
            return {
                "mode": "gateway",
                "gateway_url": config.gateway_url,
                "has_key": bool(config.gateway_key),
            }
        else:
            return {
                "mode": "direct",
                "providers": {
                    "google": bool(config.google_api_key),
                    "openai": bool(config.openai_api_key),
                    "anthropic": bool(config.anthropic_api_key),
                    "ollama": bool(config.ollama_base_url),
                },
            }
