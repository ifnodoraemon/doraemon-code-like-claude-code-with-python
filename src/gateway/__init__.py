"""
Doraemon Model Gateway

A unified API gateway for multiple AI model providers.

Architecture:
    CLI ──► Gateway Server ──► Provider APIs
                │
                ├── Google (Gemini)
                ├── OpenAI (GPT)
                ├── Anthropic (Claude)
                └── Ollama (Local)

Usage:
    # Start gateway server
    python -m src.gateway.server

    # CLI only needs:
    DORAEMON_GATEWAY_URL=http://localhost:8000
    DORAEMON_API_KEY=your-gateway-key
"""

from .client import GatewayClient, GatewayConfig, create_client
from .router import ModelRouter
from .schema import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ModelInfo,
    ToolCall,
    ToolResult,
    Usage,
)

__all__ = [
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ModelInfo",
    "ToolCall",
    "ToolResult",
    "Usage",
    "GatewayClient",
    "GatewayConfig",
    "create_client",
    "ModelRouter",
]
