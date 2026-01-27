"""
Model Manager

Runtime model switching and configuration.

Features:
- List available models
- Switch models at runtime
- Model aliases
- Model capability detection
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ModelCapability(Enum):
    """Model capabilities."""

    TEXT = "text"
    VISION = "vision"
    CODE = "code"
    REASONING = "reasoning"
    LONG_CONTEXT = "long_context"
    FUNCTION_CALLING = "function_calling"


@dataclass
class ModelInfo:
    """Information about a model."""

    id: str
    name: str
    provider: str
    description: str
    context_window: int
    max_output: int
    input_price: float  # per 1M tokens
    output_price: float  # per 1M tokens
    capabilities: list[ModelCapability]
    aliases: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "description": self.description,
            "context_window": self.context_window,
            "max_output": self.max_output,
            "input_price": self.input_price,
            "output_price": self.output_price,
            "capabilities": [c.value for c in self.capabilities],
            "aliases": self.aliases,
        }


# Available models registry
AVAILABLE_MODELS: dict[str, ModelInfo] = {
    # Google Gemini 3 Series
    "gemini-3-pro-preview": ModelInfo(
        id="gemini-3-pro-preview",
        name="Gemini 3 Pro",
        provider="google",
        description="Most capable model with multimodal understanding",
        context_window=1_000_000,
        max_output=64_000,
        input_price=2.00,
        output_price=12.00,
        capabilities=[
            ModelCapability.TEXT,
            ModelCapability.VISION,
            ModelCapability.CODE,
            ModelCapability.REASONING,
            ModelCapability.LONG_CONTEXT,
            ModelCapability.FUNCTION_CALLING,
        ],
        aliases=["gemini-3-pro", "g3p", "pro"],
    ),
    "gemini-3-flash-preview": ModelInfo(
        id="gemini-3-flash-preview",
        name="Gemini 3 Flash",
        provider="google",
        description="Fast and capable, best balance of speed and quality",
        context_window=1_000_000,
        max_output=64_000,
        input_price=0.50,
        output_price=3.00,
        capabilities=[
            ModelCapability.TEXT,
            ModelCapability.VISION,
            ModelCapability.CODE,
            ModelCapability.REASONING,
            ModelCapability.LONG_CONTEXT,
            ModelCapability.FUNCTION_CALLING,
        ],
        aliases=["gemini-3-flash", "g3f", "flash"],
    ),
    # OpenAI GPT-5 Series
    "gpt-5": ModelInfo(
        id="gpt-5",
        name="GPT-5",
        provider="openai",
        description="OpenAI's flagship model",
        context_window=256_000,
        max_output=32_000,
        input_price=5.00,
        output_price=15.00,
        capabilities=[
            ModelCapability.TEXT,
            ModelCapability.VISION,
            ModelCapability.CODE,
            ModelCapability.REASONING,
            ModelCapability.FUNCTION_CALLING,
        ],
        aliases=["gpt5"],
    ),
    "gpt-5-mini": ModelInfo(
        id="gpt-5-mini",
        name="GPT-5 Mini",
        provider="openai",
        description="Smaller, faster GPT-5 variant",
        context_window=128_000,
        max_output=16_000,
        input_price=2.50,
        output_price=2.00,
        capabilities=[
            ModelCapability.TEXT,
            ModelCapability.CODE,
            ModelCapability.FUNCTION_CALLING,
        ],
        aliases=["gpt5-mini", "mini"],
    ),
    # Anthropic Claude 4.5 Series
    "claude-opus-4.5": ModelInfo(
        id="claude-opus-4.5",
        name="Claude Opus 4.5",
        provider="anthropic",
        description="Anthropic's most intelligent model",
        context_window=200_000,
        max_output=32_000,
        input_price=5.00,
        output_price=25.00,
        capabilities=[
            ModelCapability.TEXT,
            ModelCapability.VISION,
            ModelCapability.CODE,
            ModelCapability.REASONING,
            ModelCapability.FUNCTION_CALLING,
        ],
        aliases=["opus", "claude-opus"],
    ),
    "claude-sonnet-4.5": ModelInfo(
        id="claude-sonnet-4.5",
        name="Claude Sonnet 4.5",
        provider="anthropic",
        description="Balance of intelligence and efficiency",
        context_window=200_000,
        max_output=32_000,
        input_price=3.00,
        output_price=15.00,
        capabilities=[
            ModelCapability.TEXT,
            ModelCapability.VISION,
            ModelCapability.CODE,
            ModelCapability.REASONING,
            ModelCapability.FUNCTION_CALLING,
        ],
        aliases=["sonnet", "claude-sonnet"],
    ),
}


class ModelManager:
    """
    Manages model selection and switching.

    Usage:
        mgr = ModelManager()

        # List models
        models = mgr.list_models()

        # Get current model
        current = mgr.get_current_model()

        # Switch model
        mgr.switch_model("gemini-3-pro")
        mgr.switch_model("flash")  # Using alias

        # Get model info
        info = mgr.get_model_info("gemini-3-flash-preview")
    """

    def __init__(self, default_model: str | None = None):
        """
        Initialize model manager.

        Args:
            default_model: Default model ID
        """
        self._current_model = default_model or os.getenv(
            "DORAEMON_MODEL", "gemini-3-pro-preview"
        )
        self._model_history: list[str] = [self._current_model]

    def list_models(self, provider: str | None = None) -> list[ModelInfo]:
        """
        List available models.

        Args:
            provider: Filter by provider

        Returns:
            List of ModelInfo
        """
        models = list(AVAILABLE_MODELS.values())

        if provider:
            models = [m for m in models if m.provider == provider]

        return models

    def get_model_info(self, model_id: str) -> ModelInfo | None:
        """
        Get information about a model.

        Args:
            model_id: Model ID or alias

        Returns:
            ModelInfo or None
        """
        # Direct lookup
        if model_id in AVAILABLE_MODELS:
            return AVAILABLE_MODELS[model_id]

        # Alias lookup
        for model in AVAILABLE_MODELS.values():
            if model_id in model.aliases:
                return model

        return None

    def resolve_model_id(self, model_id_or_alias: str) -> str | None:
        """
        Resolve a model ID or alias to the actual model ID.

        Args:
            model_id_or_alias: Model ID or alias

        Returns:
            Resolved model ID or None
        """
        # Direct match
        if model_id_or_alias in AVAILABLE_MODELS:
            return model_id_or_alias

        # Alias match
        for model_id, model in AVAILABLE_MODELS.items():
            if model_id_or_alias in model.aliases:
                return model_id

        return None

    def get_current_model(self) -> str:
        """Get current model ID."""
        return self._current_model

    def get_current_model_info(self) -> ModelInfo | None:
        """Get info for current model."""
        return self.get_model_info(self._current_model)

    def switch_model(self, model_id_or_alias: str) -> bool:
        """
        Switch to a different model.

        Args:
            model_id_or_alias: Model ID or alias

        Returns:
            True if switch successful
        """
        resolved = self.resolve_model_id(model_id_or_alias)

        if not resolved:
            logger.error(f"Unknown model: {model_id_or_alias}")
            return False

        old_model = self._current_model
        self._current_model = resolved
        self._model_history.append(resolved)

        logger.info(f"Switched model: {old_model} -> {resolved}")
        return True

    def switch_previous(self) -> bool:
        """Switch to the previous model."""
        if len(self._model_history) < 2:
            return False

        # Remove current
        self._model_history.pop()
        # Get previous
        self._current_model = self._model_history[-1]
        return True

    def has_capability(self, capability: ModelCapability) -> bool:
        """Check if current model has a capability."""
        model = self.get_current_model_info()
        return model is not None and capability in model.capabilities

    def get_models_with_capability(
        self, capability: ModelCapability
    ) -> list[ModelInfo]:
        """Get all models with a specific capability."""
        return [
            m for m in AVAILABLE_MODELS.values() if capability in m.capabilities
        ]

    def format_model_list(self) -> str:
        """Format model list for display."""
        lines = []

        # Group by provider
        by_provider: dict[str, list[ModelInfo]] = {}
        for model in AVAILABLE_MODELS.values():
            if model.provider not in by_provider:
                by_provider[model.provider] = []
            by_provider[model.provider].append(model)

        for provider, models in by_provider.items():
            lines.append(f"\n[{provider.upper()}]")
            for model in models:
                current = "→ " if model.id == self._current_model else "  "
                price = f"${model.input_price}/${model.output_price}"
                aliases = f" ({', '.join(model.aliases)})" if model.aliases else ""
                lines.append(
                    f"{current}{model.id}: {model.name} - {price}{aliases}"
                )

        return "\n".join(lines)

    def get_summary(self) -> dict[str, Any]:
        """Get model manager summary."""
        current_info = self.get_current_model_info()
        return {
            "current_model": self._current_model,
            "current_info": current_info.to_dict() if current_info else None,
            "available_count": len(AVAILABLE_MODELS),
            "history": self._model_history[-5:],
        }
