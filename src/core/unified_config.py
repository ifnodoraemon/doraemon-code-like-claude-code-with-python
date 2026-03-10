"""
Unified Configuration System

Consolidates all configuration classes into a single, validated configuration.
Supports loading from environment variables and config files with clear precedence.

Precedence: Environment Variables > Config File > Defaults
"""

import json
import os
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from .paths import config_path as default_config_path


class UnifiedConfig(BaseModel):
    """Unified configuration for all Doraemon Code components."""

    # ========================================
    # Model Settings
    # ========================================
    model: str = Field(
        default="gemini-3-pro-preview",
        description="Default model to use"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Model temperature for response generation"
    )

    # ========================================
    # Context Settings
    # ========================================
    max_context_tokens: int = Field(
        default=100_000,
        gt=0,
        description="Maximum context window size in tokens"
    )
    summarize_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Trigger summarization when context reaches this fraction of max"
    )
    keep_recent_messages: int = Field(
        default=6,
        ge=1,
        description="Number of recent messages to always keep (not summarized)"
    )

    # ========================================
    # Tool Settings
    # ========================================
    max_tool_steps: int = Field(
        default=15,
        ge=1,
        description="Maximum number of tool execution steps per turn"
    )
    tool_timeout: float = Field(
        default=300.0,
        gt=0,
        description="Default timeout for tool execution in seconds"
    )
    enable_hitl: bool = Field(
        default=True,
        description="Enable Human-in-the-Loop approval for sensitive tools"
    )

    # ========================================
    # Checkpoint Settings
    # ========================================
    checkpoint_enabled: bool = Field(
        default=True,
        description="Enable automatic file checkpoints before modifications"
    )
    checkpoint_retention_days: int = Field(
        default=30,
        ge=1,
        description="Number of days to retain checkpoints"
    )
    max_checkpoints_per_file: int = Field(
        default=10,
        ge=1,
        description="Maximum number of checkpoints to keep per file"
    )

    # ========================================
    # Budget Settings
    # ========================================
    daily_budget_usd: float | None = Field(
        default=None,
        ge=0,
        description="Daily spending budget in USD (None = unlimited)"
    )
    session_budget_usd: float | None = Field(
        default=None,
        ge=0,
        description="Per-session spending budget in USD (None = unlimited)"
    )

    # ========================================
    # Logging Settings
    # ========================================
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    log_file: str | None = Field(
        default=None,
        description="Path to log file (None = console only)"
    )

    # ========================================
    # Performance Settings
    # ========================================
    enable_caching: bool = Field(
        default=True,
        description="Enable result caching for read operations"
    )
    cache_ttl_seconds: int = Field(
        default=300,
        ge=0,
        description="Cache time-to-live in seconds"
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is valid."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v_upper

    @classmethod
    def from_env_and_file(
        cls,
        config_path: str | Path | None = None,
        validate: bool = True
    ) -> "UnifiedConfig":
        """
        Load configuration from environment variables and config file.

        Precedence: Environment Variables > Config File > Defaults

        Args:
            config_path: Path to config file (default: .agent/config.json)
            validate: Whether to validate the configuration

        Returns:
            Validated UnifiedConfig instance

        Raises:
            FileNotFoundError: If config_path specified but doesn't exist
            ValueError: If configuration is invalid
        """
        # Default config path
        if config_path is None:
            config_path = default_config_path()
        else:
            config_path = Path(config_path)

        # Load from file if exists
        file_config = {}
        if config_path.exists():
            try:
                with open(config_path) as f:
                    file_config = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in config file: {e}") from e

        # Environment variable overrides
        env_overrides = {
            # Model settings
            "model": os.getenv("DORAEMON_MODEL"),
            "temperature": _parse_float(os.getenv("DORAEMON_TEMPERATURE")),

            # Context settings
            "max_context_tokens": _parse_int(os.getenv("DORAEMON_MAX_CONTEXT_TOKENS")),
            "summarize_threshold": _parse_float(os.getenv("DORAEMON_SUMMARIZE_THRESHOLD")),
            "keep_recent_messages": _parse_int(os.getenv("DORAEMON_KEEP_RECENT_MESSAGES")),

            # Tool settings
            "max_tool_steps": _parse_int(os.getenv("DORAEMON_MAX_TOOL_STEPS")),
            "tool_timeout": _parse_float(os.getenv("DORAEMON_TOOL_TIMEOUT")),
            "enable_hitl": _parse_bool(os.getenv("DORAEMON_ENABLE_HITL")),

            # Checkpoint settings
            "checkpoint_enabled": _parse_bool(os.getenv("DORAEMON_CHECKPOINT_ENABLED")),
            "checkpoint_retention_days": _parse_int(os.getenv("DORAEMON_CHECKPOINT_RETENTION_DAYS")),
            "max_checkpoints_per_file": _parse_int(os.getenv("DORAEMON_MAX_CHECKPOINTS_PER_FILE")),

            # Budget settings
            "daily_budget_usd": _parse_float(os.getenv("DORAEMON_DAILY_BUDGET")),
            "session_budget_usd": _parse_float(os.getenv("DORAEMON_SESSION_BUDGET")),

            # Logging settings
            "log_level": os.getenv("DORAEMON_LOG_LEVEL"),
            "log_file": os.getenv("DORAEMON_LOG_FILE"),

            # Performance settings
            "enable_caching": _parse_bool(os.getenv("DORAEMON_ENABLE_CACHING")),
            "cache_ttl_seconds": _parse_int(os.getenv("DORAEMON_CACHE_TTL")),
        }

        # Merge: env > file > defaults (filter out None values from env)
        config_dict = {
            **file_config,
            **{k: v for k, v in env_overrides.items() if v is not None}
        }

        # Create and validate
        if validate:
            return cls(**config_dict)
        else:
            return cls.model_construct(**config_dict)

    def to_file(self, config_path: str | Path) -> None:
        """
        Save configuration to file.

        Args:
            config_path: Path to save config file
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)


# ========================================
# Helper Functions
# ========================================

def _parse_int(value: str | None) -> int | None:
    """Parse integer from environment variable."""
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_float(value: str | None) -> float | None:
    """Parse float from environment variable."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_bool(value: str | None) -> bool | None:
    """Parse boolean from environment variable."""
    if value is None:
        return None
    value_lower = value.lower()
    if value_lower in ("true", "1", "yes", "on"):
        return True
    elif value_lower in ("false", "0", "no", "off"):
        return False
    return None
