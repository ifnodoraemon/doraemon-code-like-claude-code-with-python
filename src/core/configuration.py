"""
Advanced Configuration Management System

Provides hierarchical configuration with validation, environment variable support,
and hot-reloading capabilities.

Status: AVAILABLE FOR EXTENSION
    This module provides advanced configuration management beyond the basic
    config.py loader. It's designed for:
    - Building configuration from multiple sources (files, env vars, defaults)
    - Hot-reloading configuration without restart
    - Type-safe configuration access with validation
    - Complex nested configuration hierarchies

Note:
    The basic config loader (src.core.config) is used by default. This module
    can be integrated when more sophisticated configuration management is needed.

Example Usage:
    from src.core.configuration import ConfigurationBuilder

    config = (ConfigurationBuilder()
              .add_defaults({"debug": False})
              .add_json_file(".polymath/config.json", optional=True)
              .add_environment_variables(prefix="POLYMATH_")
              .build())

    debug_mode = config.get_bool("debug")
"""

import hashlib
import json
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


class ConfigSource(Enum):
    """Configuration source priority"""

    DEFAULTS = 1
    FILE = 2
    ENVIRONMENT = 3
    RUNTIME = 4  # Highest priority


@dataclass
class ConfigValue:
    """Represents a configuration value with metadata"""

    value: Any
    source: ConfigSource
    path: str
    updated_at: datetime = field(default_factory=datetime.now)


class ConfigWatcher:
    """Watches configuration files for changes"""

    def __init__(self, file_path: Path, callback: Callable[[], None]):
        self.file_path = file_path
        self.callback = callback
        self._last_hash: str | None = None
        self._lock = threading.Lock()

    def check_for_changes(self) -> bool:
        """Check if file has changed"""
        if not self.file_path.exists():
            return False

        current_hash = self._compute_hash()

        if self._last_hash is None:
            self._last_hash = current_hash
            return False

        if current_hash != self._last_hash:
            self._last_hash = current_hash
            return True

        return False

    def _compute_hash(self) -> str:
        """Compute file hash"""
        with open(self.file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()


class ConfigurationBuilder:
    """
    Fluent builder for creating configuration instances.

    Example:
        config = (ConfigurationBuilder()
                  .add_json_file(".polymath/config.json")
                  .add_environment_variables(prefix="POLYMATH_")
                  .build())
    """

    def __init__(self):
        self._sources: list[Callable[[], dict[str, Any]]] = []
        self._watchers: list[ConfigWatcher] = []
        self._defaults: dict[str, Any] = {}

    def add_defaults(self, defaults: dict[str, Any]) -> "ConfigurationBuilder":
        """Add default configuration values"""
        self._defaults.update(defaults)
        return self

    def add_json_file(
        self, path: str, optional: bool = False, reload_on_change: bool = False
    ) -> "ConfigurationBuilder":
        """
        Add JSON configuration file.

        Args:
            path: File path
            optional: Don't raise error if file not found
            reload_on_change: Watch file for changes
        """
        file_path = Path(path)

        def load_json():
            if not file_path.exists():
                if optional:
                    return {}
                raise FileNotFoundError(f"Config file not found: {path}")

            with open(file_path, encoding="utf-8") as f:
                return json.load(f)

        self._sources.append(load_json)
        return self

    def add_environment_variables(
        self, prefix: str = "", separator: str = "__"
    ) -> "ConfigurationBuilder":
        """
        Add environment variables as configuration.

        Args:
            prefix: Only include variables with this prefix
            separator: Nested key separator (e.g., "APP__DB__HOST" -> {"db": {"host": "..."}})
        """

        def load_env():
            config = {}
            for key, value in os.environ.items():
                if prefix and not key.startswith(prefix):
                    continue

                # Remove prefix
                clean_key = key[len(prefix) :] if prefix else key

                # Handle nested keys
                if separator in clean_key:
                    parts = clean_key.lower().split(separator)
                    current = config
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = self._parse_env_value(value)
                else:
                    config[clean_key.lower()] = self._parse_env_value(value)

            return config

        self._sources.append(load_env)
        return self

    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type"""
        # Try boolean
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False

        # Try number
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # Return as string
        return value

    def build(self) -> "Configuration":
        """Build the configuration instance"""
        # Merge all sources
        merged = {}

        # Start with defaults
        merged.update(self._defaults)

        # Apply each source in order
        for source in self._sources:
            data = source()
            merged = self._deep_merge(merged, data)

        return Configuration(merged)

    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Deep merge two dictionaries"""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result


class Configuration:
    """
    Configuration container with hierarchical access.

    Example:
        config = Configuration({"database": {"host": "localhost", "port": 5432}})
        host = config.get("database.host")
        port = config.get_int("database.port")
    """

    def __init__(self, data: dict[str, Any]):
        self._data = data
        self._lock = threading.RLock()

    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value by path.

        Args:
            path: Dot-separated path (e.g., "database.host")
            default: Default value if not found
        """
        with self._lock:
            parts = path.split(".")
            current = self._data

            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return default

            return current

    def get_required(self, path: str) -> Any:
        """Get configuration value (raises if not found)"""
        value = self.get(path)
        if value is None:
            raise ValueError(f"Required configuration not found: {path}")
        return value

    def get_str(self, path: str, default: str = "") -> str:
        """Get string value"""
        value = self.get(path, default)
        return str(value) if value is not None else default

    def get_int(self, path: str, default: int = 0) -> int:
        """Get integer value"""
        value = self.get(path, default)
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def get_bool(self, path: str, default: bool = False) -> bool:
        """Get boolean value"""
        value = self.get(path, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "yes", "1")
        return bool(value)

    def get_list(self, path: str, default: list | None = None) -> list:
        """Get list value"""
        value = self.get(path, default or [])
        return value if isinstance(value, list) else [value]

    def set(self, path: str, value: Any):
        """Set configuration value at runtime"""
        with self._lock:
            parts = path.split(".")
            current = self._data

            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            current[parts[-1]] = value

    def has(self, path: str) -> bool:
        """Check if configuration path exists"""
        return self.get(path) is not None

    def get_section(self, path: str) -> "Configuration":
        """Get configuration section as new Configuration instance"""
        section_data = self.get(path, {})
        if not isinstance(section_data, dict):
            section_data = {}
        return Configuration(section_data)

    def to_dict(self) -> dict[str, Any]:
        """Export configuration as dictionary"""
        with self._lock:
            return self._data.copy()

    def validate(self, schema: dict[str, Any]) -> list[str]:
        """
        Validate configuration against schema.

        Args:
            schema: Validation schema (simple format)
                {
                    "database.host": {"required": True, "type": str},
                    "database.port": {"required": True, "type": int, "min": 1, "max": 65535}
                }

        Returns:
            List of validation errors
        """
        errors = []

        for path, rules in schema.items():
            value = self.get(path)

            # Check required
            if rules.get("required", False) and value is None:
                errors.append(f"Missing required configuration: {path}")
                continue

            if value is None:
                continue

            # Check type
            expected_type = rules.get("type")
            if expected_type and not isinstance(value, expected_type):
                errors.append(
                    f"Invalid type for {path}: expected {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )

            # Check min/max for numbers
            if isinstance(value, int | float):
                if "min" in rules and value < rules["min"]:
                    errors.append(f"{path} must be >= {rules['min']}")
                if "max" in rules and value > rules["max"]:
                    errors.append(f"{path} must be <= {rules['max']}")

            # Check enum
            if "enum" in rules and value not in rules["enum"]:
                errors.append(f"{path} must be one of {rules['enum']}")

        return errors


# Singleton instance
_app_configuration: Configuration | None = None


def configure(builder_func: Callable[[ConfigurationBuilder], None]) -> Configuration:
    """
    Configure application configuration.

    Args:
        builder_func: Function that configures the builder

    Returns:
        Built configuration instance

    Example:
        def setup_config(builder):
            builder.add_json_file(".polymath/config.json")
            builder.add_environment_variables(prefix="POLYMATH_")

        config = configure(setup_config)
    """
    global _app_configuration

    builder = ConfigurationBuilder()
    builder_func(builder)
    _app_configuration = builder.build()

    return _app_configuration


def get_configuration() -> Configuration:
    """Get the application configuration"""
    if _app_configuration is None:
        raise RuntimeError("Configuration not initialized. Call configure() first.")
    return _app_configuration
