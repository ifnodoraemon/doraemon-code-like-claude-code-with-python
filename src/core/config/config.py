import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.core.logger import get_logger
from src.core.paths import config_path as default_config_path
from .schema import get_default_config, validate_config_file

logger = get_logger(__name__)

_CONFIG_CACHE: dict[tuple[str, bool], tuple[tuple[int, int] | None, dict[str, Any]]] = {}


def _get_config_signature(path: Path | None) -> tuple[int, int] | None:
    """Return a lightweight file signature for cache invalidation."""
    if path is None or not path.exists():
        return None

    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def load_config(override_path: str | None = None, validate: bool = True) -> dict[str, Any]:
    """
    Cascading config loading with optional validation:
    1. Runtime override (if provided)
    2. Project specific: ./.agent/config.json
    3. Built-in defaults

    Args:
        override_path: Optional path to override config file
        validate: Whether to validate configuration (default: True)

    Returns:
        Configuration dictionary

    Raises:
        ValueError: If validation is enabled and configuration is invalid
    """
    config_file = None
    config_data = None

    # 1. Runtime override
    if override_path and os.path.exists(override_path):
        config_file = Path(override_path)
        logger.info(f"Loading config from override: {config_file}")
    # 2. Project Level
    elif default_config_path().exists():
        config_file = default_config_path()
        logger.info(f"Loading config from project: {config_file}")
    else:
        logger.warning("No project config file found, using defaults")
        return get_default_config()

    cache_key = (str(config_file.resolve()), validate)
    signature = _get_config_signature(config_file)
    cached = _CONFIG_CACHE.get(cache_key)
    if cached and cached[0] == signature:
        logger.debug("Configuration cache hit")
        return deepcopy(cached[1])

    # Load and optionally validate
    try:
        if validate:
            validated_config = validate_config_file(config_file)
            config_data = validated_config.model_dump(by_alias=True)
            logger.info("Configuration validated successfully")
        else:
            with open(config_file) as f:
                config_data = json.load(f)
                logger.info("Configuration loaded (validation skipped)")

        _CONFIG_CACHE[cache_key] = (signature, config_data)
        return config_data

    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise
    except (PermissionError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load configuration: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}", exc_info=True)
        return get_default_config()


def get_required_config_value(key: str, override_path: str | None = None) -> Any:
    """Return a required top-level config value, raising if absent."""
    config_data = load_config(override_path=override_path)
    value = config_data.get(key)
    if value is None or value == "":
        config_file = override_path or str(default_config_path())
        raise ValueError(f"Missing required config '{key}' in {config_file}")
    return value


def get_optional_config_value(
    key: str,
    default: Any = None,
    override_path: str | None = None,
) -> Any:
    """Return an optional top-level config value."""
    config_data = load_config(override_path=override_path)
    return config_data.get(key, default)
