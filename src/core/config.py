import json
import os
from pathlib import Path
from typing import Any

from .logger import get_logger
from .paths import config_path as default_config_path
from .schema import get_default_config, validate_config_file

logger = get_logger(__name__)


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
