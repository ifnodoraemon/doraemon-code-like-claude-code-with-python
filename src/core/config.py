import json
import os
from pathlib import Path
from typing import Any

from .logger import get_logger
from .schema import get_default_config, validate_config_file

logger = get_logger(__name__)


def load_config(override_path: str | None = None, validate: bool = True) -> dict[str, Any]:
    """
    Cascading config loading with optional validation:
    1. Runtime override (if provided)
    2. Project specific: ./.doraemon/config.json
    3. User global: ~/.doraemon/config.json
    4. Package default: (Installed dir)/.doraemon/config.json

    Args:
        override_path: Optional path to override config file
        validate: Whether to validate configuration (default: True)

    Returns:
        Configuration dictionary

    Raises:
        ValueError: If validation is enabled and configuration is invalid
    """
    config_path = None
    config_data = None

    # 1. Runtime override
    if override_path and os.path.exists(override_path):
        config_path = Path(override_path)
        logger.info(f"Loading config from override: {config_path}")
    # 2. Project Level
    elif Path.cwd().joinpath(".doraemon", "config.json").exists():
        config_path = Path.cwd() / ".doraemon" / "config.json"
        logger.info(f"Loading config from project: {config_path}")
    # 3. User Level
    elif Path.home().joinpath(".doraemon", "config.json").exists():
        config_path = Path.home() / ".doraemon" / "config.json"
        logger.info(f"Loading config from user home: {config_path}")
    # 4. Package Default
    else:
        base_dir = Path(__file__).parent.parent.parent
        pkg_config = base_dir / ".doraemon" / "config.json"
        if pkg_config.exists():
            config_path = pkg_config
            logger.info(f"Loading config from package: {config_path}")
        else:
            logger.warning("No config file found, using defaults")
            return get_default_config()

    # Load and optionally validate
    try:
        if validate:
            validated_config = validate_config_file(config_path)
            config_data = validated_config.model_dump(by_alias=True)
            logger.info("Configuration validated successfully")
        else:
            with open(config_path) as f:
                config_data = json.load(f)
                logger.info("Configuration loaded (validation skipped)")

        return config_data

    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return get_default_config()
