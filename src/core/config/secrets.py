"""
Secure secret resolution with keyring integration.

Priority order:
  1. Environment variable
  2. OS keyring
  3. Plaintext config.json fallback (deprecated)
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

SERVICE_NAME = "doraemon-code"

_ENV_OVERRIDES: dict[str, str] = {
    "google_api_key": "GOOGLE_API_KEY",
    "openai_api_key": "OPENAI_API_KEY",
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "gateway_key": "AGENT_API_KEY",
}

_SENSITIVE_HEADER_PATTERNS = frozenset(
    {
        "authorization",
        "api-key",
        "x-api-key",
        "context7_api_key",
    }
)


def _keyring_available() -> bool:
    try:
        import keyring

        backend = keyring.get_keyring()
        return getattr(backend, "priority", 0) > 0
    except ImportError:
        return False


def get_secret(key: str, config_value: str | None = None) -> str | None:
    env_name = _ENV_OVERRIDES.get(key)
    if env_name:
        env_val = os.environ.get(env_name)
        if env_val:
            logger.debug("Resolved '%s' from env var %s", key, env_name)
            return env_val

    if _keyring_available():
        try:
            import keyring

            val = keyring.get_password(SERVICE_NAME, key)
            if val is not None:
                logger.debug("Resolved '%s' from keyring", key)
                return val
        except Exception:
            logger.warning("Keyring lookup failed for '%s'; falling back", key, exc_info=True)

    if config_value:
        logger.warning(
            "Secret '%s' loaded from plaintext config.json -- migrate to keyring with: dora keyring set %s",
            key,
            key,
        )
        return config_value

    return None


def set_secret(key: str, value: str) -> None:
    import keyring

    keyring.set_password(SERVICE_NAME, key, value)
    logger.info("Stored '%s' in keyring", key)


def delete_secret(key: str) -> None:
    import keyring

    keyring.delete_password(SERVICE_NAME, key)
    logger.info("Deleted '%s' from keyring", key)


def is_sensitive_header(header_name: str) -> bool:
    return header_name.lower().replace("-", "_") in _SENSITIVE_HEADER_PATTERNS


def mcp_header_key(server_name: str, header_name: str) -> str:
    return f"mcp:{server_name}:header:{header_name}"


def resolve_mcp_headers(
    server_name: str,
    config_headers: dict[str, str],
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for hdr_name, hdr_value in config_headers.items():
        if is_sensitive_header(hdr_name):
            kr_key = mcp_header_key(server_name, hdr_name)
            secret = get_secret(kr_key, config_value=hdr_value)
            resolved[hdr_name] = secret or hdr_value
        else:
            resolved[hdr_name] = hdr_value
    return resolved
