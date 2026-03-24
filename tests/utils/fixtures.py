"""
Pytest Fixtures for Testing

Provides reusable fixtures for test setup.
"""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from src.core.context_manager import ContextManager
from src.core.model_client_base import BaseModelClient
from src.host.tools import ToolRegistry
from tests.utils.factories import (
    create_mock_model_client,
    create_test_context_manager,
    create_test_tool_registry,
)


@pytest.fixture
def mock_model_client() -> BaseModelClient:
    """Fixture for a mock model client with default responses."""
    return create_mock_model_client(["Test response 1", "Test response 2"])


@pytest.fixture
def test_context_manager() -> ContextManager:
    """Fixture for a test context manager."""
    return create_test_context_manager()


@pytest.fixture
def test_tool_registry() -> ToolRegistry:
    """Fixture for a test tool registry."""
    return create_test_tool_registry()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Fixture for a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_config_file(temp_dir: Path) -> Path:
    """Fixture for a temporary config file."""
    config_dir = temp_dir / ".agent"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.json"
    return config_file


@pytest.fixture
def sample_messages() -> list[dict]:
    """Fixture for sample message history."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help you?"},
        {"role": "user", "content": "What's the weather?"},
    ]


@pytest.fixture
def sample_tools() -> list[dict]:
    """Fixture for sample tool definitions."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name",
                        }
                    },
                    "required": ["location"],
                },
            },
        }
    ]


@pytest.fixture(autouse=True)
def reset_singletons():
    """
    Reset singleton instances before each test.

    This ensures tests don't interfere with each other.
    """
    # Reset tool registry singleton
    import src.host.tools as tools_module

    tools_module._default_registry = None

    yield

    # Cleanup after test
    tools_module._default_registry = None
