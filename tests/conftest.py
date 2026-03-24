"""
Pytest configuration and shared fixtures.

This file is automatically loaded by pytest and makes fixtures available to all tests.
"""

# Import all fixtures to make them available
from tests.utils.fixtures import (  # noqa: F401
    mock_model_client,
    reset_singletons,
    sample_messages,
    sample_tools,
    temp_config_file,
    temp_dir,
    test_agent_state,
    test_tool_registry,
)
