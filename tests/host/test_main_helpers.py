"""Tests for main.py helper functions"""

from unittest.mock import MagicMock

from src.host.cli.chat_loop import build_system_prompt, convert_tools_to_definitions


def test_build_system_prompt_build_mode():
    """Test system prompt for build mode."""
    prompt = build_system_prompt("build")
    assert "build" in prompt.lower()
    assert len(prompt) > 100


def test_build_system_prompt_plan_mode():
    """Test system prompt for plan mode."""
    prompt = build_system_prompt("plan")
    assert "plan" in prompt.lower()
    assert len(prompt) > 100


def test_convert_tools_to_definitions():
    """Test converting tools to definitions."""
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.description = "A test"
    mock_tool.parameters = {"type": "object"}

    result = convert_tools_to_definitions([mock_tool])
    assert len(result) == 1
    assert result[0].name == "test_tool"


def test_convert_tools_empty_list():
    """Test converting empty tool list."""
    result = convert_tools_to_definitions([])
    assert result == []
