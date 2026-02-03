"""Comprehensive tests for src/host/cli/main.py

Tests cover:
- CLI initialization (parse_args, setup_logging, load_config)
- Main chat loop (run_chat_loop)
- User input handling
- Tool execution integration
- Context management
- Command routing
- Error handling
- Session persistence
- Signal handling (SIGINT)
- Mode switching
- Project isolation
"""

import asyncio
import json
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch, call, mock_open
from io import StringIO

from src.host.cli.chat_loop import (
    build_system_prompt,
    convert_tools_to_definitions,
    chat_loop,
)
from src.core.model_client import Message, ClientMode


class TestBuildSystemPrompt:
    """Test build_system_prompt function."""

    def test_build_system_prompt_build_mode(self):
        """Test system prompt generation for build mode."""
        with patch("src.host.cli.chat_loop.load_config") as mock_config, \
             patch("src.host.cli.chat_loop.get_system_prompt") as mock_sys_prompt, \
             patch("src.host.cli.chat_loop.load_all_instructions") as mock_instructions, \
             patch("src.host.cli.chat_loop.format_instructions_for_prompt") as mock_format:

            mock_config.return_value = {"persona": {"name": "TestBot"}}
            mock_sys_prompt.return_value = "Base prompt"
            mock_instructions.return_value = "Instructions"
            mock_format.return_value = "Formatted instructions"

            result = build_system_prompt("build")

            assert "Base prompt" in result
            mock_sys_prompt.assert_called_once_with("build", {"name": "TestBot"})

    def test_build_system_prompt_plan_mode(self):
        """Test system prompt generation for plan mode."""
        with patch("src.host.cli.chat_loop.load_config") as mock_config, \
             patch("src.host.cli.chat_loop.get_system_prompt") as mock_sys_prompt, \
             patch("src.host.cli.chat_loop.load_all_instructions") as mock_instructions, \
             patch("src.host.cli.chat_loop.format_instructions_for_prompt") as mock_format:

            mock_config.return_value = {"persona": {}}
            mock_sys_prompt.return_value = "Plan mode prompt"
            mock_instructions.return_value = ""

            result = build_system_prompt("plan")

            assert "Plan mode prompt" in result
            mock_sys_prompt.assert_called_once_with("plan", {})

    def test_build_system_prompt_with_skills(self):
        """Test system prompt includes skills content."""
        with patch("src.host.cli.chat_loop.load_config") as mock_config, \
             patch("src.host.cli.chat_loop.get_system_prompt") as mock_sys_prompt, \
             patch("src.host.cli.chat_loop.load_all_instructions") as mock_instructions, \
             patch("src.host.cli.chat_loop.format_instructions_for_prompt") as mock_format:

            mock_config.return_value = {"persona": {}}
            mock_sys_prompt.return_value = "Base"
            mock_instructions.return_value = ""

            skills_content = "Skill: Python coding"
            result = build_system_prompt("build", skills_content)

            assert "Skill: Python coding" in result

    def test_build_system_prompt_with_instructions(self):
        """Test system prompt includes project instructions."""
        with patch("src.host.cli.chat_loop.load_config") as mock_config, \
             patch("src.host.cli.chat_loop.get_system_prompt") as mock_sys_prompt, \
             patch("src.host.cli.chat_loop.load_all_instructions") as mock_instructions, \
             patch("src.host.cli.chat_loop.format_instructions_for_prompt") as mock_format:

            mock_config.return_value = {"persona": {}}
            mock_sys_prompt.return_value = "Base"
            mock_instructions.return_value = "Project rules"
            mock_format.return_value = "Formatted rules"

            result = build_system_prompt("build")

            assert "Formatted rules" in result
            mock_format.assert_called_once_with("Project rules")

    def test_build_system_prompt_empty_persona(self):
        """Test system prompt with missing persona config."""
        with patch("src.host.cli.chat_loop.load_config") as mock_config, \
             patch("src.host.cli.chat_loop.get_system_prompt") as mock_sys_prompt, \
             patch("src.host.cli.chat_loop.load_all_instructions") as mock_instructions, \
             patch("src.host.cli.chat_loop.format_instructions_for_prompt") as mock_format:

            mock_config.return_value = {}
            mock_sys_prompt.return_value = "Default prompt"
            mock_instructions.return_value = ""

            result = build_system_prompt("build")

            assert "Default prompt" in result
            mock_sys_prompt.assert_called_once_with("build", {})

    def test_build_system_prompt_with_all_components(self):
        """Test system prompt with all components combined."""
        with patch("src.host.cli.chat_loop.load_config") as mock_config, \
             patch("src.host.cli.chat_loop.get_system_prompt") as mock_sys_prompt, \
             patch("src.host.cli.chat_loop.load_all_instructions") as mock_instructions, \
             patch("src.host.cli.chat_loop.format_instructions_for_prompt") as mock_format:

            mock_config.return_value = {"persona": {"name": "Doraemon", "role": "Assistant"}}
            mock_sys_prompt.return_value = "System: "
            mock_instructions.return_value = "Rules"
            mock_format.return_value = "\n\nRules: "

            skills = "\n\nSkills: Python"
            result = build_system_prompt("build", skills)

            assert "System: " in result
            assert "Rules: " in result
            assert "Skills: Python" in result

    def test_build_system_prompt_returns_string(self):
        """Test that build_system_prompt returns a string."""
        with patch("src.host.cli.chat_loop.load_config") as mock_config, \
             patch("src.host.cli.chat_loop.get_system_prompt") as mock_sys_prompt, \
             patch("src.host.cli.chat_loop.load_all_instructions") as mock_instructions, \
             patch("src.host.cli.chat_loop.format_instructions_for_prompt") as mock_format:

            mock_config.return_value = {"persona": {}}
            mock_sys_prompt.return_value = "Prompt"
            mock_instructions.return_value = ""

            result = build_system_prompt("build")

            assert isinstance(result, str)
            assert len(result) > 0


class TestConvertToolsToDefinitions:
    """Test convert_tools_to_definitions function."""

    def test_convert_tools_with_function_declaration(self):
        """Test converting tools with FunctionDeclaration format."""
        mock_tool = MagicMock()
        mock_tool.name = "read_file"
        mock_tool.description = "Read a file"
        mock_tool.parameters = {"type": "object", "properties": {}}

        result = convert_tools_to_definitions([mock_tool])

        assert len(result) == 1
        assert result[0].name == "read_file"
        assert result[0].description == "Read a file"

    def test_convert_tools_with_dict_format(self):
        """Test converting tools with dict format."""
        tool_dict = {
            "name": "write_file",
            "description": "Write to a file",
            "parameters": {"type": "object"}
        }

        result = convert_tools_to_definitions([tool_dict])

        assert len(result) == 1
        assert result[0].name == "write_file"
        assert result[0].description == "Write to a file"

    def test_convert_tools_empty_list(self):
        """Test converting empty tool list."""
        result = convert_tools_to_definitions([])

        assert result == []

    def test_convert_tools_mixed_formats(self):
        """Test converting tools with mixed formats."""
        mock_tool = MagicMock()
        mock_tool.name = "tool1"
        mock_tool.description = "Tool 1"
        mock_tool.parameters = {}

        tool_dict = {
            "name": "tool2",
            "description": "Tool 2",
            "parameters": {}
        }

        result = convert_tools_to_definitions([mock_tool, tool_dict])

        assert len(result) == 2
        assert result[0].name == "tool1"
        assert result[1].name == "tool2"

    def test_convert_tools_missing_parameters(self):
        """Test converting tools with missing parameters."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test"
        mock_tool.parameters = None

        result = convert_tools_to_definitions([mock_tool])

        assert len(result) == 1
        assert result[0].parameters == {}

    def test_convert_tools_preserves_all_fields(self):
        """Test that all tool fields are preserved."""
        mock_tool = MagicMock()
        mock_tool.name = "complex_tool"
        mock_tool.description = "A complex tool"
        mock_tool.parameters = {
            "type": "object",
            "properties": {
                "param1": {"type": "string"},
                "param2": {"type": "number"}
            }
        }

        result = convert_tools_to_definitions([mock_tool])

        assert result[0].name == "complex_tool"
        assert result[0].description == "A complex tool"
        assert "param1" in result[0].parameters.get("properties", {})
        assert "param2" in result[0].parameters.get("properties", {})

    def test_convert_tools_multiple_tools(self):
        """Test converting multiple tools."""
        tools = []
        for i in range(5):
            mock_tool = MagicMock()
            mock_tool.name = f"tool_{i}"
            mock_tool.description = f"Tool {i}"
            mock_tool.parameters = {}
            tools.append(mock_tool)

        result = convert_tools_to_definitions(tools)

        assert len(result) == 5
        for i, tool_def in enumerate(result):
            assert tool_def.name == f"tool_{i}"


@pytest.mark.asyncio
class TestChatLoopInitialization:
    """Test chat_loop initialization and setup."""

    async def test_chat_loop_gateway_mode_initialization(self):
        """Test chat loop initializes correctly in gateway mode."""
        with patch("src.host.cli.chat_loop.sys.stdin.isatty", return_value=True), \
             patch("src.host.cli.chat_loop.ModelClient.get_mode") as mock_get_mode, \
             patch("src.host.cli.chat_loop.ModelClient.get_mode_info") as mock_mode_info, \
             patch("src.host.cli.chat_loop.ModelClient.create") as mock_create, \
             patch("src.host.cli.chat_loop.get_default_registry") as mock_registry, \
             patch("src.host.cli.chat_loop.ToolSelector") as mock_selector, \
             patch("src.host.cli.chat_loop.ContextManager") as mock_ctx, \
             patch("src.host.cli.chat_loop.SkillManager") as mock_skill, \
             patch("src.host.cli.chat_loop.CheckpointManager") as mock_checkpoint, \
             patch("src.host.cli.chat_loop.get_task_manager") as mock_task, \
             patch("src.host.cli.chat_loop.HookManager") as mock_hook, \
             patch("src.host.cli.chat_loop.CostTracker") as mock_cost, \
             patch("src.host.cli.chat_loop.CommandHistory") as mock_history, \
             patch("src.host.cli.chat_loop.BashModeExecutor") as mock_bash, \
             patch("src.host.cli.chat_loop.SessionManager") as mock_session, \
             patch("src.host.cli.chat_loop.console") as mock_console, \
             patch("src.host.cli.chat_loop.Prompt.ask") as mock_prompt:

            mock_get_mode.return_value = ClientMode.GATEWAY
            mock_mode_info.return_value = {"gateway_url": "http://localhost:8000"}
            mock_client = AsyncMock()
            mock_create.return_value = mock_client
            mock_registry_inst = MagicMock()
            mock_registry_inst.get_sensitive_tools.return_value = []
            mock_registry_inst.get_genai_tools.return_value = []
            mock_registry.return_value = mock_registry_inst

            mock_ctx_inst = MagicMock()
            mock_ctx_inst.session_id = "test-session"
            mock_ctx_inst.get_context_stats.return_value = {"messages": 0, "summaries": 0}
            mock_ctx_inst.get_history_for_api.return_value = []
            mock_ctx_inst.messages = []
            mock_ctx.return_value = mock_ctx_inst

            mock_selector_inst = MagicMock()
            mock_selector_inst.get_tools_for_mode.return_value = []
            mock_selector.return_value = mock_selector_inst

            mock_skill_inst = MagicMock()
            mock_skill_inst.get_skills_for_context.return_value = ""
            mock_skill.return_value = mock_skill_inst

            mock_checkpoint_inst = MagicMock()
            mock_checkpoint.return_value = mock_checkpoint_inst

            mock_task_inst = MagicMock()
            mock_task_inst.get_running_tasks.return_value = []
            mock_task.return_value = mock_task_inst

            mock_hook_inst = AsyncMock()
            mock_hook_inst.trigger = AsyncMock()
            mock_hook.return_value = mock_hook_inst

            mock_cost_inst = MagicMock()
            mock_cost_inst.check_budget.return_value = {}
            mock_cost.return_value = mock_cost_inst

            mock_history_inst = MagicMock()
            mock_history.return_value = mock_history_inst

            mock_bash_inst = MagicMock()
            mock_bash.return_value = mock_bash_inst

            mock_session_inst = MagicMock()
            mock_session.return_value = mock_session_inst

            mock_prompt.side_effect = KeyboardInterrupt()

            await chat_loop(project="test-project")

            mock_create.assert_called_once()
            mock_registry.assert_called_once()

    async def test_chat_loop_direct_mode_initialization(self):
        """Test chat loop initializes correctly in direct mode."""
        with patch("src.host.cli.chat_loop.sys.stdin.isatty", return_value=True), \
             patch("src.host.cli.chat_loop.ModelClient.get_mode") as mock_get_mode, \
             patch("src.host.cli.chat_loop.ModelClient.get_mode_info") as mock_mode_info, \
             patch("src.host.cli.chat_loop.ModelClient.create") as mock_create, \
             patch("src.host.cli.chat_loop.get_default_registry") as mock_registry, \
             patch("src.host.cli.chat_loop.ToolSelector") as mock_selector, \
             patch("src.host.cli.chat_loop.ContextManager") as mock_ctx, \
             patch("src.host.cli.chat_loop.SkillManager") as mock_skill, \
             patch("src.host.cli.chat_loop.CheckpointManager") as mock_checkpoint, \
             patch("src.host.cli.chat_loop.get_task_manager") as mock_task, \
             patch("src.host.cli.chat_loop.HookManager") as mock_hook, \
             patch("src.host.cli.chat_loop.CostTracker") as mock_cost, \
             patch("src.host.cli.chat_loop.CommandHistory") as mock_history, \
             patch("src.host.cli.chat_loop.BashModeExecutor") as mock_bash, \
             patch("src.host.cli.chat_loop.SessionManager") as mock_session, \
             patch("src.host.cli.chat_loop.console") as mock_console, \
             patch("src.host.cli.chat_loop.Prompt.ask") as mock_prompt:

            mock_get_mode.return_value = ClientMode.DIRECT
            mock_mode_info.return_value = {"providers": {"google": True, "openai": False}}
            mock_client = AsyncMock()
            mock_create.return_value = mock_client
            mock_registry_inst = MagicMock()
            mock_registry_inst.get_sensitive_tools.return_value = []
            mock_registry_inst.get_genai_tools.return_value = []
            mock_registry.return_value = mock_registry_inst

            mock_ctx_inst = MagicMock()
            mock_ctx_inst.session_id = "test-session"
            mock_ctx_inst.get_context_stats.return_value = {"messages": 0, "summaries": 0}
            mock_ctx_inst.get_history_for_api.return_value = []
            mock_ctx_inst.messages = []
            mock_ctx.return_value = mock_ctx_inst

            mock_selector_inst = MagicMock()
            mock_selector_inst.get_tools_for_mode.return_value = []
            mock_selector.return_value = mock_selector_inst

            mock_skill_inst = MagicMock()
            mock_skill_inst.get_skills_for_context.return_value = ""
            mock_skill.return_value = mock_skill_inst

            mock_checkpoint_inst = MagicMock()
            mock_checkpoint.return_value = mock_checkpoint_inst

            mock_task_inst = MagicMock()
            mock_task_inst.get_running_tasks.return_value = []
            mock_task.return_value = mock_task_inst

            mock_hook_inst = AsyncMock()
            mock_hook_inst.trigger = AsyncMock()
            mock_hook.return_value = mock_hook_inst

            mock_cost_inst = MagicMock()
            mock_cost_inst.check_budget.return_value = {}
            mock_cost.return_value = mock_cost_inst

            mock_history_inst = MagicMock()
            mock_history.return_value = mock_history_inst

            mock_bash_inst = MagicMock()
            mock_bash.return_value = mock_bash_inst

            mock_session_inst = MagicMock()
            mock_session.return_value = mock_session_inst

            mock_prompt.side_effect = KeyboardInterrupt()

            await chat_loop(project="test-project")

            mock_create.assert_called_once()

    async def test_chat_loop_no_api_keys_error(self):
        """Test chat loop exits when no API keys configured."""
        with patch("src.host.cli.chat_loop.sys.stdin.isatty", return_value=True), \
             patch("src.host.cli.chat_loop.ModelClient.get_mode") as mock_get_mode, \
             patch("src.host.cli.chat_loop.ModelClient.get_mode_info") as mock_mode_info, \
             patch("src.host.cli.chat_loop.console") as mock_console:

            mock_get_mode.return_value = ClientMode.DIRECT
            mock_mode_info.return_value = {"providers": {"google": False, "openai": False}}

            await chat_loop(project="test-project")

            mock_console.print.assert_called()
            calls = [str(call) for call in mock_console.print.call_args_list]
            assert any("No API keys" in str(call) for call in calls)

    async def test_chat_loop_gateway_url_missing_error(self):
        """Test chat loop exits when gateway URL not configured."""
        with patch("src.host.cli.chat_loop.sys.stdin.isatty", return_value=True), \
             patch("src.host.cli.chat_loop.ModelClient.get_mode") as mock_get_mode, \
             patch("src.host.cli.chat_loop.ModelClient.get_mode_info") as mock_mode_info, \
             patch("src.host.cli.chat_loop.console") as mock_console:

            mock_get_mode.return_value = ClientMode.GATEWAY
            mock_mode_info.return_value = {"gateway_url": None}

            await chat_loop(project="test-project")

            mock_console.print.assert_called()
            calls = [str(call) for call in mock_console.print.call_args_list]
            assert any("DORAEMON_GATEWAY_URL" in str(call) for call in calls)

    async def test_chat_loop_model_client_creation_failure(self):
        """Test chat loop handles model client creation failure."""
        with patch("src.host.cli.chat_loop.sys.stdin.isatty", return_value=True), \
             patch("src.host.cli.chat_loop.ModelClient.get_mode") as mock_get_mode, \
             patch("src.host.cli.chat_loop.ModelClient.get_mode_info") as mock_mode_info, \
             patch("src.host.cli.chat_loop.ModelClient.create") as mock_create, \
             patch("src.host.cli.chat_loop.console") as mock_console:

            mock_get_mode.return_value = ClientMode.DIRECT
            mock_mode_info.return_value = {"providers": {"google": True}}
            mock_create.side_effect = Exception("Connection failed")

            await chat_loop(project="test-project")

            mock_console.print.assert_called()
            calls = [str(call) for call in mock_console.print.call_args_list]
            assert any("Failed to initialize" in str(call) for call in calls)

    async def test_chat_loop_with_project_isolation(self):
        """Test chat loop respects project isolation."""
        with patch("src.host.cli.chat_loop.sys.stdin.isatty", return_value=True), \
             patch("src.host.cli.chat_loop.ModelClient.get_mode") as mock_get_mode, \
             patch("src.host.cli.chat_loop.ModelClient.get_mode_info") as mock_mode_info, \
             patch("src.host.cli.chat_loop.ModelClient.create") as mock_create, \
             patch("src.host.cli.chat_loop.get_default_registry") as mock_registry, \
             patch("src.host.cli.chat_loop.ToolSelector") as mock_selector, \
             patch("src.host.cli.chat_loop.ContextManager") as mock_ctx, \
             patch("src.host.cli.chat_loop.SkillManager") as mock_skill, \
             patch("src.host.cli.chat_loop.CheckpointManager") as mock_checkpoint, \
             patch("src.host.cli.chat_loop.get_task_manager") as mock_task, \
             patch("src.host.cli.chat_loop.HookManager") as mock_hook, \
             patch("src.host.cli.chat_loop.CostTracker") as mock_cost, \
             patch("src.host.cli.chat_loop.CommandHistory") as mock_history, \
             patch("src.host.cli.chat_loop.BashModeExecutor") as mock_bash, \
             patch("src.host.cli.chat_loop.SessionManager") as mock_session, \
             patch("src.host.cli.chat_loop.console") as mock_console, \
             patch("src.host.cli.chat_loop.Prompt.ask") as mock_prompt:

            mock_get_mode.return_value = ClientMode.DIRECT
            mock_mode_info.return_value = {"providers": {"google": True}}
            mock_client = AsyncMock()
            mock_create.return_value = mock_client
            mock_registry_inst = MagicMock()
            mock_registry_inst.get_sensitive_tools.return_value = []
            mock_registry_inst.get_genai_tools.return_value = []
            mock_registry.return_value = mock_registry_inst

            mock_ctx_inst = MagicMock()
            mock_ctx_inst.session_id = "test-session"
            mock_ctx_inst.get_context_stats.return_value = {"messages": 0, "summaries": 0}
            mock_ctx_inst.get_history_for_api.return_value = []
            mock_ctx_inst.messages = []
            mock_ctx.return_value = mock_ctx_inst

            mock_selector_inst = MagicMock()
            mock_selector_inst.get_tools_for_mode.return_value = []
            mock_selector.return_value = mock_selector_inst

            mock_skill_inst = MagicMock()
            mock_skill_inst.get_skills_for_context.return_value = ""
            mock_skill.return_value = mock_skill_inst

            mock_checkpoint_inst = MagicMock()
            mock_checkpoint.return_value = mock_checkpoint_inst

            mock_task_inst = MagicMock()
            mock_task_inst.get_running_tasks.return_value = []
            mock_task.return_value = mock_task_inst

            mock_hook_inst = AsyncMock()
            mock_hook_inst.trigger = AsyncMock()
            mock_hook.return_value = mock_hook_inst

            mock_cost_inst = MagicMock()
            mock_cost_inst.check_budget.return_value = {}
            mock_cost.return_value = mock_cost_inst

            mock_history_inst = MagicMock()
            mock_history.return_value = mock_history_inst

            mock_bash_inst = MagicMock()
            mock_bash.return_value = mock_bash_inst

            mock_session_inst = MagicMock()
            mock_session.return_value = mock_session_inst

            mock_prompt.side_effect = KeyboardInterrupt()

            await chat_loop(project="my-project")

            mock_ctx.assert_called_once()
            call_kwargs = mock_ctx.call_args[1]
            assert call_kwargs.get("project") == "my-project"

    async def test_chat_loop_headless_mode_with_prompt(self):
        """Test chat loop enters headless mode when prompt provided."""
        with patch("src.host.cli.chat_loop.sys.stdin.isatty", return_value=True), \
             patch("src.host.cli.chat_loop.ModelClient.get_mode") as mock_get_mode, \
             patch("src.host.cli.chat_loop.ModelClient.get_mode_info") as mock_mode_info, \
             patch("src.host.cli.chat_loop.ModelClient.create") as mock_create, \
             patch("src.host.cli.chat_loop.get_default_registry") as mock_registry, \
             patch("src.host.cli.chat_loop.ToolSelector") as mock_selector, \
             patch("src.host.cli.chat_loop.ContextManager") as mock_ctx, \
             patch("src.host.cli.chat_loop.SkillManager") as mock_skill, \
             patch("src.host.cli.chat_loop.CheckpointManager") as mock_checkpoint, \
             patch("src.host.cli.chat_loop.get_task_manager") as mock_task, \
             patch("src.host.cli.chat_loop.HookManager") as mock_hook, \
             patch("src.host.cli.chat_loop.CostTracker") as mock_cost, \
             patch("src.host.cli.chat_loop.CommandHistory") as mock_history, \
             patch("src.host.cli.chat_loop.BashModeExecutor") as mock_bash, \
             patch("src.host.cli.chat_loop.SessionManager") as mock_session, \
             patch("src.host.cli.chat_loop.console") as mock_console:

            mock_get_mode.return_value = ClientMode.DIRECT
            mock_mode_info.return_value = {"providers": {"google": True}}
            mock_client = AsyncMock()
            mock_create.return_value = mock_client
            mock_registry_inst = MagicMock()
            mock_registry_inst.get_sensitive_tools.return_value = []
            mock_registry_inst.get_genai_tools.return_value = []
            mock_registry.return_value = mock_registry_inst

            mock_ctx_inst = MagicMock()
            mock_ctx_inst.session_id = "test-session"
            mock_ctx_inst.get_context_stats.return_value = {"messages": 0, "summaries": 0}
            mock_ctx_inst.get_history_for_api.return_value = []
            mock_ctx_inst.messages = []
            mock_ctx.return_value = mock_ctx_inst

            mock_selector_inst = MagicMock()
            mock_selector_inst.get_tools_for_mode.return_value = []
            mock_selector.return_value = mock_selector_inst

            mock_skill_inst = MagicMock()
            mock_skill_inst.get_skills_for_context.return_value = ""
            mock_skill.return_value = mock_skill_inst

            mock_checkpoint_inst = MagicMock()
            mock_checkpoint.return_value = mock_checkpoint_inst

            mock_task_inst = MagicMock()
            mock_task_inst.get_running_tasks.return_value = []
            mock_task.return_value = mock_task_inst

            mock_hook_inst = AsyncMock()
            mock_hook_inst.trigger = AsyncMock()
            mock_hook.return_value = mock_hook_inst

            mock_cost_inst = MagicMock()
            mock_cost_inst.check_budget.return_value = {}
            mock_cost.return_value = mock_cost_inst

            mock_history_inst = MagicMock()
            mock_history.return_value = mock_history_inst

            mock_bash_inst = MagicMock()
            mock_bash.return_value = mock_bash_inst

            mock_session_inst = MagicMock()
            mock_session.return_value = mock_session_inst

            await chat_loop(project="test", prompt="Hello")

            mock_console.print.assert_called()
            calls = [str(call) for call in mock_console.print.call_args_list]
            assert any("headless" in str(call).lower() for call in calls)


@pytest.mark.asyncio
class TestChatLoopUserInputHandling:
    """Test user input handling in chat loop."""

    async def test_chat_loop_exit_command(self):
        """Test chat loop exits on exit command."""
        with patch("src.host.cli.chat_loop.sys.stdin.isatty", return_value=True), \
             patch("src.host.cli.chat_loop.ModelClient.get_mode") as mock_get_mode, \
             patch("src.host.cli.chat_loop.ModelClient.get_mode_info") as mock_mode_info, \
             patch("src.host.cli.chat_loop.ModelClient.create") as mock_create, \
             patch("src.host.cli.chat_loop.get_default_registry") as mock_registry, \
             patch("src.host.cli.chat_loop.ToolSelector") as mock_selector, \
             patch("src.host.cli.chat_loop.ContextManager") as mock_ctx, \
             patch("src.host.cli.chat_loop.SkillManager") as mock_skill, \
             patch("src.host.cli.chat_loop.CheckpointManager") as mock_checkpoint, \
             patch("src.host.cli.chat_loop.get_task_manager") as mock_task, \
             patch("src.host.cli.chat_loop.HookManager") as mock_hook, \
             patch("src.host.cli.chat_loop.CostTracker") as mock_cost, \
             patch("src.host.cli.chat_loop.CommandHistory") as mock_history, \
             patch("src.host.cli.chat_loop.BashModeExecutor") as mock_bash, \
             patch("src.host.cli.chat_loop.SessionManager") as mock_session, \
             patch("src.host.cli.chat_loop.console") as mock_console, \
             patch("src.host.cli.chat_loop.Prompt.ask") as mock_prompt:

            mock_get_mode.return_value = ClientMode.DIRECT
            mock_mode_info.return_value = {"providers": {"google": True}}
            mock_client = AsyncMock()
            mock_create.return_value = mock_client
            mock_registry_inst = MagicMock()
            mock_registry_inst.get_sensitive_tools.return_value = []
            mock_registry_inst.get_genai_tools.return_value = []
            mock_registry.return_value = mock_registry_inst

            mock_ctx_inst = MagicMock()
            mock_ctx_inst.session_id = "test-session"
            mock_ctx_inst.get_context_stats.return_value = {"messages": 0, "summaries": 0}
            mock_ctx_inst.get_history_for_api.return_value = []
            mock_ctx_inst.messages = []
            mock_ctx.return_value = mock_ctx_inst

            mock_selector_inst = MagicMock()
            mock_selector_inst.get_tools_for_mode.return_value = []
            mock_selector.return_value = mock_selector_inst

            mock_skill_inst = MagicMock()
            mock_skill_inst.get_skills_for_context.return_value = ""
            mock_skill.return_value = mock_skill_inst

            mock_checkpoint_inst = MagicMock()
            mock_checkpoint.return_value = mock_checkpoint_inst

            mock_task_inst = MagicMock()
            mock_task_inst.get_running_tasks.return_value = []
            mock_task.return_value = mock_task_inst

            mock_hook_inst = AsyncMock()
            mock_hook_inst.trigger = AsyncMock()
            mock_hook.return_value = mock_hook_inst

            mock_cost_inst = MagicMock()
            mock_cost_inst.check_budget.return_value = {}
            mock_cost.return_value = mock_cost_inst

            mock_history_inst = MagicMock()
            mock_history.return_value = mock_history_inst

            mock_bash_inst = MagicMock()
            mock_bash.return_value = mock_bash_inst

            mock_session_inst = MagicMock()
            mock_session.return_value = mock_session_inst

            mock_prompt.return_value = "exit"

            await chat_loop(project="test")

            mock_hook_inst.trigger.assert_called()

    async def test_chat_loop_quit_command(self):
        """Test chat loop exits on quit command."""
        with patch("src.host.cli.chat_loop.sys.stdin.isatty", return_value=True), \
             patch("src.host.cli.chat_loop.ModelClient.get_mode") as mock_get_mode, \
             patch("src.host.cli.chat_loop.ModelClient.get_mode_info") as mock_mode_info, \
             patch("src.host.cli.chat_loop.ModelClient.create") as mock_create, \
             patch("src.host.cli.chat_loop.get_default_registry") as mock_registry, \
             patch("src.host.cli.chat_loop.ToolSelector") as mock_selector, \
             patch("src.host.cli.chat_loop.ContextManager") as mock_ctx, \
             patch("src.host.cli.chat_loop.SkillManager") as mock_skill, \
             patch("src.host.cli.chat_loop.CheckpointManager") as mock_checkpoint, \
             patch("src.host.cli.chat_loop.get_task_manager") as mock_task, \
             patch("src.host.cli.chat_loop.HookManager") as mock_hook, \
             patch("src.host.cli.chat_loop.CostTracker") as mock_cost, \
             patch("src.host.cli.chat_loop.CommandHistory") as mock_history, \
             patch("src.host.cli.chat_loop.BashModeExecutor") as mock_bash, \
             patch("src.host.cli.chat_loop.SessionManager") as mock_session, \
             patch("src.host.cli.chat_loop.console") as mock_console, \
             patch("src.host.cli.chat_loop.Prompt.ask") as mock_prompt:

            mock_get_mode.return_value = ClientMode.DIRECT
            mock_mode_info.return_value = {"providers": {"google": True}}
            mock_client = AsyncMock()
            mock_create.return_value = mock_client
            mock_registry_inst = MagicMock()
            mock_registry_inst.get_sensitive_tools.return_value = []
            mock_registry_inst.get_genai_tools.return_value = []
            mock_registry.return_value = mock_registry_inst

            mock_ctx_inst = MagicMock()
            mock_ctx_inst.session_id = "test-session"
            mock_ctx_inst.get_context_stats.return_value = {"messages": 0, "summaries": 0}
            mock_ctx_inst.get_history_for_api.return_value = []
            mock_ctx_inst.messages = []
            mock_ctx.return_value = mock_ctx_inst

            mock_selector_inst = MagicMock()
            mock_selector_inst.get_tools_for_mode.return_value = []
            mock_selector.return_value = mock_selector_inst

            mock_skill_inst = MagicMock()
            mock_skill_inst.get_skills_for_context.return_value = ""
            mock_skill.return_value = mock_skill_inst

            mock_checkpoint_inst = MagicMock()
            mock_checkpoint.return_value = mock_checkpoint_inst

            mock_task_inst = MagicMock()
            mock_task_inst.get_running_tasks.return_value = []
            mock_task.return_value = mock_task_inst

            mock_hook_inst = AsyncMock()
            mock_hook_inst.trigger = AsyncMock()
            mock_hook.return_value = mock_hook_inst

            mock_cost_inst = MagicMock()
            mock_cost_inst.check_budget.return_value = {}
            mock_cost.return_value = mock_cost_inst

            mock_history_inst = MagicMock()
            mock_history.return_value = mock_history_inst

            mock_bash_inst = MagicMock()
            mock_bash.return_value = mock_bash_inst

            mock_session_inst = MagicMock()
            mock_session.return_value = mock_session_inst

            mock_prompt.return_value = "quit"

            await chat_loop(project="test")

            mock_client.close.assert_called_once()

    async def test_chat_loop_bash_mode_execution(self):
        """Test bash mode (! prefix) command execution."""
        with patch("src.host.cli.chat_loop.sys.stdin.isatty", return_value=True), \
             patch("src.host.cli.chat_loop.ModelClient.get_mode") as mock_get_mode, \
             patch("src.host.cli.chat_loop.ModelClient.get_mode_info") as mock_mode_info, \
             patch("src.host.cli.chat_loop.ModelClient.create") as mock_create, \
             patch("src.host.cli.chat_loop.get_default_registry") as mock_registry, \
             patch("src.host.cli.chat_loop.ToolSelector") as mock_selector, \
             patch("src.host.cli.chat_loop.ContextManager") as mock_ctx, \
             patch("src.host.cli.chat_loop.SkillManager") as mock_skill, \
             patch("src.host.cli.chat_loop.CheckpointManager") as mock_checkpoint, \
             patch("src.host.cli.chat_loop.get_task_manager") as mock_task, \
             patch("src.host.cli.chat_loop.HookManager") as mock_hook, \
             patch("src.host.cli.chat_loop.CostTracker") as mock_cost, \
             patch("src.host.cli.chat_loop.CommandHistory") as mock_history, \
             patch("src.host.cli.chat_loop.BashModeExecutor") as mock_bash, \
             patch("src.host.cli.chat_loop.SessionManager") as mock_session, \
             patch("src.host.cli.chat_loop.console") as mock_console, \
             patch("src.host.cli.chat_loop.Prompt.ask") as mock_prompt:

            mock_get_mode.return_value = ClientMode.DIRECT
            mock_mode_info.return_value = {"providers": {"google": True}}
            mock_client = AsyncMock()
            mock_create.return_value = mock_client
            mock_registry_inst = MagicMock()
            mock_registry_inst.get_sensitive_tools.return_value = []
            mock_registry_inst.get_genai_tools.return_value = []
            mock_registry.return_value = mock_registry_inst

            mock_ctx_inst = MagicMock()
            mock_ctx_inst.session_id = "test-session"
            mock_ctx_inst.get_context_stats.return_value = {"messages": 0, "summaries": 0}
            mock_ctx_inst.get_history_for_api.return_value = []
            mock_ctx_inst.messages = []
            mock_ctx.return_value = mock_ctx_inst

            mock_selector_inst = MagicMock()
            mock_selector_inst.get_tools_for_mode.return_value = []
            mock_selector.return_value = mock_selector_inst

            mock_skill_inst = MagicMock()
            mock_skill_inst.get_skills_for_context.return_value = ""
            mock_skill.return_value = mock_skill_inst

            mock_checkpoint_inst = MagicMock()
            mock_checkpoint.return_value = mock_checkpoint_inst

            mock_task_inst = MagicMock()
            mock_task_inst.get_running_tasks.return_value = []
            mock_task.return_value = mock_task_inst

            mock_hook_inst = AsyncMock()
            mock_hook_inst.trigger = AsyncMock()
            mock_hook.return_value = mock_hook_inst

            mock_cost_inst = MagicMock()
            mock_cost_inst.check_budget.return_value = {}
            mock_cost.return_value = mock_cost_inst

            mock_history_inst = MagicMock()
            mock_history.return_value = mock_history_inst

            mock_bash_inst = MagicMock()
            mock_bash_inst.execute.return_value = {"output": "result", "error": ""}
            mock_bash_inst.execute_for_context.return_value = "Command executed"
            mock_bash.return_value = mock_bash_inst

            mock_session_inst = MagicMock()
            mock_session.return_value = mock_session_inst

            mock_prompt.side_effect = ["! ls -la", "exit"]

            await chat_loop(project="test")

            mock_bash_inst.execute.assert_called()
            mock_history_inst.add.assert_called()

    async def test_chat_loop_slash_command_routing(self):
        """Test slash command routing to CommandHandler."""
        with patch("src.host.cli.chat_loop.sys.stdin.isatty", return_value=True), \
             patch("src.host.cli.chat_loop.ModelClient.get_mode") as mock_get_mode, \
             patch("src.host.cli.chat_loop.ModelClient.get_mode_info") as mock_mode_info, \
             patch("src.host.cli.chat_loop.ModelClient.create") as mock_create, \
             patch("src.host.cli.chat_loop.get_default_registry") as mock_registry, \
             patch("src.host.cli.chat_loop.ToolSelector") as mock_selector, \
             patch("src.host.cli.chat_loop.ContextManager") as mock_ctx, \
             patch("src.host.cli.chat_loop.SkillManager") as mock_skill, \
             patch("src.host.cli.chat_loop.CheckpointManager") as mock_checkpoint, \
             patch("src.host.cli.chat_loop.get_task_manager") as mock_task, \
             patch("src.host.cli.chat_loop.HookManager") as mock_hook, \
             patch("src.host.cli.chat_loop.CostTracker") as mock_cost, \
             patch("src.host.cli.chat_loop.CommandHistory") as mock_history, \
             patch("src.host.cli.chat_loop.BashModeExecutor") as mock_bash, \
             patch("src.host.cli.chat_loop.SessionManager") as mock_session, \
             patch("src.host.cli.chat_loop.CommandHandler") as mock_cmd_handler, \
             patch("src.host.cli.chat_loop.console") as mock_console, \
             patch("src.host.cli.chat_loop.Prompt.ask") as mock_prompt:

            mock_get_mode.return_value = ClientMode.DIRECT
            mock_mode_info.return_value = {"providers": {"google": True}}
            mock_client = AsyncMock()
            mock_create.return_value = mock_client
            mock_registry_inst = MagicMock()
            mock_registry_inst.get_sensitive_tools.return_value = []
            mock_registry_inst.get_genai_tools.return_value = []
            mock_registry.return_value = mock_registry_inst

            mock_ctx_inst = MagicMock()
            mock_ctx_inst.session_id = "test-session"
            mock_ctx_inst.get_context_stats.return_value = {"messages": 0, "summaries": 0}
            mock_ctx_inst.get_history_for_api.return_value = []
            mock_ctx_inst.messages = []
            mock_ctx.return_value = mock_ctx_inst

            mock_selector_inst = MagicMock()
            mock_selector_inst.get_tools_for_mode.return_value = []
            mock_selector.return_value = mock_selector_inst

            mock_skill_inst = MagicMock()
            mock_skill_inst.get_skills_for_context.return_value = ""
            mock_skill.return_value = mock_skill_inst

            mock_checkpoint_inst = MagicMock()
            mock_checkpoint.return_value = mock_checkpoint_inst

            mock_task_inst = MagicMock()
            mock_task_inst.get_running_tasks.return_value = []
            mock_task.return_value = mock_task_inst

            mock_hook_inst = AsyncMock()
            mock_hook_inst.trigger = AsyncMock()
            mock_hook.return_value = mock_hook_inst

            mock_cost_inst = MagicMock()
            mock_cost_inst.check_budget.return_value = {}
            mock_cost.return_value = mock_cost_inst

            mock_history_inst = MagicMock()
            mock_history.return_value = mock_history_inst

            mock_bash_inst = MagicMock()
            mock_bash.return_value = mock_bash_inst

            mock_session_inst = MagicMock()
            mock_session.return_value = mock_session_inst

            mock_handler_inst = AsyncMock()
            mock_handler_inst.handle = AsyncMock(return_value={
                "mode": "build",
                "tool_names": [],
                "tool_definitions": [],
                "active_skills_content": "",
                "conversation_history": [],
                "system_prompt": None,
            })
            mock_cmd_handler.return_value = mock_handler_inst

            mock_prompt.side_effect = ["/help", "exit"]

            await chat_loop(project="test")

            mock_handler_inst.handle.assert_called()

    async def test_chat_loop_keyboard_interrupt_handling(self):
        """Test chat loop handles KeyboardInterrupt gracefully."""
        with patch("src.host.cli.chat_loop.sys.stdin.isatty", return_value=True), \
             patch("src.host.cli.chat_loop.ModelClient.get_mode") as mock_get_mode, \
             patch("src.host.cli.chat_loop.ModelClient.get_mode_info") as mock_mode_info, \
             patch("src.host.cli.chat_loop.ModelClient.create") as mock_create, \
             patch("src.host.cli.chat_loop.get_default_registry") as mock_registry, \
             patch("src.host.cli.chat_loop.ToolSelector") as mock_selector, \
             patch("src.host.cli.chat_loop.ContextManager") as mock_ctx, \
             patch("src.host.cli.chat_loop.SkillManager") as mock_skill, \
             patch("src.host.cli.chat_loop.CheckpointManager") as mock_checkpoint, \
             patch("src.host.cli.chat_loop.get_task_manager") as mock_task, \
             patch("src.host.cli.chat_loop.HookManager") as mock_hook, \
             patch("src.host.cli.chat_loop.CostTracker") as mock_cost, \
             patch("src.host.cli.chat_loop.CommandHistory") as mock_history, \
             patch("src.host.cli.chat_loop.BashModeExecutor") as mock_bash, \
             patch("src.host.cli.chat_loop.SessionManager") as mock_session, \
             patch("src.host.cli.chat_loop.console") as mock_console, \
             patch("src.host.cli.chat_loop.Prompt.ask") as mock_prompt:

            mock_get_mode.return_value = ClientMode.DIRECT
            mock_mode_info.return_value = {"providers": {"google": True}}
            mock_client = AsyncMock()
            mock_create.return_value = mock_client
            mock_registry_inst = MagicMock()
            mock_registry_inst.get_sensitive_tools.return_value = []
            mock_registry_inst.get_genai_tools.return_value = []
            mock_registry.return_value = mock_registry_inst

            mock_ctx_inst = MagicMock()
            mock_ctx_inst.session_id = "test-session"
            mock_ctx_inst.get_context_stats.return_value = {"messages": 0, "summaries": 0}
            mock_ctx_inst.get_history_for_api.return_value = []
            mock_ctx_inst.messages = []
            mock_ctx.return_value = mock_ctx_inst

            mock_selector_inst = MagicMock()
            mock_selector_inst.get_tools_for_mode.return_value = []
            mock_selector.return_value = mock_selector_inst

            mock_skill_inst = MagicMock()
            mock_skill_inst.get_skills_for_context.return_value = ""
            mock_skill.return_value = mock_skill_inst

            mock_checkpoint_inst = MagicMock()
            mock_checkpoint.return_value = mock_checkpoint_inst

            mock_task_inst = MagicMock()
            mock_task_inst.get_running_tasks.return_value = []
            mock_task.return_value = mock_task_inst

            mock_hook_inst = AsyncMock()
            mock_hook_inst.trigger = AsyncMock()
            mock_hook.return_value = mock_hook_inst

            mock_cost_inst = MagicMock()
            mock_cost_inst.check_budget.return_value = {}
            mock_cost.return_value = mock_cost_inst

            mock_history_inst = MagicMock()
            mock_history.return_value = mock_history_inst

            mock_bash_inst = MagicMock()
            mock_bash.return_value = mock_bash_inst

            mock_session_inst = MagicMock()
            mock_session.return_value = mock_session_inst

            mock_prompt.side_effect = KeyboardInterrupt()

            await chat_loop(project="test")

            mock_console.print.assert_called()
            mock_client.close.assert_called_once()

    async def test_chat_loop_exception_handling(self):
        """Test chat loop handles exceptions gracefully."""
        with patch("src.host.cli.chat_loop.sys.stdin.isatty", return_value=True), \
             patch("src.host.cli.chat_loop.ModelClient.get_mode") as mock_get_mode, \
             patch("src.host.cli.chat_loop.ModelClient.get_mode_info") as mock_mode_info, \
             patch("src.host.cli.chat_loop.ModelClient.create") as mock_create, \
             patch("src.host.cli.chat_loop.get_default_registry") as mock_registry, \
             patch("src.host.cli.chat_loop.ToolSelector") as mock_selector, \
             patch("src.host.cli.chat_loop.ContextManager") as mock_ctx, \
             patch("src.host.cli.chat_loop.SkillManager") as mock_skill, \
             patch("src.host.cli.chat_loop.CheckpointManager") as mock_checkpoint, \
             patch("src.host.cli.chat_loop.get_task_manager") as mock_task, \
             patch("src.host.cli.chat_loop.HookManager") as mock_hook, \
             patch("src.host.cli.chat_loop.CostTracker") as mock_cost, \
             patch("src.host.cli.chat_loop.CommandHistory") as mock_history, \
             patch("src.host.cli.chat_loop.BashModeExecutor") as mock_bash, \
             patch("src.host.cli.chat_loop.SessionManager") as mock_session, \
             patch("src.host.cli.chat_loop.console") as mock_console, \
             patch("src.host.cli.chat_loop.Prompt.ask") as mock_prompt:

            mock_get_mode.return_value = ClientMode.DIRECT
            mock_mode_info.return_value = {"providers": {"google": True}}
            mock_client = AsyncMock()
            mock_create.return_value = mock_client
            mock_registry_inst = MagicMock()
            mock_registry_inst.get_sensitive_tools.return_value = []
            mock_registry_inst.get_genai_tools.return_value = []
            mock_registry.return_value = mock_registry_inst

            mock_ctx_inst = MagicMock()
            mock_ctx_inst.session_id = "test-session"
            mock_ctx_inst.get_context_stats.return_value = {"messages": 0, "summaries": 0}
            mock_ctx_inst.get_history_for_api.return_value = []
            mock_ctx_inst.messages = []
            mock_ctx.return_value = mock_ctx_inst

            mock_selector_inst = MagicMock()
            mock_selector_inst.get_tools_for_mode.return_value = []
            mock_selector.return_value = mock_selector_inst

            mock_skill_inst = MagicMock()
            mock_skill_inst.get_skills_for_context.return_value = ""
            mock_skill.return_value = mock_skill_inst

            mock_checkpoint_inst = MagicMock()
            mock_checkpoint.return_value = mock_checkpoint_inst

            mock_task_inst = MagicMock()
            mock_task_inst.get_running_tasks.return_value = []
            mock_task.return_value = mock_task_inst

            mock_hook_inst = AsyncMock()
            mock_hook_inst.trigger = AsyncMock(side_effect=Exception("Hook error"))
            mock_hook.return_value = mock_hook_inst

            mock_cost_inst = MagicMock()
            mock_cost_inst.check_budget.return_value = {}
            mock_cost.return_value = mock_cost_inst

            mock_history_inst = MagicMock()
            mock_history.return_value = mock_history_inst

            mock_bash_inst = MagicMock()
            mock_bash.return_value = mock_bash_inst

            mock_session_inst = MagicMock()
            mock_session.return_value = mock_session_inst

            mock_prompt.side_effect = KeyboardInterrupt()

            await chat_loop(project="test")

            mock_client.close.assert_called_once()

    async def test_chat_loop_session_resume(self):
        """Test chat loop resumes previous session."""
        with patch("src.host.cli.chat_loop.sys.stdin.isatty", return_value=True), \
             patch("src.host.cli.chat_loop.ModelClient.get_mode") as mock_get_mode, \
             patch("src.host.cli.chat_loop.ModelClient.get_mode_info") as mock_mode_info, \
             patch("src.host.cli.chat_loop.ModelClient.create") as mock_create, \
             patch("src.host.cli.chat_loop.get_default_registry") as mock_registry, \
             patch("src.host.cli.chat_loop.ToolSelector") as mock_selector, \
             patch("src.host.cli.chat_loop.ContextManager") as mock_ctx, \
             patch("src.host.cli.chat_loop.SkillManager") as mock_skill, \
             patch("src.host.cli.chat_loop.CheckpointManager") as mock_checkpoint, \
             patch("src.host.cli.chat_loop.get_task_manager") as mock_task, \
             patch("src.host.cli.chat_loop.HookManager") as mock_hook, \
             patch("src.host.cli.chat_loop.CostTracker") as mock_cost, \
             patch("src.host.cli.chat_loop.CommandHistory") as mock_history, \
             patch("src.host.cli.chat_loop.BashModeExecutor") as mock_bash, \
             patch("src.host.cli.chat_loop.SessionManager") as mock_session, \
             patch("src.host.cli.chat_loop.console") as mock_console, \
             patch("src.host.cli.chat_loop.Prompt.ask") as mock_prompt:

            mock_get_mode.return_value = ClientMode.DIRECT
            mock_mode_info.return_value = {"providers": {"google": True}}
            mock_client = AsyncMock()
            mock_create.return_value = mock_client
            mock_registry_inst = MagicMock()
            mock_registry_inst.get_sensitive_tools.return_value = []
            mock_registry_inst.get_genai_tools.return_value = []
            mock_registry.return_value = mock_registry_inst

            mock_ctx_inst = MagicMock()
            mock_ctx_inst.session_id = "test-session"
            mock_ctx_inst.get_context_stats.return_value = {"messages": 0, "summaries": 0}
            mock_ctx_inst.get_history_for_api.return_value = []
            mock_ctx_inst.messages = []
            mock_ctx.return_value = mock_ctx_inst

            mock_selector_inst = MagicMock()
            mock_selector_inst.get_tools_for_mode.return_value = []
            mock_selector.return_value = mock_selector_inst

            mock_skill_inst = MagicMock()
            mock_skill_inst.get_skills_for_context.return_value = ""
            mock_skill.return_value = mock_skill_inst

            mock_checkpoint_inst = MagicMock()
            mock_checkpoint.return_value = mock_checkpoint_inst

            mock_task_inst = MagicMock()
            mock_task_inst.get_running_tasks.return_value = []
            mock_task.return_value = mock_task_inst

            mock_hook_inst = AsyncMock()
            mock_hook_inst.trigger = AsyncMock()
            mock_hook.return_value = mock_hook_inst

            mock_cost_inst = MagicMock()
            mock_cost_inst.check_budget.return_value = {}
            mock_cost.return_value = mock_cost_inst

            mock_history_inst = MagicMock()
            mock_history.return_value = mock_history_inst

            mock_bash_inst = MagicMock()
            mock_bash.return_value = mock_bash_inst

            mock_session_inst = MagicMock()
            mock_session_data = MagicMock()
            mock_session_data.metadata.get_display_name.return_value = "Previous Session"
            mock_session_data.messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"}
            ]
            mock_session_inst.resume_session.return_value = mock_session_data
            mock_session.return_value = mock_session_inst

            mock_prompt.side_effect = KeyboardInterrupt()

            await chat_loop(project="test", resume_session="session-123")

            mock_session_inst.resume_session.assert_called_once_with("session-123")
            mock_ctx_inst.add_user_message.assert_called()
            mock_ctx_inst.add_assistant_message.assert_called()


@pytest.mark.asyncio
class TestChatLoopToolExecution:
    """Test tool execution in chat loop."""

    async def test_chat_loop_tool_execution_success(self):
        """Test successful tool execution."""
        with patch("src.host.cli.chat_loop.sys.stdin.isatty", return_value=True), \
             patch("src.host.cli.chat_loop.ModelClient.get_mode") as mock_get_mode, \
             patch("src.host.cli.chat_loop.ModelClient.get_mode_info") as mock_mode_info, \
             patch("src.host.cli.chat_loop.ModelClient.create") as mock_create, \
             patch("src.host.cli.chat_loop.get_default_registry") as mock_registry, \
             patch("src.host.cli.chat_loop.ToolSelector") as mock_selector, \
             patch("src.host.cli.chat_loop.ContextManager") as mock_ctx, \
             patch("src.host.cli.chat_loop.SkillManager") as mock_skill, \
             patch("src.host.cli.chat_loop.CheckpointManager") as mock_checkpoint, \
             patch("src.host.cli.chat_loop.get_task_manager") as mock_task, \
             patch("src.host.cli.chat_loop.HookManager") as mock_hook, \
             patch("src.host.cli.chat_loop.CostTracker") as mock_cost, \
             patch("src.host.cli.chat_loop.CommandHistory") as mock_history, \
             patch("src.host.cli.chat_loop.BashModeExecutor") as mock_bash, \
             patch("src.host.cli.chat_loop.SessionManager") as mock_session, \
             patch("src.host.cli.chat_loop.console") as mock_console, \
             patch("src.host.cli.chat_loop.Prompt.ask") as mock_prompt:

            mock_get_mode.return_value = ClientMode.DIRECT
            mock_mode_info.return_value = {"providers": {"google": True}}
            mock_client = AsyncMock()

            # Mock response with tool call
            mock_response = MagicMock()
            mock_response.content = "I'll read the file"
            mock_response.thought = None
            mock_response.tool_calls = [{
                "id": "call_123",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path": "test.py"}'
                }
            }]
            mock_response.has_tool_calls = True
            mock_response.usage = {"prompt_tokens": 100, "completion_tokens": 50}

            # Second response without tool calls
            mock_response2 = MagicMock()
            mock_response2.content = "File content: test"
            mock_response2.thought = None
            mock_response2.tool_calls = []
            mock_response2.has_tool_calls = False
            mock_response2.usage = {"prompt_tokens": 100, "completion_tokens": 50}

            mock_client.chat = AsyncMock(side_effect=[mock_response, mock_response2])
            mock_create.return_value = mock_client

            mock_registry_inst = MagicMock()
            mock_registry_inst.get_sensitive_tools.return_value = []
            mock_registry_inst.get_genai_tools.return_value = []
            mock_registry_inst.call_tool = AsyncMock(return_value="File content")
            mock_registry.return_value = mock_registry_inst

            mock_ctx_inst = MagicMock()
            mock_ctx_inst.session_id = "test-session"
            mock_ctx_inst.get_context_stats.return_value = {"messages": 0, "summaries": 0, "usage_percent": 50}
            mock_ctx_inst.get_history_for_api.return_value = []
            mock_ctx_inst.messages = []
            mock_ctx_inst.summaries = []
            mock_ctx.return_value = mock_ctx_inst

            mock_selector_inst = MagicMock()
            mock_selector_inst.get_tools_for_mode.return_value = ["read_file"]
            mock_selector.return_value = mock_selector_inst

            mock_skill_inst = MagicMock()
            mock_skill_inst.get_skills_for_context.return_value = ""
            mock_skill.return_value = mock_skill_inst

            mock_checkpoint_inst = MagicMock()
            mock_checkpoint.return_value = mock_checkpoint_inst

            mock_task_inst = MagicMock()
            mock_task_inst.get_running_tasks.return_value = []
            mock_task.return_value = mock_task_inst

            mock_hook_inst = AsyncMock()
            mock_hook_inst.trigger = AsyncMock(return_value=MagicMock(continue_processing=True, reason=None, decision=MagicMock(value="allow"), modified_input=None))
            mock_hook.return_value = mock_hook_inst

            mock_cost_inst = MagicMock()
            mock_cost_inst.check_budget.return_value = {}
            mock_cost_inst.calculate_cost.return_value = 0.01
            mock_cost.return_value = mock_cost_inst

            mock_history_inst = MagicMock()
            mock_history.return_value = mock_history_inst

            mock_bash_inst = MagicMock()
            mock_bash.return_value = mock_bash_inst

            mock_session_inst = MagicMock()
            mock_session.return_value = mock_session_inst

            mock_prompt.side_effect = ["read test.py", "exit"]

            await chat_loop(project="test")

            mock_registry_inst.call_tool.assert_called()

    async def test_chat_loop_tool_execution_failure(self):
        """Test tool execution failure handling."""
        with patch("src.host.cli.chat_loop.sys.stdin.isatty", return_value=True), \
             patch("src.host.cli.chat_loop.ModelClient.get_mode") as mock_get_mode, \
             patch("src.host.cli.chat_loop.ModelClient.get_mode_info") as mock_mode_info, \
             patch("src.host.cli.chat_loop.ModelClient.create") as mock_create, \
             patch("src.host.cli.chat_loop.get_default_registry") as mock_registry, \
             patch("src.host.cli.chat_loop.ToolSelector") as mock_selector, \
             patch("src.host.cli.chat_loop.ContextManager") as mock_ctx, \
             patch("src.host.cli.chat_loop.SkillManager") as mock_skill, \
             patch("src.host.cli.chat_loop.CheckpointManager") as mock_checkpoint, \
             patch("src.host.cli.chat_loop.get_task_manager") as mock_task, \
             patch("src.host.cli.chat_loop.HookManager") as mock_hook, \
             patch("src.host.cli.chat_loop.CostTracker") as mock_cost, \
             patch("src.host.cli.chat_loop.CommandHistory") as mock_history, \
             patch("src.host.cli.chat_loop.BashModeExecutor") as mock_bash, \
             patch("src.host.cli.chat_loop.SessionManager") as mock_session, \
             patch("src.host.cli.chat_loop.console") as mock_console, \
             patch("src.host.cli.chat_loop.Prompt.ask") as mock_prompt:

            mock_get_mode.return_value = ClientMode.DIRECT
            mock_mode_info.return_value = {"providers": {"google": True}}
            mock_client = AsyncMock()

            mock_response = MagicMock()
            mock_response.content = "I'll read the file"
            mock_response.thought = None
            mock_response.tool_calls = [{
                "id": "call_123",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path": "nonexistent.py"}'
                }
            }]
            mock_response.has_tool_calls = True
            mock_response.usage = {"prompt_tokens": 100, "completion_tokens": 50}

            mock_response2 = MagicMock()
            mock_response2.content = "Error occurred"
            mock_response2.thought = None
            mock_response2.tool_calls = []
            mock_response2.has_tool_calls = False
            mock_response2.usage = {"prompt_tokens": 100, "completion_tokens": 50}

            mock_client.chat = AsyncMock(side_effect=[mock_response, mock_response2])
            mock_create.return_value = mock_client

            mock_registry_inst = MagicMock()
            mock_registry_inst.get_sensitive_tools.return_value = []
            mock_registry_inst.get_genai_tools.return_value = []
            mock_registry_inst.call_tool = AsyncMock(side_effect=Exception("File not found"))
            mock_registry.return_value = mock_registry_inst

            mock_ctx_inst = MagicMock()
            mock_ctx_inst.session_id = "test-session"
            mock_ctx_inst.get_context_stats.return_value = {"messages": 0, "summaries": 0, "usage_percent": 50}
            mock_ctx_inst.get_history_for_api.return_value = []
            mock_ctx_inst.messages = []
            mock_ctx_inst.summaries = []
            mock_ctx.return_value = mock_ctx_inst

            mock_selector_inst = MagicMock()
            mock_selector_inst.get_tools_for_mode.return_value = ["read_file"]
            mock_selector.return_value = mock_selector_inst

            mock_skill_inst = MagicMock()
            mock_skill_inst.get_skills_for_context.return_value = ""
            mock_skill.return_value = mock_skill_inst

            mock_checkpoint_inst = MagicMock()
            mock_checkpoint.return_value = mock_checkpoint_inst

            mock_task_inst = MagicMock()
            mock_task_inst.get_running_tasks.return_value = []
            mock_task.return_value = mock_task_inst

            mock_hook_inst = AsyncMock()
            mock_hook_inst.trigger = AsyncMock(return_value=MagicMock(continue_processing=True, reason=None, decision=MagicMock(value="allow"), modified_input=None))
            mock_hook.return_value = mock_hook_inst

            mock_cost_inst = MagicMock()
            mock_cost_inst.check_budget.return_value = {}
            mock_cost_inst.calculate_cost.return_value = 0.01
            mock_cost.return_value = mock_cost_inst

            mock_history_inst = MagicMock()
            mock_history.return_value = mock_history_inst

            mock_bash_inst = MagicMock()
            mock_bash.return_value = mock_bash_inst

            mock_session_inst = MagicMock()
            mock_session.return_value = mock_session_inst

            mock_prompt.side_effect = ["read nonexistent.py", "exit"]

            await chat_loop(project="test")

            mock_registry_inst.call_tool.assert_called()

    async def test_chat_loop_sensitive_tool_approval(self):
        """Test sensitive tool requires user approval."""
        with patch("src.host.cli.chat_loop.sys.stdin.isatty", return_value=True), \
             patch("src.host.cli.chat_loop.ModelClient.get_mode") as mock_get_mode, \
             patch("src.host.cli.chat_loop.ModelClient.get_mode_info") as mock_mode_info, \
             patch("src.host.cli.chat_loop.ModelClient.create") as mock_create, \
             patch("src.host.cli.chat_loop.get_default_registry") as mock_registry, \
             patch("src.host.cli.chat_loop.ToolSelector") as mock_selector, \
             patch("src.host.cli.chat_loop.ContextManager") as mock_ctx, \
             patch("src.host.cli.chat_loop.SkillManager") as mock_skill, \
             patch("src.host.cli.chat_loop.CheckpointManager") as mock_checkpoint, \
             patch("src.host.cli.chat_loop.get_task_manager") as mock_task, \
             patch("src.host.cli.chat_loop.HookManager") as mock_hook, \
             patch("src.host.cli.chat_loop.CostTracker") as mock_cost, \
             patch("src.host.cli.chat_loop.CommandHistory") as mock_history, \
             patch("src.host.cli.chat_loop.BashModeExecutor") as mock_bash, \
             patch("src.host.cli.chat_loop.SessionManager") as mock_session, \
             patch("src.host.cli.chat_loop.console") as mock_console, \
             patch("src.host.cli.chat_loop.Prompt.ask") as mock_prompt:

            mock_get_mode.return_value = ClientMode.DIRECT
            mock_mode_info.return_value = {"providers": {"google": True}}
            mock_client = AsyncMock()

            mock_response = MagicMock()
            mock_response.content = "I'll write the file"
            mock_response.thought = None
            mock_response.tool_calls = [{
                "id": "call_123",
                "function": {
                    "name": "write_file",
                    "arguments": '{"path": "test.py", "content": "print(1)"}'
                }
            }]
            mock_response.has_tool_calls = True
            mock_response.usage = {"prompt_tokens": 100, "completion_tokens": 50}

            mock_response2 = MagicMock()
            mock_response2.content = "File written"
            mock_response2.thought = None
            mock_response2.tool_calls = []
            mock_response2.has_tool_calls = False
            mock_response2.usage = {"prompt_tokens": 100, "completion_tokens": 50}

            mock_client.chat = AsyncMock(side_effect=[mock_response, mock_response2])
            mock_create.return_value = mock_client

            mock_registry_inst = MagicMock()
            mock_registry_inst.get_sensitive_tools.return_value = ["write_file"]
            mock_registry_inst.get_genai_tools.return_value = []
            mock_registry_inst.call_tool = AsyncMock(return_value="Written")
            mock_registry.return_value = mock_registry_inst

            mock_ctx_inst = MagicMock()
            mock_ctx_inst.session_id = "test-session"
            mock_ctx_inst.get_context_stats.return_value = {"messages": 0, "summaries": 0, "usage_percent": 50}
            mock_ctx_inst.get_history_for_api.return_value = []
            mock_ctx_inst.messages = []
            mock_ctx_inst.summaries = []
            mock_ctx.return_value = mock_ctx_inst

            mock_selector_inst = MagicMock()
            mock_selector_inst.get_tools_for_mode.return_value = ["write_file"]
            mock_selector.return_value = mock_selector_inst

            mock_skill_inst = MagicMock()
            mock_skill_inst.get_skills_for_context.return_value = ""
            mock_skill.return_value = mock_skill_inst

            mock_checkpoint_inst = MagicMock()
            mock_checkpoint.return_value = mock_checkpoint_inst

            mock_task_inst = MagicMock()
            mock_task_inst.get_running_tasks.return_value = []
            mock_task.return_value = mock_task_inst

            mock_hook_inst = AsyncMock()
            mock_hook_inst.trigger = AsyncMock(return_value=MagicMock(continue_processing=True, reason=None, decision=MagicMock(value="allow"), modified_input=None))
            mock_hook.return_value = mock_hook_inst

            mock_cost_inst = MagicMock()
            mock_cost_inst.check_budget.return_value = {}
            mock_cost_inst.calculate_cost.return_value = 0.01
            mock_cost.return_value = mock_cost_inst

            mock_history_inst = MagicMock()
            mock_history.return_value = mock_history_inst

            mock_bash_inst = MagicMock()
            mock_bash.return_value = mock_bash_inst

            mock_session_inst = MagicMock()
            mock_session.return_value = mock_session_inst

            mock_prompt.side_effect = ["write test.py", "y", "exit"]

            await chat_loop(project="test")

            mock_registry_inst.call_tool.assert_called()

    async def test_chat_loop_loop_detection(self):
        """Test detection of infinite tool loops."""
        with patch("src.host.cli.chat_loop.sys.stdin.isatty", return_value=True), \
             patch("src.host.cli.chat_loop.ModelClient.get_mode") as mock_get_mode, \
             patch("src.host.cli.chat_loop.ModelClient.get_mode_info") as mock_mode_info, \
             patch("src.host.cli.chat_loop.ModelClient.create") as mock_create, \
             patch("src.host.cli.chat_loop.get_default_registry") as mock_registry, \
             patch("src.host.cli.chat_loop.ToolSelector") as mock_selector, \
             patch("src.host.cli.chat_loop.ContextManager") as mock_ctx, \
             patch("src.host.cli.chat_loop.SkillManager") as mock_skill, \
             patch("src.host.cli.chat_loop.CheckpointManager") as mock_checkpoint, \
             patch("src.host.cli.chat_loop.get_task_manager") as mock_task, \
             patch("src.host.cli.chat_loop.HookManager") as mock_hook, \
             patch("src.host.cli.chat_loop.CostTracker") as mock_cost, \
             patch("src.host.cli.chat_loop.CommandHistory") as mock_history, \
             patch("src.host.cli.chat_loop.BashModeExecutor") as mock_bash, \
             patch("src.host.cli.chat_loop.SessionManager") as mock_session, \
             patch("src.host.cli.chat_loop.console") as mock_console, \
             patch("src.host.cli.chat_loop.Prompt.ask") as mock_prompt:

            mock_get_mode.return_value = ClientMode.DIRECT
            mock_mode_info.return_value = {"providers": {"google": True}}
            mock_client = AsyncMock()

            # Create responses that repeat the same tool call
            responses = []
            for i in range(20):
                mock_response = MagicMock()
                mock_response.content = f"Attempt {i}"
                mock_response.thought = None
                mock_response.tool_calls = [{
                    "id": f"call_{i}",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "test.py"}'
                    }
                }]
                mock_response.has_tool_calls = True
                mock_response.usage = {"prompt_tokens": 100, "completion_tokens": 50}
                responses.append(mock_response)

            mock_client.chat = AsyncMock(side_effect=responses)
            mock_create.return_value = mock_client

            mock_registry_inst = MagicMock()
            mock_registry_inst.get_sensitive_tools.return_value = []
            mock_registry_inst.get_genai_tools.return_value = []
            mock_registry_inst.call_tool = AsyncMock(return_value="File content")
            mock_registry.return_value = mock_registry_inst

            mock_ctx_inst = MagicMock()
            mock_ctx_inst.session_id = "test-session"
            mock_ctx_inst.get_context_stats.return_value = {"messages": 0, "summaries": 0, "usage_percent": 50}
            mock_ctx_inst.get_history_for_api.return_value = []
            mock_ctx_inst.messages = []
            mock_ctx_inst.summaries = []
            mock_ctx.return_value = mock_ctx_inst

            mock_selector_inst = MagicMock()
            mock_selector_inst.get_tools_for_mode.return_value = ["read_file"]
            mock_selector.return_value = mock_selector_inst

            mock_skill_inst = MagicMock()
            mock_skill_inst.get_skills_for_context.return_value = ""
            mock_skill.return_value = mock_skill_inst

            mock_checkpoint_inst = MagicMock()
            mock_checkpoint.return_value = mock_checkpoint_inst

            mock_task_inst = MagicMock()
            mock_task_inst.get_running_tasks.return_value = []
            mock_task.return_value = mock_task_inst

            mock_hook_inst = AsyncMock()
            mock_hook_inst.trigger = AsyncMock(return_value=MagicMock(continue_processing=True, reason=None, decision=MagicMock(value="allow"), modified_input=None))
            mock_hook.return_value = mock_hook_inst

            mock_cost_inst = MagicMock()
            mock_cost_inst.check_budget.return_value = {}
            mock_cost_inst.calculate_cost.return_value = 0.01
            mock_cost.return_value = mock_cost_inst

            mock_history_inst = MagicMock()
            mock_history.return_value = mock_history_inst

            mock_bash_inst = MagicMock()
            mock_bash.return_value = mock_bash_inst

            mock_session_inst = MagicMock()
            mock_session.return_value = mock_session_inst

            mock_prompt.side_effect = ["read test.py", "exit"]

            await chat_loop(project="test")

            mock_console.print.assert_called()

    async def test_chat_loop_max_tool_steps_limit(self):
        """Test max tool steps limit prevents infinite loops."""
        with patch("src.host.cli.chat_loop.sys.stdin.isatty", return_value=True), \
             patch("src.host.cli.chat_loop.ModelClient.get_mode") as mock_get_mode, \
             patch("src.host.cli.chat_loop.ModelClient.get_mode_info") as mock_mode_info, \
             patch("src.host.cli.chat_loop.ModelClient.create") as mock_create, \
             patch("src.host.cli.chat_loop.get_default_registry") as mock_registry, \
             patch("src.host.cli.chat_loop.ToolSelector") as mock_selector, \
             patch("src.host.cli.chat_loop.ContextManager") as mock_ctx, \
             patch("src.host.cli.chat_loop.SkillManager") as mock_skill, \
             patch("src.host.cli.chat_loop.CheckpointManager") as mock_checkpoint, \
             patch("src.host.cli.chat_loop.get_task_manager") as mock_task, \
             patch("src.host.cli.chat_loop.HookManager") as mock_hook, \
             patch("src.host.cli.chat_loop.CostTracker") as mock_cost, \
             patch("src.host.cli.chat_loop.CommandHistory") as mock_history, \
             patch("src.host.cli.chat_loop.BashModeExecutor") as mock_bash, \
             patch("src.host.cli.chat_loop.SessionManager") as mock_session, \
             patch("src.host.cli.chat_loop.console") as mock_console, \
             patch("src.host.cli.chat_loop.Prompt.ask") as mock_prompt:

            mock_get_mode.return_value = ClientMode.DIRECT
            mock_mode_info.return_value = {"providers": {"google": True}}
            mock_client = AsyncMock()

            # Create many responses with tool calls
            responses = []
            for i in range(30):
                mock_response = MagicMock()
                mock_response.content = f"Step {i}"
                mock_response.thought = None
                mock_response.tool_calls = [{
                    "id": f"call_{i}",
                    "function": {
                        "name": "read_file",
                        "arguments": f'{{"path": "file_{i}.py"}}'
                    }
                }]
                mock_response.has_tool_calls = True
                mock_response.usage = {"prompt_tokens": 100, "completion_tokens": 50}
                responses.append(mock_response)

            mock_client.chat = AsyncMock(side_effect=responses)
            mock_create.return_value = mock_client

            mock_registry_inst = MagicMock()
            mock_registry_inst.get_sensitive_tools.return_value = []
            mock_registry_inst.get_genai_tools.return_value = []
            mock_registry_inst.call_tool = AsyncMock(return_value="Result")
            mock_registry.return_value = mock_registry_inst

            mock_ctx_inst = MagicMock()
            mock_ctx_inst.session_id = "test-session"
            mock_ctx_inst.get_context_stats.return_value = {"messages": 0, "summaries": 0, "usage_percent": 50}
            mock_ctx_inst.get_history_for_api.return_value = []
            mock_ctx_inst.messages = []
            mock_ctx_inst.summaries = []
            mock_ctx.return_value = mock_ctx_inst

            mock_selector_inst = MagicMock()
            mock_selector_inst.get_tools_for_mode.return_value = ["read_file"]
            mock_selector.return_value = mock_selector_inst

            mock_skill_inst = MagicMock()
            mock_skill_inst.get_skills_for_context.return_value = ""
            mock_skill.return_value = mock_skill_inst

            mock_checkpoint_inst = MagicMock()
            mock_checkpoint.return_value = mock_checkpoint_inst

            mock_task_inst = MagicMock()
            mock_task_inst.get_running_tasks.return_value = []
            mock_task.return_value = mock_task_inst

            mock_hook_inst = AsyncMock()
            mock_hook_inst.trigger = AsyncMock(return_value=MagicMock(continue_processing=True, reason=None, decision=MagicMock(value="allow"), modified_input=None))
            mock_hook.return_value = mock_hook_inst

            mock_cost_inst = MagicMock()
            mock_cost_inst.check_budget.return_value = {}
            mock_cost_inst.calculate_cost.return_value = 0.01
            mock_cost.return_value = mock_cost_inst

            mock_history_inst = MagicMock()
            mock_history.return_value = mock_history_inst

            mock_bash_inst = MagicMock()
            mock_bash.return_value = mock_bash_inst

            mock_session_inst = MagicMock()
            mock_session.return_value = mock_session_inst

            mock_prompt.side_effect = ["read many files", "exit"]

            await chat_loop(project="test")

            mock_console.print.assert_called()
            calls = [str(call) for call in mock_console.print.call_args_list]
            assert any("Max tool steps" in str(call) for call in calls)


@pytest.mark.asyncio
class TestChatLoopContextManagement:
    """Test context management in chat loop."""

    async def test_chat_loop_context_summarization(self):
        """Test context summarization when threshold reached."""
        with patch("src.host.cli.chat_loop.sys.stdin.isatty", return_value=True), \
             patch("src.host.cli.chat_loop.ModelClient.get_mode") as mock_get_mode, \
             patch("src.host.cli.chat_loop.ModelClient.get_mode_info") as mock_mode_info, \
             patch("src.host.cli.chat_loop.ModelClient.create") as mock_create, \
             patch("src.host.cli.chat_loop.get_default_registry") as mock_registry, \
             patch("src.host.cli.chat_loop.ToolSelector") as mock_selector, \
             patch("src.host.cli.chat_loop.ContextManager") as mock_ctx, \
             patch("src.host.cli.chat_loop.SkillManager") as mock_skill, \
             patch("src.host.cli.chat_loop.CheckpointManager") as mock_checkpoint, \
             patch("src.host.cli.chat_loop.get_task_manager") as mock_task, \
             patch("src.host.cli.chat_loop.HookManager") as mock_hook, \
             patch("src.host.cli.chat_loop.CostTracker") as mock_cost, \
             patch("src.host.cli.chat_loop.CommandHistory") as mock_history, \
             patch("src.host.cli.chat_loop.BashModeExecutor") as mock_bash, \
             patch("src.host.cli.chat_loop.SessionManager") as mock_session, \
             patch("src.host.cli.chat_loop.console") as mock_console, \
             patch("src.host.cli.chat_loop.Prompt.ask") as mock_prompt:

            mock_get_mode.return_value = ClientMode.DIRECT
            mock_mode_info.return_value = {"providers": {"google": True}}
            mock_client = AsyncMock()

            mock_response = MagicMock()
            mock_response.content = "Response"
            mock_response.thought = None
            mock_response.tool_calls = []
            mock_response.has_tool_calls = False
            mock_response.usage = {"prompt_tokens": 100, "completion_tokens": 50}

            mock_client.chat = AsyncMock(return_value=mock_response)
            mock_create.return_value = mock_client

            mock_registry_inst = MagicMock()
            mock_registry_inst.get_sensitive_tools.return_value = []
            mock_registry_inst.get_genai_tools.return_value = []
            mock_registry.return_value = mock_registry_inst

            mock_ctx_inst = MagicMock()
            mock_ctx_inst.session_id = "test-session"
            mock_ctx_inst.get_context_stats.return_value = {"messages": 0, "summaries": 0, "usage_percent": 50}
            mock_ctx_inst.get_history_for_api.return_value = []
            mock_ctx_inst.messages = []
            # Simulate summarization
            mock_ctx_inst.summaries = [MagicMock()]
            mock_ctx.return_value = mock_ctx_inst

            mock_selector_inst = MagicMock()
            mock_selector_inst.get_tools_for_mode.return_value = []
            mock_selector.return_value = mock_selector_inst

            mock_skill_inst = MagicMock()
            mock_skill_inst.get_skills_for_context.return_value = ""
            mock_skill.return_value = mock_skill_inst

            mock_checkpoint_inst = MagicMock()
            mock_checkpoint.return_value = mock_checkpoint_inst

            mock_task_inst = MagicMock()
            mock_task_inst.get_running_tasks.return_value = []
            mock_task.return_value = mock_task_inst

            mock_hook_inst = AsyncMock()
            mock_hook_inst.trigger = AsyncMock(return_value=MagicMock(continue_processing=True, reason=None, decision=MagicMock(value="allow"), modified_input=None))
            mock_hook.return_value = mock_hook_inst

            mock_cost_inst = MagicMock()
            mock_cost_inst.check_budget.return_value = {}
            mock_cost_inst.calculate_cost.return_value = 0.01
            mock_cost.return_value = mock_cost_inst

            mock_history_inst = MagicMock()
            mock_history.return_value = mock_history_inst

            mock_bash_inst = MagicMock()
            mock_bash.return_value = mock_bash_inst

            mock_session_inst = MagicMock()
            mock_session.return_value = mock_session_inst

            mock_prompt.side_effect = ["test", "exit"]

            await chat_loop(project="test")

            mock_ctx_inst.add_assistant_message.assert_called()

    async def test_chat_loop_mode_switching_plan_to_build(self):
        """Test mode switching from plan to build."""
        with patch("src.host.cli.chat_loop.sys.stdin.isatty", return_value=True), \
             patch("src.host.cli.chat_loop.ModelClient.get_mode") as mock_get_mode, \
             patch("src.host.cli.chat_loop.ModelClient.get_mode_info") as mock_mode_info, \
             patch("src.host.cli.chat_loop.ModelClient.create") as mock_create, \
             patch("src.host.cli.chat_loop.get_default_registry") as mock_registry, \
             patch("src.host.cli.chat_loop.ToolSelector") as mock_selector, \
             patch("src.host.cli.chat_loop.ContextManager") as mock_ctx, \
             patch("src.host.cli.chat_loop.SkillManager") as mock_skill, \
             patch("src.host.cli.chat_loop.CheckpointManager") as mock_checkpoint, \
             patch("src.host.cli.chat_loop.get_task_manager") as mock_task, \
             patch("src.host.cli.chat_loop.HookManager") as mock_hook, \
             patch("src.host.cli.chat_loop.CostTracker") as mock_cost, \
             patch("src.host.cli.chat_loop.CommandHistory") as mock_history, \
             patch("src.host.cli.chat_loop.BashModeExecutor") as mock_bash, \
             patch("src.host.cli.chat_loop.SessionManager") as mock_session, \
             patch("src.host.cli.chat_loop.console") as mock_console, \
             patch("src.host.cli.chat_loop.Prompt.ask") as mock_prompt:

            mock_get_mode.return_value = ClientMode.DIRECT
            mock_mode_info.return_value = {"providers": {"google": True}}
            mock_client = AsyncMock()

            mock_response = MagicMock()
            mock_response.content = "Response"
            mock_response.thought = None
            mock_response.tool_calls = [{
                "id": "call_123",
                "function": {
                    "name": "switch_mode",
                    "arguments": '{"mode": "build"}'
                }
            }]
            mock_response.has_tool_calls = True
            mock_response.usage = {"prompt_tokens": 100, "completion_tokens": 50}

            mock_response2 = MagicMock()
            mock_response2.content = "Switched"
            mock_response2.thought = None
            mock_response2.tool_calls = []
            mock_response2.has_tool_calls = False
            mock_response2.usage = {"prompt_tokens": 100, "completion_tokens": 50}

            mock_client.chat = AsyncMock(side_effect=[mock_response, mock_response2])
            mock_create.return_value = mock_client

            mock_registry_inst = MagicMock()
            mock_registry_inst.get_sensitive_tools.return_value = []
            mock_registry_inst.get_genai_tools.return_value = []
            mock_registry_inst.call_tool = AsyncMock(return_value="Mode switched")
            mock_registry.return_value = mock_registry_inst

            mock_ctx_inst = MagicMock()
            mock_ctx_inst.session_id = "test-session"
            mock_ctx_inst.get_context_stats.return_value = {"messages": 0, "summaries": 0, "usage_percent": 50}
            mock_ctx_inst.get_history_for_api.return_value = []
            mock_ctx_inst.messages = []
            mock_ctx_inst.summaries = []
            mock_ctx.return_value = mock_ctx_inst

            mock_selector_inst = MagicMock()
            mock_selector_inst.get_tools_for_mode.return_value = ["switch_mode"]
            mock_selector.return_value = mock_selector_inst

            mock_skill_inst = MagicMock()
            mock_skill_inst.get_skills_for_context.return_value = ""
            mock_skill.return_value = mock_skill_inst

            mock_checkpoint_inst = MagicMock()
            mock_checkpoint.return_value = mock_checkpoint_inst

            mock_task_inst = MagicMock()
            mock_task_inst.get_running_tasks.return_value = []
            mock_task.return_value = mock_task_inst

            mock_hook_inst = AsyncMock()
            mock_hook_inst.trigger = AsyncMock(return_value=MagicMock(continue_processing=True, reason=None, decision=MagicMock(value="allow"), modified_input=None))
            mock_hook.return_value = mock_hook_inst

            mock_cost_inst = MagicMock()
            mock_cost_inst.check_budget.return_value = {}
            mock_cost_inst.calculate_cost.return_value = 0.01
            mock_cost.return_value = mock_cost_inst

            mock_history_inst = MagicMock()
            mock_history.return_value = mock_history_inst

            mock_bash_inst = MagicMock()
            mock_bash.return_value = mock_bash_inst

            mock_session_inst = MagicMock()
            mock_session.return_value = mock_session_inst

            mock_prompt.side_effect = ["switch to build", "exit"]

            await chat_loop(project="test")

            mock_registry_inst.call_tool.assert_called()

    async def test_chat_loop_checkpoint_creation(self):
        """Test checkpoint creation on file modifications."""
        with patch("src.host.cli.chat_loop.sys.stdin.isatty", return_value=True), \
             patch("src.host.cli.chat_loop.ModelClient.get_mode") as mock_get_mode, \
             patch("src.host.cli.chat_loop.ModelClient.get_mode_info") as mock_mode_info, \
             patch("src.host.cli.chat_loop.ModelClient.create") as mock_create, \
             patch("src.host.cli.chat_loop.get_default_registry") as mock_registry, \
             patch("src.host.cli.chat_loop.ToolSelector") as mock_selector, \
             patch("src.host.cli.chat_loop.ContextManager") as mock_ctx, \
             patch("src.host.cli.chat_loop.SkillManager") as mock_skill, \
             patch("src.host.cli.chat_loop.CheckpointManager") as mock_checkpoint, \
             patch("src.host.cli.chat_loop.get_task_manager") as mock_task, \
             patch("src.host.cli.chat_loop.HookManager") as mock_hook, \
             patch("src.host.cli.chat_loop.CostTracker") as mock_cost, \
             patch("src.host.cli.chat_loop.CommandHistory") as mock_history, \
             patch("src.host.cli.chat_loop.BashModeExecutor") as mock_bash, \
             patch("src.host.cli.chat_loop.SessionManager") as mock_session, \
             patch("src.host.cli.chat_loop.console") as mock_console, \
             patch("src.host.cli.chat_loop.Prompt.ask") as mock_prompt:

            mock_get_mode.return_value = ClientMode.DIRECT
            mock_mode_info.return_value = {"providers": {"google": True}}
            mock_client = AsyncMock()

            mock_response = MagicMock()
            mock_response.content = "Writing file"
            mock_response.thought = None
            mock_response.tool_calls = [{
                "id": "call_123",
                "function": {
                    "name": "write_file",
                    "arguments": '{"path": "test.py", "content": "print(1)"}'
                }
            }]
            mock_response.has_tool_calls = True
            mock_response.usage = {"prompt_tokens": 100, "completion_tokens": 50}

            mock_response2 = MagicMock()
            mock_response2.content = "File written"
            mock_response2.thought = None
            mock_response2.tool_calls = []
            mock_response2.has_tool_calls = False
            mock_response2.usage = {"prompt_tokens": 100, "completion_tokens": 50}

            mock_client.chat = AsyncMock(side_effect=[mock_response, mock_response2])
            mock_create.return_value = mock_client

            mock_registry_inst = MagicMock()
            mock_registry_inst.get_sensitive_tools.return_value = ["write_file"]
            mock_registry_inst.get_genai_tools.return_value = []
            mock_registry_inst.call_tool = AsyncMock(return_value="Written")
            mock_registry.return_value = mock_registry_inst

            mock_ctx_inst = MagicMock()
            mock_ctx_inst.session_id = "test-session"
            mock_ctx_inst.get_context_stats.return_value = {"messages": 0, "summaries": 0, "usage_percent": 50}
            mock_ctx_inst.get_history_for_api.return_value = []
            mock_ctx_inst.messages = []
            mock_ctx_inst.summaries = []
            mock_ctx.return_value = mock_ctx_inst

            mock_selector_inst = MagicMock()
            mock_selector_inst.get_tools_for_mode.return_value = ["write_file"]
            mock_selector.return_value = mock_selector_inst

            mock_skill_inst = MagicMock()
            mock_skill_inst.get_skills_for_context.return_value = ""
            mock_skill.return_value = mock_skill_inst

            mock_checkpoint_inst = MagicMock()
            mock_checkpoint.return_value = mock_checkpoint_inst

            mock_task_inst = MagicMock()
            mock_task_inst.get_running_tasks.return_value = []
            mock_task.return_value = mock_task_inst

            mock_hook_inst = AsyncMock()
            mock_hook_inst.trigger = AsyncMock(return_value=MagicMock(continue_processing=True, reason=None, decision=MagicMock(value="allow"), modified_input=None))
            mock_hook.return_value = mock_hook_inst

            mock_cost_inst = MagicMock()
            mock_cost_inst.check_budget.return_value = {}
            mock_cost_inst.calculate_cost.return_value = 0.01
            mock_cost.return_value = mock_cost_inst

            mock_history_inst = MagicMock()
            mock_history.return_value = mock_history_inst

            mock_bash_inst = MagicMock()
            mock_bash.return_value = mock_bash_inst

            mock_session_inst = MagicMock()
            mock_session.return_value = mock_session_inst

            mock_prompt.side_effect = ["write test.py", "y", "exit"]

            await chat_loop(project="test")

            mock_checkpoint_inst.begin_checkpoint.assert_called()
            mock_checkpoint_inst.finalize_checkpoint.assert_called()

    async def test_chat_loop_cost_tracking(self):
        """Test cost tracking during chat loop."""
        with patch("src.host.cli.chat_loop.sys.stdin.isatty", return_value=True), \
             patch("src.host.cli.chat_loop.ModelClient.get_mode") as mock_get_mode, \
             patch("src.host.cli.chat_loop.ModelClient.get_mode_info") as mock_mode_info, \
             patch("src.host.cli.chat_loop.ModelClient.create") as mock_create, \
             patch("src.host.cli.chat_loop.get_default_registry") as mock_registry, \
             patch("src.host.cli.chat_loop.ToolSelector") as mock_selector, \
             patch("src.host.cli.chat_loop.ContextManager") as mock_ctx, \
             patch("src.host.cli.chat_loop.SkillManager") as mock_skill, \
             patch("src.host.cli.chat_loop.CheckpointManager") as mock_checkpoint, \
             patch("src.host.cli.chat_loop.get_task_manager") as mock_task, \
             patch("src.host.cli.chat_loop.HookManager") as mock_hook, \
             patch("src.host.cli.chat_loop.CostTracker") as mock_cost, \
             patch("src.host.cli.chat_loop.CommandHistory") as mock_history, \
             patch("src.host.cli.chat_loop.BashModeExecutor") as mock_bash, \
             patch("src.host.cli.chat_loop.SessionManager") as mock_session, \
             patch("src.host.cli.chat_loop.console") as mock_console, \
             patch("src.host.cli.chat_loop.Prompt.ask") as mock_prompt:

            mock_get_mode.return_value = ClientMode.DIRECT
            mock_mode_info.return_value = {"providers": {"google": True}}
            mock_client = AsyncMock()

            mock_response = MagicMock()
            mock_response.content = "Response"
            mock_response.thought = None
            mock_response.tool_calls = []
            mock_response.has_tool_calls = False
            mock_response.usage = {"prompt_tokens": 100, "completion_tokens": 50}

            mock_client.chat = AsyncMock(return_value=mock_response)
            mock_create.return_value = mock_client

            mock_registry_inst = MagicMock()
            mock_registry_inst.get_sensitive_tools.return_value = []
            mock_registry_inst.get_genai_tools.return_value = []
            mock_registry.return_value = mock_registry_inst

            mock_ctx_inst = MagicMock()
            mock_ctx_inst.session_id = "test-session"
            mock_ctx_inst.get_context_stats.return_value = {"messages": 0, "summaries": 0, "usage_percent": 50}
            mock_ctx_inst.get_history_for_api.return_value = []
            mock_ctx_inst.messages = []
            mock_ctx_inst.summaries = []
            mock_ctx.return_value = mock_ctx_inst

            mock_selector_inst = MagicMock()
            mock_selector_inst.get_tools_for_mode.return_value = []
            mock_selector.return_value = mock_selector_inst

            mock_skill_inst = MagicMock()
            mock_skill_inst.get_skills_for_context.return_value = ""
            mock_skill.return_value = mock_skill_inst

            mock_checkpoint_inst = MagicMock()
            mock_checkpoint.return_value = mock_checkpoint_inst

            mock_task_inst = MagicMock()
            mock_task_inst.get_running_tasks.return_value = []
            mock_task.return_value = mock_task_inst

            mock_hook_inst = AsyncMock()
            mock_hook_inst.trigger = AsyncMock(return_value=MagicMock(continue_processing=True, reason=None, decision=MagicMock(value="allow"), modified_input=None))
            mock_hook.return_value = mock_hook_inst

            mock_cost_inst = MagicMock()
            mock_cost_inst.check_budget.return_value = {}
            mock_cost_inst.calculate_cost.return_value = 0.01
            mock_cost.return_value = mock_cost_inst

            mock_history_inst = MagicMock()
            mock_history.return_value = mock_history_inst

            mock_bash_inst = MagicMock()
            mock_bash.return_value = mock_bash_inst

            mock_session_inst = MagicMock()
            mock_session.return_value = mock_session_inst

            mock_prompt.side_effect = ["test", "exit"]

            await chat_loop(project="test")

            mock_cost_inst.track.assert_called()
            mock_cost_inst.calculate_cost.assert_called()

    async def test_chat_loop_budget_warning(self):
        """Test budget warning display."""
        with patch("src.host.cli.chat_loop.sys.stdin.isatty", return_value=True), \
             patch("src.host.cli.chat_loop.ModelClient.get_mode") as mock_get_mode, \
             patch("src.host.cli.chat_loop.ModelClient.get_mode_info") as mock_mode_info, \
             patch("src.host.cli.chat_loop.ModelClient.create") as mock_create, \
             patch("src.host.cli.chat_loop.get_default_registry") as mock_registry, \
             patch("src.host.cli.chat_loop.ToolSelector") as mock_selector, \
             patch("src.host.cli.chat_loop.ContextManager") as mock_ctx, \
             patch("src.host.cli.chat_loop.SkillManager") as mock_skill, \
             patch("src.host.cli.chat_loop.CheckpointManager") as mock_checkpoint, \
             patch("src.host.cli.chat_loop.get_task_manager") as mock_task, \
             patch("src.host.cli.chat_loop.HookManager") as mock_hook, \
             patch("src.host.cli.chat_loop.CostTracker") as mock_cost, \
             patch("src.host.cli.chat_loop.CommandHistory") as mock_history, \
             patch("src.host.cli.chat_loop.BashModeExecutor") as mock_bash, \
             patch("src.host.cli.chat_loop.SessionManager") as mock_session, \
             patch("src.host.cli.chat_loop.console") as mock_console, \
             patch("src.host.cli.chat_loop.Prompt.ask") as mock_prompt:

            mock_get_mode.return_value = ClientMode.DIRECT
            mock_mode_info.return_value = {"providers": {"google": True}}
            mock_client = AsyncMock()

            mock_response = MagicMock()
            mock_response.content = "Response"
            mock_response.thought = None
            mock_response.tool_calls = []
            mock_response.has_tool_calls = False
            mock_response.usage = {"prompt_tokens": 100, "completion_tokens": 50}

            mock_client.chat = AsyncMock(return_value=mock_response)
            mock_create.return_value = mock_client

            mock_registry_inst = MagicMock()
            mock_registry_inst.get_sensitive_tools.return_value = []
            mock_registry_inst.get_genai_tools.return_value = []
            mock_registry.return_value = mock_registry_inst

            mock_ctx_inst = MagicMock()
            mock_ctx_inst.session_id = "test-session"
            mock_ctx_inst.get_context_stats.return_value = {"messages": 0, "summaries": 0, "usage_percent": 50}
            mock_ctx_inst.get_history_for_api.return_value = []
            mock_ctx_inst.messages = []
            mock_ctx_inst.summaries = []
            mock_ctx.return_value = mock_ctx_inst

            mock_selector_inst = MagicMock()
            mock_selector_inst.get_tools_for_mode.return_value = []
            mock_selector.return_value = mock_selector_inst

            mock_skill_inst = MagicMock()
            mock_skill_inst.get_skills_for_context.return_value = ""
            mock_skill.return_value = mock_skill_inst

            mock_checkpoint_inst = MagicMock()
            mock_checkpoint.return_value = mock_checkpoint_inst

            mock_task_inst = MagicMock()
            mock_task_inst.get_running_tasks.return_value = []
            mock_task.return_value = mock_task_inst

            mock_hook_inst = AsyncMock()
            mock_hook_inst.trigger = AsyncMock(return_value=MagicMock(continue_processing=True, reason=None, decision=MagicMock(value="allow"), modified_input=None))
            mock_hook.return_value = mock_hook_inst

            mock_cost_inst = MagicMock()
            mock_cost_inst.check_budget.return_value = {"warning": "Budget limit approaching"}
            mock_cost_inst.calculate_cost.return_value = 0.01
            mock_cost.return_value = mock_cost_inst

            mock_history_inst = MagicMock()
            mock_history.return_value = mock_history_inst

            mock_bash_inst = MagicMock()
            mock_bash.return_value = mock_bash_inst

            mock_session_inst = MagicMock()
            mock_session.return_value = mock_session_inst

            mock_prompt.side_effect = ["test", "exit"]

            await chat_loop(project="test")

            mock_console.print.assert_called()
            calls = [str(call) for call in mock_console.print.call_args_list]
            assert any("Budget" in str(call) for call in calls)


@pytest.mark.asyncio
class TestChatLoopSkillsAndHooks:
    """Test skills and hooks integration in chat loop."""

    async def test_chat_loop_skill_loading(self):
        """Test skill loading based on context."""
        with patch("src.host.cli.chat_loop.sys.stdin.isatty", return_value=True), \
             patch("src.host.cli.chat_loop.ModelClient.get_mode") as mock_get_mode, \
             patch("src.host.cli.chat_loop.ModelClient.get_mode_info") as mock_mode_info, \
             patch("src.host.cli.chat_loop.ModelClient.create") as mock_create, \
             patch("src.host.cli.chat_loop.get_default_registry") as mock_registry, \
             patch("src.host.cli.chat_loop.ToolSelector") as mock_selector, \
             patch("src.host.cli.chat_loop.ContextManager") as mock_ctx, \
             patch("src.host.cli.chat_loop.SkillManager") as mock_skill, \
             patch("src.host.cli.chat_loop.CheckpointManager") as mock_checkpoint, \
             patch("src.host.cli.chat_loop.get_task_manager") as mock_task, \
             patch("src.host.cli.chat_loop.HookManager") as mock_hook, \
             patch("src.host.cli.chat_loop.CostTracker") as mock_cost, \
             patch("src.host.cli.chat_loop.CommandHistory") as mock_history, \
             patch("src.host.cli.chat_loop.BashModeExecutor") as mock_bash, \
             patch("src.host.cli.chat_loop.SessionManager") as mock_session, \
             patch("src.host.cli.chat_loop.console") as mock_console, \
             patch("src.host.cli.chat_loop.Prompt.ask") as mock_prompt:

            mock_get_mode.return_value = ClientMode.DIRECT
            mock_mode_info.return_value = {"providers": {"google": True}}
            mock_client = AsyncMock()

            mock_response = MagicMock()
            mock_response.content = "Response"
            mock_response.thought = None
            mock_response.tool_calls = []
            mock_response.has_tool_calls = False
            mock_response.usage = {"prompt_tokens": 100, "completion_tokens": 50}

            mock_client.chat = AsyncMock(return_value=mock_response)
            mock_create.return_value = mock_client

            mock_registry_inst = MagicMock()
            mock_registry_inst.get_sensitive_tools.return_value = []
            mock_registry_inst.get_genai_tools.return_value = []
            mock_registry.return_value = mock_registry_inst

            mock_ctx_inst = MagicMock()
            mock_ctx_inst.session_id = "test-session"
            mock_ctx_inst.get_context_stats.return_value = {"messages": 0, "summaries": 0, "usage_percent": 50}
            mock_ctx_inst.get_history_for_api.return_value = []
            mock_ctx_inst.messages = []
            mock_ctx_inst.summaries = []
            mock_ctx.return_value = mock_ctx_inst

            mock_selector_inst = MagicMock()
            mock_selector_inst.get_tools_for_mode.return_value = []
            mock_selector.return_value = mock_selector_inst

            mock_skill_inst = MagicMock()
            mock_skill_inst.get_skills_for_context.return_value = "Skill: Python"
            mock_skill_inst.get_active_skills.return_value = ["python"]
            mock_skill.return_value = mock_skill_inst

            mock_checkpoint_inst = MagicMock()
            mock_checkpoint.return_value = mock_checkpoint_inst

            mock_task_inst = MagicMock()
            mock_task_inst.get_running_tasks.return_value = []
            mock_task.return_value = mock_task_inst

            mock_hook_inst = AsyncMock()
            mock_hook_inst.trigger = AsyncMock(return_value=MagicMock(continue_processing=True, reason=None, decision=MagicMock(value="allow"), modified_input=None))
            mock_hook.return_value = mock_hook_inst

            mock_cost_inst = MagicMock()
            mock_cost_inst.check_budget.return_value = {}
            mock_cost_inst.calculate_cost.return_value = 0.01
            mock_cost.return_value = mock_cost_inst

            mock_history_inst = MagicMock()
            mock_history.return_value = mock_history_inst

            mock_bash_inst = MagicMock()
            mock_bash.return_value = mock_bash_inst

            mock_session_inst = MagicMock()
            mock_session.return_value = mock_session_inst

            mock_prompt.side_effect = ["python code", "exit"]

            await chat_loop(project="test")

            mock_skill_inst.get_skills_for_context.assert_called()

    async def test_chat_loop_hook_triggers(self):
        """Test hook triggers during chat loop."""
        with patch("src.host.cli.chat_loop.sys.stdin.isatty", return_value=True), \
             patch("src.host.cli.chat_loop.ModelClient.get_mode") as mock_get_mode, \
             patch("src.host.cli.chat_loop.ModelClient.get_mode_info") as mock_mode_info, \
             patch("src.host.cli.chat_loop.ModelClient.create") as mock_create, \
             patch("src.host.cli.chat_loop.get_default_registry") as mock_registry, \
             patch("src.host.cli.chat_loop.ToolSelector") as mock_selector, \
             patch("src.host.cli.chat_loop.ContextManager") as mock_ctx, \
             patch("src.host.cli.chat_loop.SkillManager") as mock_skill, \
             patch("src.host.cli.chat_loop.CheckpointManager") as mock_checkpoint, \
             patch("src.host.cli.chat_loop.get_task_manager") as mock_task, \
             patch("src.host.cli.chat_loop.HookManager") as mock_hook, \
             patch("src.host.cli.chat_loop.CostTracker") as mock_cost, \
             patch("src.host.cli.chat_loop.CommandHistory") as mock_history, \
             patch("src.host.cli.chat_loop.BashModeExecutor") as mock_bash, \
             patch("src.host.cli.chat_loop.SessionManager") as mock_session, \
             patch("src.host.cli.chat_loop.console") as mock_console, \
             patch("src.host.cli.chat_loop.Prompt.ask") as mock_prompt:

            mock_get_mode.return_value = ClientMode.DIRECT
            mock_mode_info.return_value = {"providers": {"google": True}}
            mock_client = AsyncMock()

            mock_response = MagicMock()
            mock_response.content = "Response"
            mock_response.thought = None
            mock_response.tool_calls = []
            mock_response.has_tool_calls = False
            mock_response.usage = {"prompt_tokens": 100, "completion_tokens": 50}

            mock_client.chat = AsyncMock(return_value=mock_response)
            mock_create.return_value = mock_client

            mock_registry_inst = MagicMock()
            mock_registry_inst.get_sensitive_tools.return_value = []
            mock_registry_inst.get_genai_tools.return_value = []
            mock_registry.return_value = mock_registry_inst

            mock_ctx_inst = MagicMock()
            mock_ctx_inst.session_id = "test-session"
            mock_ctx_inst.get_context_stats.return_value = {"messages": 0, "summaries": 0, "usage_percent": 50}
            mock_ctx_inst.get_history_for_api.return_value = []
            mock_ctx_inst.messages = []
            mock_ctx_inst.summaries = []
            mock_ctx.return_value = mock_ctx_inst

            mock_selector_inst = MagicMock()
            mock_selector_inst.get_tools_for_mode.return_value = []
            mock_selector.return_value = mock_selector_inst

            mock_skill_inst = MagicMock()
            mock_skill_inst.get_skills_for_context.return_value = ""
            mock_skill.return_value = mock_skill_inst

            mock_checkpoint_inst = MagicMock()
            mock_checkpoint.return_value = mock_checkpoint_inst

            mock_task_inst = MagicMock()
            mock_task_inst.get_running_tasks.return_value = []
            mock_task.return_value = mock_task_inst

            mock_hook_inst = AsyncMock()
            mock_hook_inst.trigger = AsyncMock(return_value=MagicMock(continue_processing=True, reason=None, decision=MagicMock(value="allow"), modified_input=None))
            mock_hook.return_value = mock_hook_inst

            mock_cost_inst = MagicMock()
            mock_cost_inst.check_budget.return_value = {}
            mock_cost_inst.calculate_cost.return_value = 0.01
            mock_cost.return_value = mock_cost_inst

            mock_history_inst = MagicMock()
            mock_history.return_value = mock_history_inst

            mock_bash_inst = MagicMock()
            mock_bash.return_value = mock_bash_inst

            mock_session_inst = MagicMock()
            mock_session.return_value = mock_session_inst

            mock_prompt.side_effect = ["test", "exit"]

            await chat_loop(project="test")

            # Verify hooks were triggered
            assert mock_hook_inst.trigger.call_count > 0

    async def test_chat_loop_hook_blocks_execution(self):
        """Test hook can block user prompt execution."""
        with patch("src.host.cli.chat_loop.sys.stdin.isatty", return_value=True), \
             patch("src.host.cli.chat_loop.ModelClient.get_mode") as mock_get_mode, \
             patch("src.host.cli.chat_loop.ModelClient.get_mode_info") as mock_mode_info, \
             patch("src.host.cli.chat_loop.ModelClient.create") as mock_create, \
             patch("src.host.cli.chat_loop.get_default_registry") as mock_registry, \
             patch("src.host.cli.chat_loop.ToolSelector") as mock_selector, \
             patch("src.host.cli.chat_loop.ContextManager") as mock_ctx, \
             patch("src.host.cli.chat_loop.SkillManager") as mock_skill, \
             patch("src.host.cli.chat_loop.CheckpointManager") as mock_checkpoint, \
             patch("src.host.cli.chat_loop.get_task_manager") as mock_task, \
             patch("src.host.cli.chat_loop.HookManager") as mock_hook, \
             patch("src.host.cli.chat_loop.CostTracker") as mock_cost, \
             patch("src.host.cli.chat_loop.CommandHistory") as mock_history, \
             patch("src.host.cli.chat_loop.BashModeExecutor") as mock_bash, \
             patch("src.host.cli.chat_loop.SessionManager") as mock_session, \
             patch("src.host.cli.chat_loop.console") as mock_console, \
             patch("src.host.cli.chat_loop.Prompt.ask") as mock_prompt:

            mock_get_mode.return_value = ClientMode.DIRECT
            mock_mode_info.return_value = {"providers": {"google": True}}
            mock_client = AsyncMock()

            mock_registry_inst = MagicMock()
            mock_registry_inst.get_sensitive_tools.return_value = []
            mock_registry_inst.get_genai_tools.return_value = []
            mock_registry.return_value = mock_registry_inst

            mock_ctx_inst = MagicMock()
            mock_ctx_inst.session_id = "test-session"
            mock_ctx_inst.get_context_stats.return_value = {"messages": 0, "summaries": 0, "usage_percent": 50}
            mock_ctx_inst.get_history_for_api.return_value = []
            mock_ctx_inst.messages = []
            mock_ctx_inst.summaries = []
            mock_ctx.return_value = mock_ctx_inst

            mock_selector_inst = MagicMock()
            mock_selector_inst.get_tools_for_mode.return_value = []
            mock_selector.return_value = mock_selector_inst

            mock_skill_inst = MagicMock()
            mock_skill_inst.get_skills_for_context.return_value = ""
            mock_skill.return_value = mock_skill_inst

            mock_checkpoint_inst = MagicMock()
            mock_checkpoint.return_value = mock_checkpoint_inst

            mock_task_inst = MagicMock()
            mock_task_inst.get_running_tasks.return_value = []
            mock_task.return_value = mock_task_inst

            mock_hook_inst = AsyncMock()
            # Block execution on USER_PROMPT_SUBMIT
            def hook_trigger_side_effect(event, **kwargs):
                if event.name == "USER_PROMPT_SUBMIT":
                    return MagicMock(continue_processing=False, reason="Blocked by hook")
                return MagicMock(continue_processing=True, reason=None, decision=MagicMock(value="allow"), modified_input=None)

            mock_hook_inst.trigger = AsyncMock(side_effect=hook_trigger_side_effect)
            mock_hook.return_value = mock_hook_inst

            mock_cost_inst = MagicMock()
            mock_cost_inst.check_budget.return_value = {}
            mock_cost.return_value = mock_cost_inst

            mock_history_inst = MagicMock()
            mock_history.return_value = mock_history_inst

            mock_bash_inst = MagicMock()
            mock_bash.return_value = mock_bash_inst

            mock_session_inst = MagicMock()
            mock_session.return_value = mock_session_inst

            mock_prompt.side_effect = ["test", "exit"]

            await chat_loop(project="test")

            mock_console.print.assert_called()

    async def test_chat_loop_piped_input_handling(self):
        """Test handling of piped input."""
        with patch("src.host.cli.chat_loop.sys.stdin.isatty", return_value=False), \
             patch("src.host.cli.chat_loop.sys.stdin.read", return_value="piped input"), \
             patch("src.host.cli.chat_loop.ModelClient.get_mode") as mock_get_mode, \
             patch("src.host.cli.chat_loop.ModelClient.get_mode_info") as mock_mode_info, \
             patch("src.host.cli.chat_loop.ModelClient.create") as mock_create, \
             patch("src.host.cli.chat_loop.get_default_registry") as mock_registry, \
             patch("src.host.cli.chat_loop.ToolSelector") as mock_selector, \
             patch("src.host.cli.chat_loop.ContextManager") as mock_ctx, \
             patch("src.host.cli.chat_loop.SkillManager") as mock_skill, \
             patch("src.host.cli.chat_loop.CheckpointManager") as mock_checkpoint, \
             patch("src.host.cli.chat_loop.get_task_manager") as mock_task, \
             patch("src.host.cli.chat_loop.HookManager") as mock_hook, \
             patch("src.host.cli.chat_loop.CostTracker") as mock_cost, \
             patch("src.host.cli.chat_loop.CommandHistory") as mock_history, \
             patch("src.host.cli.chat_loop.BashModeExecutor") as mock_bash, \
             patch("src.host.cli.chat_loop.SessionManager") as mock_session, \
             patch("src.host.cli.chat_loop.console") as mock_console:

            mock_get_mode.return_value = ClientMode.DIRECT
            mock_mode_info.return_value = {"providers": {"google": True}}
            mock_client = AsyncMock()

            mock_response = MagicMock()
            mock_response.content = "Response"
            mock_response.thought = None
            mock_response.tool_calls = []
            mock_response.has_tool_calls = False
            mock_response.usage = {"prompt_tokens": 100, "completion_tokens": 50}

            mock_client.chat = AsyncMock(return_value=mock_response)
            mock_create.return_value = mock_client

            mock_registry_inst = MagicMock()
            mock_registry_inst.get_sensitive_tools.return_value = []
            mock_registry_inst.get_genai_tools.return_value = []
            mock_registry.return_value = mock_registry_inst

            mock_ctx_inst = MagicMock()
            mock_ctx_inst.session_id = "test-session"
            mock_ctx_inst.get_context_stats.return_value = {"messages": 0, "summaries": 0, "usage_percent": 50}
            mock_ctx_inst.get_history_for_api.return_value = []
            mock_ctx_inst.messages = []
            mock_ctx_inst.summaries = []
            mock_ctx.return_value = mock_ctx_inst

            mock_selector_inst = MagicMock()
            mock_selector_inst.get_tools_for_mode.return_value = []
            mock_selector.return_value = mock_selector_inst

            mock_skill_inst = MagicMock()
            mock_skill_inst.get_skills_for_context.return_value = ""
            mock_skill.return_value = mock_skill_inst

            mock_checkpoint_inst = MagicMock()
            mock_checkpoint.return_value = mock_checkpoint_inst

            mock_task_inst = MagicMock()
            mock_task_inst.get_running_tasks.return_value = []
            mock_task.return_value = mock_task_inst

            mock_hook_inst = AsyncMock()
            mock_hook_inst.trigger = AsyncMock(return_value=MagicMock(continue_processing=True, reason=None, decision=MagicMock(value="allow"), modified_input=None))
            mock_hook.return_value = mock_hook_inst

            mock_cost_inst = MagicMock()
            mock_cost_inst.check_budget.return_value = {}
            mock_cost_inst.calculate_cost.return_value = 0.01
            mock_cost.return_value = mock_cost_inst

            mock_history_inst = MagicMock()
            mock_history.return_value = mock_history_inst

            mock_bash_inst = MagicMock()
            mock_bash.return_value = mock_bash_inst

            mock_session_inst = MagicMock()
            mock_session.return_value = mock_session_inst

            await chat_loop(project="test")

            mock_console.print.assert_called()
            calls = [str(call) for call in mock_console.print.call_args_list]
            assert any("headless" in str(call).lower() for call in calls)

    async def test_chat_loop_running_background_tasks_display(self):
        """Test display of running background tasks."""
        with patch("src.host.cli.chat_loop.sys.stdin.isatty", return_value=True), \
             patch("src.host.cli.chat_loop.ModelClient.get_mode") as mock_get_mode, \
             patch("src.host.cli.chat_loop.ModelClient.get_mode_info") as mock_mode_info, \
             patch("src.host.cli.chat_loop.ModelClient.create") as mock_create, \
             patch("src.host.cli.chat_loop.get_default_registry") as mock_registry, \
             patch("src.host.cli.chat_loop.ToolSelector") as mock_selector, \
             patch("src.host.cli.chat_loop.ContextManager") as mock_ctx, \
             patch("src.host.cli.chat_loop.SkillManager") as mock_skill, \
             patch("src.host.cli.chat_loop.CheckpointManager") as mock_checkpoint, \
             patch("src.host.cli.chat_loop.get_task_manager") as mock_task, \
             patch("src.host.cli.chat_loop.HookManager") as mock_hook, \
             patch("src.host.cli.chat_loop.CostTracker") as mock_cost, \
             patch("src.host.cli.chat_loop.CommandHistory") as mock_history, \
             patch("src.host.cli.chat_loop.BashModeExecutor") as mock_bash, \
             patch("src.host.cli.chat_loop.SessionManager") as mock_session, \
             patch("src.host.cli.chat_loop.console") as mock_console, \
             patch("src.host.cli.chat_loop.Prompt.ask") as mock_prompt:

            mock_get_mode.return_value = ClientMode.DIRECT
            mock_mode_info.return_value = {"providers": {"google": True}}
            mock_client = AsyncMock()

            mock_response = MagicMock()
            mock_response.content = "Response"
            mock_response.thought = None
            mock_response.tool_calls = []
            mock_response.has_tool_calls = False
            mock_response.usage = {"prompt_tokens": 100, "completion_tokens": 50}

            mock_client.chat = AsyncMock(return_value=mock_response)
            mock_create.return_value = mock_client

            mock_registry_inst = MagicMock()
            mock_registry_inst.get_sensitive_tools.return_value = []
            mock_registry_inst.get_genai_tools.return_value = []
            mock_registry.return_value = mock_registry_inst

            mock_ctx_inst = MagicMock()
            mock_ctx_inst.session_id = "test-session"
            mock_ctx_inst.get_context_stats.return_value = {"messages": 0, "summaries": 0, "usage_percent": 50}
            mock_ctx_inst.get_history_for_api.return_value = []
            mock_ctx_inst.messages = []
            mock_ctx_inst.summaries = []
            mock_ctx.return_value = mock_ctx_inst

            mock_selector_inst = MagicMock()
            mock_selector_inst.get_tools_for_mode.return_value = []
            mock_selector.return_value = mock_selector_inst

            mock_skill_inst = MagicMock()
            mock_skill_inst.get_skills_for_context.return_value = ""
            mock_skill.return_value = mock_skill_inst

            mock_checkpoint_inst = MagicMock()
            mock_checkpoint.return_value = mock_checkpoint_inst

            mock_task_inst = MagicMock()
            mock_task_inst.get_running_tasks.return_value = [MagicMock(), MagicMock()]
            mock_task.return_value = mock_task_inst

            mock_hook_inst = AsyncMock()
            mock_hook_inst.trigger = AsyncMock(return_value=MagicMock(continue_processing=True, reason=None, decision=MagicMock(value="allow"), modified_input=None))
            mock_hook.return_value = mock_hook_inst

            mock_cost_inst = MagicMock()
            mock_cost_inst.check_budget.return_value = {}
            mock_cost_inst.calculate_cost.return_value = 0.01
            mock_cost.return_value = mock_cost_inst

            mock_history_inst = MagicMock()
            mock_history.return_value = mock_history_inst

            mock_bash_inst = MagicMock()
            mock_bash.return_value = mock_bash_inst

            mock_session_inst = MagicMock()
            mock_session.return_value = mock_session_inst

            mock_prompt.side_effect = ["test", "exit"]

            await chat_loop(project="test")

            mock_console.print.assert_called()
            calls = [str(call) for call in mock_console.print.call_args_list]
            assert any("background task" in str(call).lower() for call in calls)
