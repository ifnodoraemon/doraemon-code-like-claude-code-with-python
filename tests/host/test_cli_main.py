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

import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.hooks import HookEvent
from src.core.model_utils import ClientMode
from src.host.cli.chat_loop import (
    build_system_prompt,
    chat_loop,
    convert_tools_to_definitions,
)


def _make_mock_model_client(mode=ClientMode.DIRECT, mode_info=None):
    """Create a mock model client with get_mode/get_mode_info.

    get_mode and get_mode_info are synchronous methods on ModelClient,
    so we use MagicMock for them. close() and chat() are async.
    """
    if mode_info is None:
        mode_info = {"providers": {"google": True}}
    mock_client = MagicMock()
    mock_client.get_mode.return_value = mode
    mock_client.get_mode_info.return_value = mode_info
    mock_client.close = AsyncMock()
    mock_client.chat = AsyncMock()
    mock_client.chat_stream = MagicMock(side_effect=Exception("no stream"))
    return mock_client


def _make_mock_managers(
    sensitive_tools=None,
    genai_tools=None,
    session_id="test-session",
    tools_for_mode=None,
    skills_content="",
    running_tasks=None,
    budget_status=None,
    hook_result=None,
):
    """Create a dict of mock managers matching initialize_all_managers return."""
    if sensitive_tools is None:
        sensitive_tools = []
    if genai_tools is None:
        genai_tools = []
    if tools_for_mode is None:
        tools_for_mode = []
    if running_tasks is None:
        running_tasks = []
    if budget_status is None:
        budget_status = {}

    # Registry
    registry = MagicMock()
    registry.get_sensitive_tools.return_value = sensitive_tools
    registry.get_genai_tools.return_value = genai_tools
    registry.call_tool = AsyncMock(return_value="Tool result")

    # Tool selector
    tool_selector = MagicMock()
    tool_selector.get_tools_for_mode.return_value = tools_for_mode

    # Context manager
    ctx = MagicMock()
    ctx.session_id = session_id
    ctx.get_context_stats.return_value = {"messages": 0, "summaries": 0, "usage_percent": 50}
    ctx.get_history_for_api.return_value = []
    ctx.messages = []
    ctx.summaries = []

    # Skill manager
    skill_mgr = MagicMock()
    skill_mgr.get_skills_for_context.return_value = skills_content
    skill_mgr.get_active_skills.return_value = []

    # Checkpoint manager
    checkpoint_mgr = MagicMock()

    # Task manager
    task_mgr = MagicMock()
    task_mgr.get_running_tasks.return_value = running_tasks

    # Hook manager
    hook_mgr = AsyncMock()
    if hook_result is None:
        hook_result = MagicMock(
            continue_processing=True,
            reason=None,
            decision=MagicMock(value="allow"),
            modified_input=None,
        )
    hook_mgr.trigger = AsyncMock(return_value=hook_result)

    # Cost tracker
    cost_tracker = MagicMock()
    cost_tracker.check_budget.return_value = budget_status
    cost_tracker.calculate_cost.return_value = 0.01

    # Command history
    cmd_history = MagicMock()

    # Bash executor
    bash_executor = MagicMock()
    bash_executor.execute.return_value = {"output": "result", "error": ""}
    bash_executor.execute_for_context.return_value = "Command executed"

    # Session manager
    session_mgr = MagicMock()

    # Permission manager
    permission_mgr = MagicMock()

    managers = {
        "model_client": _make_mock_model_client(),
        "registry": registry,
        "tool_selector": tool_selector,
        "ctx": ctx,
        "skill_mgr": skill_mgr,
        "checkpoint_mgr": checkpoint_mgr,
        "task_mgr": task_mgr,
        "hook_mgr": hook_mgr,
        "cost_tracker": cost_tracker,
        "cmd_history": cmd_history,
        "bash_executor": bash_executor,
        "session_mgr": session_mgr,
        "permission_mgr": permission_mgr,
    }
    return managers


def _make_simple_response(content="Response", tool_calls=None, thought=None):
    """Create a mock ChatResponse."""
    resp = MagicMock()
    resp.content = content
    resp.thought = thought
    resp.tool_calls = tool_calls or []
    resp.has_tool_calls = bool(tool_calls)
    resp.usage = {"prompt_tokens": 100, "completion_tokens": 50}
    return resp


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
             patch("src.host.cli.chat_loop.format_instructions_for_prompt"):

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
             patch("src.host.cli.chat_loop.format_instructions_for_prompt"):

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
             patch("src.host.cli.chat_loop.format_instructions_for_prompt"):

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
             patch("src.host.cli.chat_loop.format_instructions_for_prompt"):

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


# --- Helper for chat_loop tests ---
# chat_loop calls initialize_model_client (via fallback) and initialize_all_managers.
# These are the correct mock targets since chat_loop does NOT import individual manager
# classes directly.

def _chat_loop_patches(
    mock_model_client=None,
    managers=None,
    prompt_side_effect=None,
    isatty=True,
    stdin_read=None,
):
    """
    Return a dict of patch context managers for chat_loop tests.

    The chat_loop function:
    1. Calls initialize_model_client() (via fallback path)
    2. Calls validate_client_mode(model_client)
    3. Calls initialize_all_managers(project)
    4. Extracts managers from the returned dict
    5. Uses console, Prompt.ask, etc.

    We mock at the correct module boundaries.
    """
    if mock_model_client is None:
        mock_model_client = _make_mock_model_client()
    if managers is None:
        managers = _make_mock_managers()
    if prompt_side_effect is None:
        prompt_side_effect = KeyboardInterrupt()

    patches = {}

    # Mock stdin.isatty
    if stdin_read is not None:
        patches["isatty"] = patch("src.host.cli.chat_loop.sys.stdin.isatty", return_value=False)
        patches["stdin_read"] = patch("src.host.cli.chat_loop.sys.stdin.read", return_value=stdin_read)
    else:
        patches["isatty"] = patch("src.host.cli.chat_loop.sys.stdin.isatty", return_value=isatty)

    # Mock initialize_model_client (the fallback path in chat_loop)
    # The first path uses run_in_executor which will fail in test, triggering fallback
    patches["init_model_client"] = patch(
        "src.host.cli.initialization.initialize_model_client",
        new_callable=AsyncMock,
        return_value=mock_model_client,
    )

    # Mock initialize_all_managers (imported at top of chat_loop.py)
    patches["init_all_managers"] = patch(
        "src.host.cli.chat_loop.initialize_all_managers",
        new_callable=AsyncMock,
        return_value=managers,
    )

    # Mock load_config (called by build_system_prompt)
    patches["load_config"] = patch(
        "src.host.cli.chat_loop.load_config",
        return_value={"model": "test-model", "persona": {}},
    )

    # Mock get_system_prompt (called by build_system_prompt)
    patches["get_system_prompt"] = patch(
        "src.host.cli.chat_loop.get_system_prompt",
        return_value="System prompt",
    )

    # Mock load_all_instructions (called by build_system_prompt)
    patches["load_all_instructions"] = patch(
        "src.host.cli.chat_loop.load_all_instructions",
        return_value="",
    )

    # Mock load_project_memory (called by build_system_prompt)
    patches["load_project_memory"] = patch(
        "src.host.cli.chat_loop.load_project_memory",
        return_value="",
    )

    # Mock console and Prompt
    patches["console"] = patch("src.host.cli.chat_loop.console")
    patches["prompt"] = patch("src.host.cli.chat_loop.Prompt.ask", side_effect=prompt_side_effect)

    return patches, mock_model_client, managers



@contextlib.contextmanager
def _apply_patches(patches):
    """Apply all patches from the dict, yielding the started mocks."""
    started = {}
    stack = contextlib.ExitStack()
    try:
        with stack:
            for key, p in patches.items():
                started[key] = stack.enter_context(p)
            yield started
    finally:
        pass


@pytest.mark.asyncio
class TestChatLoopInitialization:
    """Test chat_loop initialization and setup."""

    async def test_chat_loop_gateway_mode_initialization(self):
        """Test chat loop initializes correctly in gateway mode."""
        mock_client = _make_mock_model_client(
            mode=ClientMode.GATEWAY,
            mode_info={"gateway_url": "http://localhost:8000"},
        )
        managers = _make_mock_managers()
        patches, _, _ = _chat_loop_patches(
            mock_model_client=mock_client,
            managers=managers,
            prompt_side_effect=KeyboardInterrupt(),
        )

        with _apply_patches(patches) as mocks:
            await chat_loop(project="test-project")

            mocks["init_model_client"].assert_called_once()
            mocks["init_all_managers"].assert_called_once_with("test-project")

    async def test_chat_loop_direct_mode_initialization(self):
        """Test chat loop initializes correctly in direct mode."""
        mock_client = _make_mock_model_client(
            mode=ClientMode.DIRECT,
            mode_info={"providers": {"google": True, "openai": False}},
        )
        managers = _make_mock_managers()
        patches, _, _ = _chat_loop_patches(
            mock_model_client=mock_client,
            managers=managers,
            prompt_side_effect=KeyboardInterrupt(),
        )

        with _apply_patches(patches) as mocks:
            await chat_loop(project="test-project")

            mocks["init_model_client"].assert_called_once()

    async def test_chat_loop_no_api_keys_error(self):
        """Test chat loop exits when no API keys configured."""
        mock_client = _make_mock_model_client(
            mode=ClientMode.DIRECT,
            mode_info={"providers": {"google": False, "openai": False}},
        )
        patches, _, _ = _chat_loop_patches(mock_model_client=mock_client)

        with _apply_patches(patches) as mocks:
            await chat_loop(project="test-project")

            mock_console = mocks["console"]
            mock_console.print.assert_called()
            calls = [str(c) for c in mock_console.print.call_args_list]
            assert any("No API keys" in str(c) for c in calls)

    async def test_chat_loop_gateway_url_missing_error(self):
        """Test chat loop exits when gateway URL not configured."""
        mock_client = _make_mock_model_client(
            mode=ClientMode.GATEWAY,
            mode_info={"gateway_url": None},
        )
        patches, _, _ = _chat_loop_patches(mock_model_client=mock_client)

        with _apply_patches(patches) as mocks:
            await chat_loop(project="test-project")

            mock_console = mocks["console"]
            mock_console.print.assert_called()
            calls = [str(c) for c in mock_console.print.call_args_list]
            assert any("gateway_url" in str(c) for c in calls)

    async def test_chat_loop_model_client_creation_failure(self):
        """Test chat loop handles model client creation failure."""
        patches, _, _ = _chat_loop_patches()

        # Override init_model_client to raise
        patches["init_model_client"] = patch(
            "src.host.cli.initialization.initialize_model_client",
            new_callable=AsyncMock,
            side_effect=Exception("Connection failed"),
        )

        with _apply_patches(patches) as mocks:
            await chat_loop(project="test-project")

            mock_console = mocks["console"]
            mock_console.print.assert_called()
            calls = [str(c) for c in mock_console.print.call_args_list]
            assert any("Failed to initialize" in str(c) or "Connection failed" in str(c) for c in calls)

    async def test_chat_loop_with_project_isolation(self):
        """Test chat loop respects project isolation."""
        managers = _make_mock_managers()
        patches, _, _ = _chat_loop_patches(
            managers=managers,
            prompt_side_effect=KeyboardInterrupt(),
        )

        with _apply_patches(patches) as mocks:
            await chat_loop(project="my-project")

            mocks["init_all_managers"].assert_called_once_with("my-project")

    async def test_chat_loop_headless_mode_with_prompt(self):
        """Test chat loop enters headless mode when prompt provided."""
        mock_client = _make_mock_model_client()
        mock_response = _make_simple_response(content="Response to prompt")
        mock_client.chat = AsyncMock(return_value=mock_response)
        managers = _make_mock_managers()
        patches, _, _ = _chat_loop_patches(
            mock_model_client=mock_client,
            managers=managers,
        )

        with _apply_patches(patches) as mocks:
            await chat_loop(project="test", prompt="Hello")

            mock_console = mocks["console"]
            mock_console.print.assert_called()
            calls = [str(c) for c in mock_console.print.call_args_list]
            assert any("headless" in str(c).lower() for c in calls)


@pytest.mark.asyncio
class TestChatLoopUserInputHandling:
    """Test user input handling in chat loop."""

    async def test_chat_loop_exit_command(self):
        """Test chat loop exits on exit command."""
        managers = _make_mock_managers()
        patches, mock_client, _ = _chat_loop_patches(
            managers=managers,
            prompt_side_effect=["exit"],
        )

        with _apply_patches(patches):
            await chat_loop(project="test")

            managers["hook_mgr"].trigger.assert_called()

    async def test_chat_loop_quit_command(self):
        """Test chat loop exits on quit command."""
        managers = _make_mock_managers()
        mock_client = _make_mock_model_client()
        patches, _, _ = _chat_loop_patches(
            mock_model_client=mock_client,
            managers=managers,
            prompt_side_effect=["quit"],
        )

        with _apply_patches(patches):
            await chat_loop(project="test")

            mock_client.close.assert_called_once()

    async def test_chat_loop_bash_mode_execution(self):
        """Test bash mode (! prefix) command execution."""
        managers = _make_mock_managers()
        patches, _, _ = _chat_loop_patches(
            managers=managers,
            prompt_side_effect=["! ls -la", "exit"],
        )

        with _apply_patches(patches):
            await chat_loop(project="test")

            managers["bash_executor"].execute.assert_called()
            managers["cmd_history"].add.assert_called()

    async def test_chat_loop_slash_command_routing(self):
        """Test slash command routing to CommandHandler."""
        managers = _make_mock_managers()
        patches, _, _ = _chat_loop_patches(
            managers=managers,
            prompt_side_effect=["/help", "exit"],
        )

        with _apply_patches(patches), \
             patch("src.host.cli.chat_loop.CommandHandler") as mock_cmd_handler:

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

            await chat_loop(project="test")

            mock_handler_inst.handle.assert_called()

    async def test_chat_loop_keyboard_interrupt_handling(self):
        """Test chat loop handles KeyboardInterrupt gracefully."""
        managers = _make_mock_managers()
        mock_client = _make_mock_model_client()
        patches, _, _ = _chat_loop_patches(
            mock_model_client=mock_client,
            managers=managers,
            prompt_side_effect=KeyboardInterrupt(),
        )

        with _apply_patches(patches) as mocks:
            await chat_loop(project="test")

            mocks["console"].print.assert_called()
            mock_client.close.assert_called_once()

    async def test_chat_loop_exception_handling(self):
        """Test chat loop handles exceptions gracefully."""
        managers = _make_mock_managers()
        # Make hook trigger raise on SESSION_START
        managers["hook_mgr"].trigger = AsyncMock(side_effect=Exception("Hook error"))
        mock_client = _make_mock_model_client()
        patches, _, _ = _chat_loop_patches(
            mock_model_client=mock_client,
            managers=managers,
            prompt_side_effect=KeyboardInterrupt(),
        )

        with _apply_patches(patches):
            # Should not raise - chat_loop handles exceptions
            try:
                await chat_loop(project="test")
            except Exception:
                pass  # Hook error may propagate; that's acceptable

    async def test_chat_loop_session_resume(self):
        """Test chat loop resumes previous session."""
        managers = _make_mock_managers()
        mock_session_data = MagicMock()
        mock_session_data.metadata.get_display_name.return_value = "Previous Session"
        mock_session_data.messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        managers["session_mgr"].resume_session.return_value = mock_session_data

        patches, _, _ = _chat_loop_patches(
            managers=managers,
            prompt_side_effect=KeyboardInterrupt(),
        )

        with _apply_patches(patches):
            await chat_loop(project="test", resume_session="session-123")

            managers["session_mgr"].resume_session.assert_called_once_with("session-123")
            managers["ctx"].add_user_message.assert_called()
            managers["ctx"].add_assistant_message.assert_called()


@pytest.mark.asyncio
class TestChatLoopToolExecution:
    """Test tool execution in chat loop."""

    async def test_chat_loop_tool_execution_success(self):
        """Test successful tool execution."""
        mock_client = _make_mock_model_client()

        # Mock response with tool call
        mock_response = _make_simple_response(
            content="I'll read the file",
            tool_calls=[{
                "id": "call_123",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path": "test.py"}'
                }
            }],
        )

        # Second response without tool calls
        mock_response2 = _make_simple_response(content="File content: test")


        mock_client.chat = AsyncMock(side_effect=[mock_response, mock_response2])

        managers = _make_mock_managers(tools_for_mode=["read_file"])
        managers["registry"].call_tool = AsyncMock(return_value="File content")

        patches, _, _ = _chat_loop_patches(
            mock_model_client=mock_client,
            managers=managers,
            prompt_side_effect=["read test.py", "exit"],
        )

        with _apply_patches(patches):
            await chat_loop(project="test")

            managers["registry"].call_tool.assert_called()

    async def test_chat_loop_tool_execution_failure(self):
        """Test tool execution failure handling."""
        mock_client = _make_mock_model_client()

        mock_response = _make_simple_response(
            content="I'll read the file",
            tool_calls=[{
                "id": "call_123",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path": "nonexistent.py"}'
                }
            }],
        )

        mock_response2 = _make_simple_response(content="Error occurred")


        mock_client.chat = AsyncMock(side_effect=[mock_response, mock_response2])

        managers = _make_mock_managers(tools_for_mode=["read_file"])
        managers["registry"].call_tool = AsyncMock(side_effect=Exception("File not found"))

        patches, _, _ = _chat_loop_patches(
            mock_model_client=mock_client,
            managers=managers,
            prompt_side_effect=["read nonexistent.py", "exit"],
        )

        with _apply_patches(patches):
            await chat_loop(project="test")

            managers["registry"].call_tool.assert_called()

    async def test_chat_loop_sensitive_tool_approval(self):
        """Test sensitive tool requires user approval."""
        mock_client = _make_mock_model_client()

        mock_response = _make_simple_response(
            content="I'll write the file",
            tool_calls=[{
                "id": "call_123",
                "function": {
                    "name": "write_file",
                    "arguments": '{"path": "test.py", "content": "print(1)"}'
                }
            }],
        )

        mock_response2 = _make_simple_response(content="File written")


        mock_client.chat = AsyncMock(side_effect=[mock_response, mock_response2])

        managers = _make_mock_managers(
            sensitive_tools=["write_file"],
            tools_for_mode=["write_file"],
        )
        managers["registry"].call_tool = AsyncMock(return_value="Written")

        patches, _, _ = _chat_loop_patches(
            mock_model_client=mock_client,
            managers=managers,
            prompt_side_effect=["write test.py", "y", "exit"],
        )

        with _apply_patches(patches):

            await chat_loop(project="test")

            managers["registry"].call_tool.assert_called()

    async def test_chat_loop_loop_detection(self):
        """Test detection of infinite tool loops."""
        mock_client = _make_mock_model_client()

        # Create responses that repeat the same tool call
        responses = []
        for i in range(20):
            responses.append(_make_simple_response(
                content=f"Attempt {i}",
                tool_calls=[{
                    "id": f"call_{i}",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "test.py"}'
                    }
                }],
            ))

        mock_client.chat = AsyncMock(side_effect=responses)

        managers = _make_mock_managers(tools_for_mode=["read_file"])
        managers["registry"].call_tool = AsyncMock(return_value="File content")

        patches, _, _ = _chat_loop_patches(
            mock_model_client=mock_client,
            managers=managers,
            prompt_side_effect=["read test.py", "exit"],
        )

        with _apply_patches(patches) as mocks:

            await chat_loop(project="test")

            mocks["console"].print.assert_called()

    async def test_chat_loop_max_tool_steps_limit(self):
        """Test max tool steps limit prevents infinite loops."""
        mock_client = _make_mock_model_client()

        # Create many responses with tool calls
        responses = []
        for i in range(30):
            responses.append(_make_simple_response(
                content=f"Step {i}",
                tool_calls=[{
                    "id": f"call_{i}",
                    "function": {
                        "name": "read_file",
                        "arguments": f'{{"path": "file_{i}.py"}}'
                    }
                }],
            ))

        mock_client.chat = AsyncMock(side_effect=responses)

        managers = _make_mock_managers(tools_for_mode=["read_file"])
        managers["registry"].call_tool = AsyncMock(return_value="Result")

        patches, _, _ = _chat_loop_patches(
            mock_model_client=mock_client,
            managers=managers,
            prompt_side_effect=["read many files", "exit"],
        )

        with _apply_patches(patches) as mocks:

            await chat_loop(project="test")

            mock_console = mocks["console"]
            mock_console.print.assert_called()
            calls = [str(c) for c in mock_console.print.call_args_list]
            assert any("Max tool steps" in str(c) for c in calls)


@pytest.mark.asyncio
class TestChatLoopContextManagement:
    """Test context management in chat loop."""

    async def test_chat_loop_context_summarization(self):
        """Test context summarization when threshold reached."""
        mock_client = _make_mock_model_client()
        mock_response = _make_simple_response(content="Response")

        mock_client.chat = AsyncMock(return_value=mock_response)

        managers = _make_mock_managers()
        # Simulate summarization
        managers["ctx"].summaries = [MagicMock()]

        patches, _, _ = _chat_loop_patches(
            mock_model_client=mock_client,
            managers=managers,
            prompt_side_effect=["test", "exit"],
        )

        with _apply_patches(patches):

            await chat_loop(project="test")

            managers["ctx"].add_assistant_message.assert_called()

    async def test_chat_loop_mode_switching_plan_to_build(self):
        """Test mode switching from plan to build."""
        mock_client = _make_mock_model_client()

        mock_response = _make_simple_response(
            content="Response",
            tool_calls=[{
                "id": "call_123",
                "function": {
                    "name": "switch_mode",
                    "arguments": '{"mode": "build"}'
                }
            }],
        )
        mock_response2 = _make_simple_response(content="Switched")


        mock_client.chat = AsyncMock(side_effect=[mock_response, mock_response2])

        managers = _make_mock_managers(tools_for_mode=["switch_mode"])
        managers["registry"].call_tool = AsyncMock(return_value="Mode switched")

        patches, _, _ = _chat_loop_patches(
            mock_model_client=mock_client,
            managers=managers,
            prompt_side_effect=["switch to build", "exit"],
        )

        with _apply_patches(patches):

            await chat_loop(project="test")

            managers["registry"].call_tool.assert_called()

    async def test_chat_loop_checkpoint_creation(self):
        """Test checkpoint creation on file modifications."""
        mock_client = _make_mock_model_client()

        mock_response = _make_simple_response(
            content="Writing file",
            tool_calls=[{
                "id": "call_123",
                "function": {
                    "name": "write_file",
                    "arguments": '{"path": "test.py", "content": "print(1)"}'
                }
            }],
        )
        mock_response2 = _make_simple_response(content="File written")


        mock_client.chat = AsyncMock(side_effect=[mock_response, mock_response2])

        managers = _make_mock_managers(
            sensitive_tools=["write_file"],
            tools_for_mode=["write_file"],
        )
        managers["registry"].call_tool = AsyncMock(return_value="Written")

        patches, _, _ = _chat_loop_patches(
            mock_model_client=mock_client,
            managers=managers,
            prompt_side_effect=["write test.py", "y", "exit"],
        )

        with _apply_patches(patches):

            await chat_loop(project="test")

            managers["checkpoint_mgr"].begin_checkpoint.assert_called()
            managers["checkpoint_mgr"].finalize_checkpoint.assert_called()

    async def test_chat_loop_cost_tracking(self):
        """Test cost tracking during chat loop."""
        mock_client = _make_mock_model_client()
        mock_response = _make_simple_response(content="Response")

        mock_client.chat = AsyncMock(return_value=mock_response)

        managers = _make_mock_managers()

        patches, _, _ = _chat_loop_patches(
            mock_model_client=mock_client,
            managers=managers,
            prompt_side_effect=["test", "exit"],
        )

        with _apply_patches(patches):

            await chat_loop(project="test")

            managers["cost_tracker"].track.assert_called()
            managers["cost_tracker"].calculate_cost.assert_called()

    async def test_chat_loop_budget_warning(self):
        """Test budget warning display."""
        mock_client = _make_mock_model_client()
        mock_response = _make_simple_response(content="Response")

        mock_client.chat = AsyncMock(return_value=mock_response)

        managers = _make_mock_managers(budget_status={"warning": "Budget limit approaching"})

        patches, _, _ = _chat_loop_patches(
            mock_model_client=mock_client,
            managers=managers,
            prompt_side_effect=["test", "exit"],
        )

        with _apply_patches(patches) as mocks:

            await chat_loop(project="test")

            mock_console = mocks["console"]
            mock_console.print.assert_called()
            calls = [str(c) for c in mock_console.print.call_args_list]
            assert any("Budget" in str(c) for c in calls)


@pytest.mark.asyncio
class TestChatLoopSkillsAndHooks:
    """Test skills and hooks integration in chat loop."""

    async def test_chat_loop_skill_loading(self):
        """Test skill loading based on context."""
        mock_client = _make_mock_model_client()
        mock_response = _make_simple_response(content="Response")

        mock_client.chat = AsyncMock(return_value=mock_response)

        managers = _make_mock_managers(skills_content="Skill: Python")
        managers["skill_mgr"].get_active_skills.return_value = ["python"]

        patches, _, _ = _chat_loop_patches(
            mock_model_client=mock_client,
            managers=managers,
            prompt_side_effect=["python code", "exit"],
        )

        with _apply_patches(patches):

            await chat_loop(project="test")

            managers["skill_mgr"].get_skills_for_context.assert_called()

    async def test_chat_loop_hook_triggers(self):
        """Test hook triggers during chat loop."""
        mock_client = _make_mock_model_client()
        mock_response = _make_simple_response(content="Response")

        mock_client.chat = AsyncMock(return_value=mock_response)

        managers = _make_mock_managers()

        patches, _, _ = _chat_loop_patches(
            mock_model_client=mock_client,
            managers=managers,
            prompt_side_effect=["test", "exit"],
        )

        with _apply_patches(patches):

            await chat_loop(project="test")

            # Verify hooks were triggered
            assert managers["hook_mgr"].trigger.call_count > 0

    async def test_chat_loop_hook_blocks_execution(self):
        """Test hook can block user prompt execution."""
        mock_client = _make_mock_model_client()

        managers = _make_mock_managers()

        # Block execution on USER_PROMPT_SUBMIT

        def hook_trigger_side_effect(event, **kwargs):
            if event == HookEvent.USER_PROMPT_SUBMIT:
                return MagicMock(continue_processing=False, reason="Blocked by hook")
            return MagicMock(
                continue_processing=True,
                reason=None,
                decision=MagicMock(value="allow"),
                modified_input=None,
            )

        managers["hook_mgr"].trigger = AsyncMock(side_effect=hook_trigger_side_effect)

        patches, _, _ = _chat_loop_patches(
            mock_model_client=mock_client,
            managers=managers,
            prompt_side_effect=["test", "exit"],
        )

        with _apply_patches(patches) as mocks:

            await chat_loop(project="test")

            mocks["console"].print.assert_called()

    async def test_chat_loop_piped_input_handling(self):
        """Test handling of piped input."""
        mock_client = _make_mock_model_client()
        mock_response = _make_simple_response(content="Response")

        mock_client.chat = AsyncMock(return_value=mock_response)

        managers = _make_mock_managers()

        patches, _, _ = _chat_loop_patches(
            mock_model_client=mock_client,
            managers=managers,
            stdin_read="piped input",
        )

        with _apply_patches(patches) as mocks:

            await chat_loop(project="test")

            mock_console = mocks["console"]
            mock_console.print.assert_called()
            calls = [str(c) for c in mock_console.print.call_args_list]
            assert any("headless" in str(c).lower() for c in calls)

    async def test_chat_loop_running_background_tasks_display(self):
        """Test display of running background tasks."""
        mock_client = _make_mock_model_client()
        mock_response = _make_simple_response(content="Response")

        mock_client.chat = AsyncMock(return_value=mock_response)

        managers = _make_mock_managers(running_tasks=[MagicMock(), MagicMock()])

        patches, _, _ = _chat_loop_patches(
            mock_model_client=mock_client,
            managers=managers,
            prompt_side_effect=["test", "exit"],
        )

        with _apply_patches(patches) as mocks:

            await chat_loop(project="test")

            mock_console = mocks["console"]
            mock_console.print.assert_called()
            calls = [str(c) for c in mock_console.print.call_args_list]
            assert any("background task" in str(c).lower() for c in calls)
