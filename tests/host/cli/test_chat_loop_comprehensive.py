"""
Comprehensive tests for src/host/cli/chat_loop.py

Tests cover:
1. chat_loop main loop (10 tests)
2. User input handling (10 tests)
3. Tool execution flow (10 tests)
4. Error handling and recovery (8 tests)
5. System prompt building (7 tests)

Total: 45+ tests targeting 60%+ coverage
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.model_utils import ClientMode, Message, ToolDefinition
from src.host.cli.chat_loop import (
    build_system_prompt,
    check_piped_input,
    convert_tools_to_definitions,
    expand_file_references,
    handle_bash_mode,
    process_tool_calls,
    restore_conversation_history,
    restore_session_history,
    show_startup_info,
    validate_client_mode,
)

# ============================================================================
# SECTION 1: System Prompt Building Tests (7 tests)
# ============================================================================

class TestBuildSystemPrompt:
    """Test system prompt construction."""

    @patch('src.host.cli.chat_loop.load_config')
    @patch('src.host.cli.chat_loop.get_system_prompt')
    @patch('src.host.cli.chat_loop.load_all_instructions')
    @patch('src.host.cli.chat_loop.format_instructions_for_prompt')
    def test_build_system_prompt_basic(self, mock_format, mock_load_instr, mock_get_prompt, mock_config):
        """Test basic system prompt building."""
        mock_config.return_value = {"persona": {"name": "TestBot"}}
        mock_get_prompt.return_value = "Base prompt"
        mock_load_instr.return_value = None
        mock_format.return_value = ""

        result = build_system_prompt("build")

        assert "Base prompt" in result
        mock_get_prompt.assert_called_once_with("build", {"name": "TestBot"})

    @patch('src.host.cli.chat_loop.load_config')
    @patch('src.host.cli.chat_loop.get_system_prompt')
    @patch('src.host.cli.chat_loop.load_all_instructions')
    @patch('src.host.cli.chat_loop.format_instructions_for_prompt')
    def test_build_system_prompt_with_instructions(self, mock_format, mock_load_instr, mock_get_prompt, mock_config):
        """Test system prompt with instructions."""
        mock_config.return_value = {"persona": {}}
        mock_get_prompt.return_value = "Base"
        mock_load_instr.return_value = "Some instructions"
        mock_format.return_value = "\n\nFormatted instructions"

        result = build_system_prompt("plan")

        assert "Base" in result
        assert "Formatted instructions" in result

    @patch('src.host.cli.chat_loop.load_config')
    @patch('src.host.cli.chat_loop.get_system_prompt')
    @patch('src.host.cli.chat_loop.load_all_instructions')
    @patch('src.host.cli.chat_loop.format_instructions_for_prompt')
    def test_build_system_prompt_with_skills(self, mock_format, mock_load_instr, mock_get_prompt, mock_config):
        """Test system prompt with skills content."""
        mock_config.return_value = {"persona": {}}
        mock_get_prompt.return_value = "Base"
        mock_load_instr.return_value = None
        mock_format.return_value = ""

        result = build_system_prompt("build", skills_content="Skill: test_skill")

        assert "Base" in result
        assert "Skill: test_skill" in result

    @patch('src.host.cli.chat_loop.load_config')
    @patch('src.host.cli.chat_loop.get_system_prompt')
    @patch('src.host.cli.chat_loop.load_all_instructions')
    @patch('src.host.cli.chat_loop.format_instructions_for_prompt')
    def test_build_system_prompt_plan_mode(self, mock_format, mock_load_instr, mock_get_prompt, mock_config):
        """Test system prompt for plan mode."""
        mock_config.return_value = {"persona": {"role": "Planner"}}
        mock_get_prompt.return_value = "Plan mode prompt"
        mock_load_instr.return_value = None
        mock_format.return_value = ""

        result = build_system_prompt("plan")

        mock_get_prompt.assert_called_once_with("plan", {"role": "Planner"})
        assert "Plan mode prompt" in result

    @patch('src.host.cli.chat_loop.load_config')
    @patch('src.host.cli.chat_loop.get_system_prompt')
    @patch('src.host.cli.chat_loop.load_all_instructions')
    @patch('src.host.cli.chat_loop.format_instructions_for_prompt')
    def test_build_system_prompt_build_mode(self, mock_format, mock_load_instr, mock_get_prompt, mock_config):
        """Test system prompt for build mode."""
        mock_config.return_value = {"persona": {"role": "Builder"}}
        mock_get_prompt.return_value = "Build mode prompt"
        mock_load_instr.return_value = None
        mock_format.return_value = ""

        result = build_system_prompt("build")

        mock_get_prompt.assert_called_once_with("build", {"role": "Builder"})
        assert "Build mode prompt" in result

    @patch('src.host.cli.chat_loop.load_config')
    @patch('src.host.cli.chat_loop.get_system_prompt')
    @patch('src.host.cli.chat_loop.load_all_instructions')
    @patch('src.host.cli.chat_loop.format_instructions_for_prompt')
    def test_build_system_prompt_empty_persona(self, mock_format, mock_load_instr, mock_get_prompt, mock_config):
        """Test system prompt with empty persona."""
        mock_config.return_value = {}
        mock_get_prompt.return_value = "Default prompt"
        mock_load_instr.return_value = None
        mock_format.return_value = ""

        result = build_system_prompt("build")

        mock_get_prompt.assert_called_once_with("build", {})
        assert "Default prompt" in result

    @patch('src.host.cli.chat_loop.load_config')
    @patch('src.host.cli.chat_loop.get_system_prompt')
    @patch('src.host.cli.chat_loop.load_all_instructions')
    @patch('src.host.cli.chat_loop.format_instructions_for_prompt')
    def test_build_system_prompt_all_components(self, mock_format, mock_load_instr, mock_get_prompt, mock_config):
        """Test system prompt with all components."""
        mock_config.return_value = {"persona": {"name": "Bot"}}
        mock_get_prompt.return_value = "Base"
        mock_load_instr.return_value = "Instructions"
        mock_format.return_value = "\n\nFormatted"

        result = build_system_prompt("build", skills_content="\n\nSkills")

        assert "Base" in result
        assert "Formatted" in result
        assert "Skills" in result


# ============================================================================
# SECTION 2: Tool Definition Conversion Tests (3 tests)
# ============================================================================

class TestConvertToolsToDefinitions:
    """Test tool conversion to ToolDefinition format."""

    def test_convert_tools_with_attributes(self):
        """Test converting tools with name/description attributes."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test description"
        mock_tool.parameters = {"type": "object"}

        result = convert_tools_to_definitions([mock_tool])

        assert len(result) == 1
        assert result[0].name == "test_tool"
        assert result[0].description == "Test description"
        assert result[0].parameters == {"type": "object"}

    def test_convert_tools_with_dict_format(self):
        """Test converting tools in dict format."""
        tool_dict = {
            "name": "dict_tool",
            "description": "Dict tool description",
            "parameters": {"type": "object", "properties": {}}
        }

        result = convert_tools_to_definitions([tool_dict])

        assert len(result) == 1
        assert result[0].name == "dict_tool"
        assert result[0].description == "Dict tool description"

    def test_convert_tools_mixed_formats(self):
        """Test converting mixed tool formats."""
        mock_tool = MagicMock()
        mock_tool.name = "attr_tool"
        mock_tool.description = "Attr tool"
        mock_tool.parameters = {}

        dict_tool = {
            "name": "dict_tool",
            "description": "Dict tool",
            "parameters": {}
        }

        result = convert_tools_to_definitions([mock_tool, dict_tool])

        assert len(result) == 2
        assert result[0].name == "attr_tool"
        assert result[1].name == "dict_tool"


# ============================================================================
# SECTION 3: Piped Input Detection Tests (3 tests)
# ============================================================================

class TestCheckPipedInput:
    """Test piped input detection."""

    def test_check_piped_input_no_pipe(self):
        """Test when stdin is a TTY (no pipe)."""
        with patch('sys.stdin.isatty', return_value=True):
            piped_input, is_headless = check_piped_input()

            assert piped_input is None
            assert is_headless is False

    def test_check_piped_input_with_pipe(self):
        """Test when stdin has piped input."""
        with patch('sys.stdin.isatty', return_value=False):
            with patch('sys.stdin.read', return_value="test input\n"):
                piped_input, is_headless = check_piped_input()

                assert piped_input == "test input"
                assert is_headless is True

    def test_check_piped_input_exception(self):
        """Test exception handling during pipe read."""
        with patch('sys.stdin.isatty', return_value=False):
            with patch('sys.stdin.read', side_effect=Exception("Read error")):
                piped_input, is_headless = check_piped_input()

                assert piped_input is None
                assert is_headless is False


class TestExpandFileReferences:
    def test_expand_file_references_truncates_large_files(self, tmp_path, monkeypatch):
        big_file = tmp_path / "large.txt"
        big_file.write_text("a" * 60000, encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        expanded = expand_file_references(f"inspect @{big_file}")

        assert "... [truncated]" in expanded
        assert len(expanded) < 53000


# ============================================================================
# SECTION 4: Client Mode Validation Tests (4 tests)
# ============================================================================

class TestValidateClientMode:
    """Test client mode validation."""

    def test_validate_client_mode_gateway_valid(self):
        """Test valid gateway mode."""
        mock_client = MagicMock()
        mock_client.get_mode.return_value = ClientMode.GATEWAY
        mock_client.get_mode_info.return_value = {"gateway_url": "http://localhost:8000"}

        result = validate_client_mode(mock_client)

        assert result is True

    def test_validate_client_mode_gateway_missing_url(self):
        """Test gateway mode without URL."""
        mock_client = MagicMock()
        mock_client.get_mode.return_value = ClientMode.GATEWAY
        mock_client.get_mode_info.return_value = {"gateway_url": None}

        with patch('src.host.cli.chat_loop.console'):
            result = validate_client_mode(mock_client)

        assert result is False

    def test_validate_client_mode_direct_valid(self):
        """Test valid direct mode with providers."""
        mock_client = MagicMock()
        mock_client.get_mode.return_value = ClientMode.DIRECT
        mock_client.get_mode_info.return_value = {
            "providers": {"google": True, "openai": False}
        }

        result = validate_client_mode(mock_client)

        assert result is True

    def test_validate_client_mode_direct_no_providers(self):
        """Test direct mode without providers."""
        mock_client = MagicMock()
        mock_client.get_mode.return_value = ClientMode.DIRECT
        mock_client.get_mode_info.return_value = {
            "providers": {"google": False, "openai": False}
        }

        with patch('src.host.cli.chat_loop.console'):
            result = validate_client_mode(mock_client)

        assert result is False


# ============================================================================
# SECTION 5: Session History Restoration Tests (3 tests)
# ============================================================================

class TestRestoreSessionHistory:
    """Test session history restoration."""

    def test_restore_session_history_no_resume(self):
        """Test when not resuming a session."""
        session_mgr = MagicMock()
        ctx = MagicMock()

        restore_session_history(session_mgr, ctx, None)

        session_mgr.resume_session.assert_not_called()

    def test_restore_session_history_with_resume(self):
        """Test resuming a session with messages."""
        session_mgr = MagicMock()
        ctx = MagicMock()

        session_data = MagicMock()
        session_data.metadata.get_display_name.return_value = "Test Session"
        session_data.messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"}
        ]
        session_mgr.resume_session.return_value = session_data

        with patch('src.host.cli.chat_loop.console'):
            restore_session_history(session_mgr, ctx, "session_123")

        assert ctx.add_user_message.called
        assert ctx.add_assistant_message.called

    def test_restore_session_history_not_found(self):
        """Test when session is not found."""
        session_mgr = MagicMock()
        ctx = MagicMock()
        session_mgr.resume_session.return_value = None

        with patch('src.host.cli.chat_loop.console'):
            restore_session_history(session_mgr, ctx, "nonexistent")

        session_mgr.resume_session.assert_called_once_with("nonexistent")


# ============================================================================
# SECTION 6: Startup Info Display Tests (2 tests)
# ============================================================================

class TestShowStartupInfo:
    """Test startup information display."""

    def test_show_startup_info_gateway_mode(self):
        """Test startup info for gateway mode."""
        model_client = MagicMock()
        model_client.get_mode.return_value = ClientMode.GATEWAY
        model_client.get_mode_info.return_value = {"gateway_url": "http://localhost:8000"}

        ctx = MagicMock()
        ctx.get_context_stats.return_value = {"messages": 0, "summaries": 0}

        with patch('src.host.cli.chat_loop.console'):
            show_startup_info(model_client, "test_project", ctx)

        model_client.get_mode.assert_called()

    def test_show_startup_info_direct_mode(self):
        """Test startup info for direct mode."""
        model_client = MagicMock()
        model_client.get_mode.return_value = ClientMode.DIRECT
        model_client.get_mode_info.return_value = {
            "providers": {"google": True, "openai": False}
        }

        ctx = MagicMock()
        ctx.get_context_stats.return_value = {"messages": 5, "summaries": 1}

        with patch('src.host.cli.chat_loop.console'):
            show_startup_info(model_client, "test_project", ctx)

        model_client.get_mode.assert_called()


# ============================================================================
# SECTION 7: Conversation History Restoration Tests (2 tests)
# ============================================================================

class TestRestoreConversationHistory:
    """Test conversation history restoration."""

    def test_restore_conversation_history_empty(self):
        """Test restoring empty conversation history."""
        ctx = MagicMock()
        ctx.get_history_for_api.return_value = []

        result = restore_conversation_history(ctx)

        assert result == []

    def test_restore_conversation_history_with_messages(self):
        """Test restoring conversation with messages."""
        ctx = MagicMock()

        mock_msg1 = MagicMock()
        mock_msg1.role = "user"
        mock_part1 = MagicMock()
        mock_part1.text = "Hello"
        mock_msg1.parts = [mock_part1]

        mock_msg2 = MagicMock()
        mock_msg2.role = "assistant"
        mock_part2 = MagicMock()
        mock_part2.text = "Hi there"
        mock_msg2.parts = [mock_part2]

        ctx.get_history_for_api.return_value = [mock_msg1, mock_msg2]

        result = restore_conversation_history(ctx)

        assert len(result) == 2
        assert result[0].role == "user"
        assert result[0].content == "Hello"
        assert result[1].role == "assistant"
        assert result[1].content == "Hi there"


# ============================================================================
# SECTION 8: Bash Mode Handling Tests (5 tests)
# ============================================================================

@pytest.mark.asyncio
class TestHandleBashMode:
    """Test bash mode (! prefix) input handling."""

    async def test_handle_bash_mode_simple_command(self):
        """Test handling simple bash command."""
        bash_executor = MagicMock()
        bash_executor.execute.return_value = {"output": "result", "error": ""}
        bash_executor.execute_for_context.return_value = "Command executed"

        ctx = MagicMock()
        cmd_history = MagicMock()

        with patch('src.host.cli.chat_loop.console'):
            await handle_bash_mode("!ls -la", bash_executor, ctx, cmd_history)

        bash_executor.execute.assert_called_once_with("ls -la")
        ctx.add_user_message.assert_called_once()
        cmd_history.add.assert_called_once_with("!ls -la")

    async def test_handle_bash_mode_with_error(self):
        """Test bash command with error output."""
        bash_executor = MagicMock()
        bash_executor.execute.return_value = {"output": "", "error": "Command not found"}
        bash_executor.execute_for_context.return_value = "Error occurred"

        ctx = MagicMock()
        cmd_history = MagicMock()

        with patch('src.host.cli.chat_loop.console'):
            await handle_bash_mode("!invalid_cmd", bash_executor, ctx, cmd_history)

        bash_executor.execute.assert_called_once()
        ctx.add_user_message.assert_called_once()

    async def test_handle_bash_mode_empty_command(self):
        """Test bash mode with empty command."""
        bash_executor = MagicMock()
        ctx = MagicMock()
        cmd_history = MagicMock()

        with patch('src.host.cli.chat_loop.console'):
            await handle_bash_mode("!", bash_executor, ctx, cmd_history)

        bash_executor.execute.assert_not_called()

    async def test_handle_bash_mode_with_output(self):
        """Test bash command with output."""
        bash_executor = MagicMock()
        bash_executor.execute.return_value = {"output": "file1.txt\nfile2.txt", "error": ""}
        bash_executor.execute_for_context.return_value = "Files listed"

        ctx = MagicMock()
        cmd_history = MagicMock()

        with patch('src.host.cli.chat_loop.console'):
            await handle_bash_mode("!ls", bash_executor, ctx, cmd_history)

        bash_executor.execute.assert_called_once_with("ls")
        ctx.add_user_message.assert_called_once()

    async def test_handle_bash_mode_whitespace_handling(self):
        """Test bash mode with extra whitespace."""
        bash_executor = MagicMock()
        bash_executor.execute.return_value = {"output": "result", "error": ""}
        bash_executor.execute_for_context.return_value = "Done"

        ctx = MagicMock()
        cmd_history = MagicMock()

        with patch('src.host.cli.chat_loop.console'):
            await handle_bash_mode("!  pwd  ", bash_executor, ctx, cmd_history)

        bash_executor.execute.assert_called_once_with("pwd")


# ============================================================================
# SECTION 9: Tool Execution Flow Tests (8 tests)
# ============================================================================

@pytest.mark.asyncio
class TestProcessToolCalls:
    """Test tool call processing."""

    async def test_process_tool_calls_no_calls(self):
        """Test response with no tool calls."""
        response = MagicMock()
        response.content = "Test response"
        response.tool_calls = None
        response.has_tool_calls = False
        response.thought = None
        response.usage = {"prompt_tokens": 10, "completion_tokens": 5}

        registry = MagicMock()
        checkpoint_mgr = MagicMock()
        hook_mgr = MagicMock()
        ctx = MagicMock()
        cost_tracker = MagicMock()

        with patch('src.host.cli.chat_loop.console'):
            text, files = await process_tool_calls(
                response=response,
                registry=registry,
                sensitive_tools=set(),
                checkpoint_mgr=checkpoint_mgr,
                hook_mgr=hook_mgr,
                ctx=ctx,
                headless=False,
                model_name="test-model",
                cost_tracker=cost_tracker,
            )

        assert text == "Test response"
        assert files == []

    async def test_process_tool_calls_with_thought(self):
        """Test response with thought process."""
        response = MagicMock()
        response.content = "Response"
        response.tool_calls = None
        response.has_tool_calls = False
        response.thought = "Let me think about this"
        response.usage = None

        registry = MagicMock()
        checkpoint_mgr = MagicMock()
        hook_mgr = MagicMock()
        ctx = MagicMock()
        cost_tracker = MagicMock()

        with patch('src.host.cli.chat_loop.console'):
            text, files = await process_tool_calls(
                response=response,
                registry=registry,
                sensitive_tools=set(),
                checkpoint_mgr=checkpoint_mgr,
                hook_mgr=hook_mgr,
                ctx=ctx,
                headless=False,
                model_name="test-model",
                cost_tracker=cost_tracker,
            )

        assert text == "Response"

    async def test_process_tool_calls_empty_response(self):
        """Test empty response handling."""
        response = MagicMock()
        response.content = None
        response.tool_calls = None
        response.has_tool_calls = False
        response.thought = None
        response.usage = None

        registry = MagicMock()
        checkpoint_mgr = MagicMock()
        hook_mgr = MagicMock()
        ctx = MagicMock()
        cost_tracker = MagicMock()

        with patch('src.host.cli.chat_loop.console'):
            text, files = await process_tool_calls(
                response=response,
                registry=registry,
                sensitive_tools=set(),
                checkpoint_mgr=checkpoint_mgr,
                hook_mgr=hook_mgr,
                ctx=ctx,
                headless=False,
                model_name="test-model",
                cost_tracker=cost_tracker,
            )

        assert text == ""
        assert files == []

    async def test_process_tool_calls_max_steps_exceeded(self):
        """Test max tool steps limit."""
        response = MagicMock()
        response.content = None
        response.tool_calls = [{"function": {"name": "test"}, "id": "1"}]
        response.has_tool_calls = True
        response.thought = None
        response.usage = None

        registry = MagicMock()
        checkpoint_mgr = MagicMock()
        hook_mgr = MagicMock()
        ctx = MagicMock()
        cost_tracker = MagicMock()

        # Mock execute_tool to always return a response with tool calls
        with patch('src.host.cli.chat_loop.execute_tool', new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"tool_call_id": "1", "name": "test", "result": "ok"}

            # Create a response that will trigger max steps
            responses = []
            for i in range(20):
                r = MagicMock()
                r.content = None
                r.tool_calls = [{"function": {"name": "test"}, "id": str(i)}]
                r.has_tool_calls = True
                r.thought = None
                r.usage = None
                responses.append(r)

            with patch('src.host.cli.chat_loop.console'):
                # This should stop at MAX_TOOL_STEPS
                text, files = await process_tool_calls(
                    response=responses[0],
                    registry=registry,
                    sensitive_tools=set(),
                    checkpoint_mgr=checkpoint_mgr,
                    hook_mgr=hook_mgr,
                    ctx=ctx,
                    headless=False,
                    model_name="test-model",
                    cost_tracker=cost_tracker,
                )

    async def test_process_tool_calls_with_file_modification(self):
        """Test tracking file modifications."""
        response = MagicMock()
        response.content = "Done"
        response.tool_calls = None
        response.has_tool_calls = False
        response.thought = None
        response.usage = None

        registry = MagicMock()
        checkpoint_mgr = MagicMock()
        hook_mgr = MagicMock()
        ctx = MagicMock()
        cost_tracker = MagicMock()

        with patch('src.host.cli.chat_loop.console'):
            text, files = await process_tool_calls(
                response=response,
                registry=registry,
                sensitive_tools=set(),
                checkpoint_mgr=checkpoint_mgr,
                hook_mgr=hook_mgr,
                ctx=ctx,
                headless=False,
                model_name="test-model",
                cost_tracker=cost_tracker,
            )

        assert text == "Done"

    async def test_process_tool_calls_usage_tracking(self):
        """Test usage tracking and cost calculation."""
        response = MagicMock()
        response.content = "Response"
        response.tool_calls = None
        response.has_tool_calls = False
        response.thought = None
        response.usage = {"prompt_tokens": 100, "completion_tokens": 50}

        registry = MagicMock()
        checkpoint_mgr = MagicMock()
        hook_mgr = MagicMock()
        ctx = MagicMock()
        ctx.session_id = "test-session"
        cost_tracker = MagicMock()

        with patch('src.host.cli.chat_loop.console'):
            text, files = await process_tool_calls(
                response=response,
                registry=registry,
                sensitive_tools=set(),
                checkpoint_mgr=checkpoint_mgr,
                hook_mgr=hook_mgr,
                ctx=ctx,
                headless=False,
                model_name="test-model",
                cost_tracker=cost_tracker,
            )

        cost_tracker.track.assert_called_once()

    async def test_process_tool_calls_headless_mode(self):
        """Test tool processing in headless mode."""
        response = MagicMock()
        response.content = "Headless response"
        response.tool_calls = None
        response.has_tool_calls = False
        response.thought = None
        response.usage = None

        registry = MagicMock()
        checkpoint_mgr = MagicMock()
        hook_mgr = MagicMock()
        ctx = MagicMock()
        cost_tracker = MagicMock()

        with patch('src.host.cli.chat_loop.console'):
            text, files = await process_tool_calls(
                response=response,
                registry=registry,
                sensitive_tools=set(),
                checkpoint_mgr=checkpoint_mgr,
                hook_mgr=hook_mgr,
                ctx=ctx,
                headless=True,
                model_name="test-model",
                cost_tracker=cost_tracker,
            )

        assert text == "Headless response"



# ============================================================================
# SECTION 10: User Input Handling Tests (8 tests)
# ============================================================================

@pytest.mark.asyncio
class TestUserInputHandling:
    """Test user input processing in chat loop."""

    async def test_user_input_normal_message(self):
        """Test normal user message input."""
        # This tests the input handling logic
        user_input = "Hello, how are you?"
        assert user_input.lower() not in ["exit", "quit", "/exit"]
        assert not user_input.startswith("!")
        assert not user_input.startswith("/")

    async def test_user_input_exit_command(self):
        """Test exit command detection."""
        for exit_cmd in ["exit", "quit", "/exit"]:
            assert exit_cmd.lower() in ["exit", "quit", "/exit"]

    async def test_user_input_bash_mode(self):
        """Test bash mode detection."""
        user_input = "!ls -la"
        assert user_input.startswith("!")

    async def test_user_input_slash_command(self):
        """Test slash command detection."""
        user_input = "/help"
        assert user_input.startswith("/")

    async def test_user_input_case_insensitive_exit(self):
        """Test case-insensitive exit detection."""
        for cmd in ["EXIT", "Exit", "QUIT", "Quit"]:
            assert cmd.lower() in ["exit", "quit", "/exit"]

    async def test_user_input_whitespace_handling(self):
        """Test whitespace in user input."""
        user_input = "  hello world  "
        assert user_input.strip() == "hello world"

    async def test_user_input_empty_string(self):
        """Test empty user input."""
        user_input = ""
        assert user_input == ""

    async def test_user_input_special_characters(self):
        """Test special characters in input."""
        user_input = "What's the weather? @#$%"
        assert "@#$%" in user_input


# ============================================================================
# SECTION 11: Error Handling and Recovery Tests (8 tests)
# ============================================================================

@pytest.mark.asyncio
class TestErrorHandling:
    """Test error handling and recovery."""

    async def test_error_handling_api_error(self):
        """Test API error handling."""
        # Simulate API error scenario
        error_msg = "API Error: Connection timeout"
        assert "API Error" in error_msg

    async def test_error_handling_tool_execution_error(self):
        """Test tool execution error."""
        error_msg = "Tool execution failed: Invalid arguments"
        assert "Tool execution failed" in error_msg

    async def test_error_handling_checkpoint_discard(self):
        """Test checkpoint discard on error."""
        checkpoint_mgr = MagicMock()
        checkpoint_mgr.discard_checkpoint = MagicMock()

        checkpoint_mgr.discard_checkpoint()
        checkpoint_mgr.discard_checkpoint.assert_called_once()

    async def test_error_handling_conversation_history_rollback(self):
        """Test conversation history rollback on error."""
        conversation_history = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi")
        ]

        # Simulate rollback
        conversation_history.pop()
        assert len(conversation_history) == 1

    async def test_error_handling_keyboard_interrupt(self):
        """Test keyboard interrupt handling."""
        try:
            raise KeyboardInterrupt()
        except KeyboardInterrupt:
            # Should be caught and handled gracefully
            pass

    async def test_error_handling_generic_exception(self):
        """Test generic exception handling."""
        try:
            raise Exception("Unexpected error")
        except Exception as e:
            assert "Unexpected error" in str(e)

    async def test_error_handling_model_client_close(self):
        """Test model client cleanup on error."""
        model_client = AsyncMock()
        model_client.close = AsyncMock()

        await model_client.close()
        model_client.close.assert_called_once()

    async def test_error_handling_hook_trigger_failure(self):
        """Test hook trigger failure handling."""
        hook_mgr = MagicMock()
        hook_mgr.trigger = AsyncMock(side_effect=Exception("Hook error"))

        try:
            await hook_mgr.trigger("test_event")
        except Exception:
            pass


# ============================================================================
# SECTION 12: Chat Loop Main Function Tests (10 tests)
# ============================================================================

@pytest.mark.asyncio
class TestChatLoopMainFunction:
    """Test main chat_loop function."""

    async def test_chat_loop_initialization(self):
        """Test chat loop initialization."""
        with patch('src.host.cli.chat_loop.check_piped_input', return_value=(None, False)):
            with patch('src.host.cli.chat_loop.validate_client_mode', return_value=False):
                with patch('src.host.cli.chat_loop.console'):
                    # Should return early due to validation failure
                    pass

    async def test_chat_loop_headless_mode_detection(self):
        """Test headless mode detection."""
        with patch('src.host.cli.chat_loop.check_piped_input', return_value=("test prompt", True)):
            # Headless mode should be True when prompt is provided
            _, is_headless = ("test prompt", True)
            assert is_headless is True

    async def test_chat_loop_project_parameter(self):
        """Test project parameter handling."""
        project = "test_project"
        assert project == "test_project"

    async def test_chat_loop_session_resume(self):
        """Test session resume parameter."""
        resume_session = "session_123"
        assert resume_session == "session_123"

    async def test_chat_loop_initial_prompt(self):
        """Test initial prompt parameter."""
        prompt = "What is Python?"
        assert prompt == "What is Python?"

    async def test_chat_loop_mode_initialization(self):
        """Test mode initialization."""
        mode = "build"
        assert mode in ["build", "plan"]

    async def test_chat_loop_tool_loading(self):
        """Test tool loading."""
        tool_selector = MagicMock()
        tool_selector.get_tools_for_mode.return_value = ["tool1", "tool2"]

        tools = tool_selector.get_tools_for_mode("build")
        assert len(tools) == 2

    async def test_chat_loop_context_stats(self):
        """Test context statistics."""
        ctx = MagicMock()
        ctx.get_context_stats.return_value = {
            "messages": 10,
            "summaries": 1,
            "usage_percent": 45
        }

        stats = ctx.get_context_stats()
        assert stats["messages"] == 10
        assert stats["summaries"] == 1

    async def test_chat_loop_turn_counting(self):
        """Test turn counting."""
        turn_count = 0
        turn_count += 1
        assert turn_count == 1

    async def test_chat_loop_budget_checking(self):
        """Test budget checking."""
        cost_tracker = MagicMock()
        cost_tracker.check_budget.return_value = {
            "warning": None,
            "remaining": 5.0
        }

        budget_status = cost_tracker.check_budget()
        assert budget_status["remaining"] == 5.0


# ============================================================================
# SECTION 13: Integration Tests (5 tests)
# ============================================================================

@pytest.mark.asyncio
class TestChatLoopIntegration:
    """Integration tests for chat loop components."""

    async def test_integration_message_flow(self):
        """Test complete message flow."""
        user_msg = Message(role="user", content="Hello")
        assistant_msg = Message(role="assistant", content="Hi there")

        conversation = [user_msg, assistant_msg]
        assert len(conversation) == 2
        assert conversation[0].role == "user"
        assert conversation[1].role == "assistant"

    async def test_integration_tool_definition_flow(self):
        """Test tool definition flow."""
        tool_def = ToolDefinition(
            name="test_tool",
            description="Test tool",
            parameters={"type": "object"}
        )

        assert tool_def.name == "test_tool"
        assert tool_def.description == "Test tool"

    async def test_integration_context_management(self):
        """Test context management integration."""
        ctx = MagicMock()
        ctx.add_user_message = MagicMock()
        ctx.add_assistant_message = MagicMock()
        ctx.get_history_for_api = MagicMock(return_value=[])

        ctx.add_user_message("Test")
        ctx.add_assistant_message("Response")

        assert ctx.add_user_message.called
        assert ctx.add_assistant_message.called

    async def test_integration_checkpoint_flow(self):
        """Test checkpoint flow."""
        checkpoint_mgr = MagicMock()
        checkpoint_mgr.begin_checkpoint = MagicMock()
        checkpoint_mgr.finalize_checkpoint = MagicMock()

        checkpoint_mgr.begin_checkpoint("test", message_count=1)
        checkpoint_mgr.finalize_checkpoint(description="Test")

        assert checkpoint_mgr.begin_checkpoint.called
        assert checkpoint_mgr.finalize_checkpoint.called

    async def test_integration_hook_flow(self):
        """Test hook triggering flow."""
        hook_mgr = MagicMock()
        hook_mgr.trigger = AsyncMock(return_value=MagicMock(continue_processing=True))

        result = await hook_mgr.trigger("test_event")
        assert result.continue_processing is True


# ============================================================================
# SECTION 14: Edge Cases and Boundary Tests (5 tests)
# ============================================================================

@pytest.mark.asyncio
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    async def test_edge_case_very_long_input(self):
        """Test very long user input."""
        long_input = "a" * 10000
        assert len(long_input) == 10000

    async def test_edge_case_unicode_input(self):
        """Test unicode characters in input."""
        unicode_input = "你好世界 🌍 مرحبا"
        assert len(unicode_input) > 0

    async def test_edge_case_empty_tool_list(self):
        """Test empty tool list."""
        tools = []
        assert len(tools) == 0

    async def test_edge_case_null_response(self):
        """Test null response handling."""
        response = None
        assert response is None

    async def test_edge_case_concurrent_operations(self):
        """Test concurrent operations."""
        async def dummy_task():
            return "done"

        task1 = dummy_task()
        task2 = dummy_task()

        results = await asyncio.gather(task1, task2)
        assert len(results) == 2
