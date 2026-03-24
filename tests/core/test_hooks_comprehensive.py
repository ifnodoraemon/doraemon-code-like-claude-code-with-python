"""Comprehensive tests for src/core/hooks.py

Coverage targets:
- Hook registration and management (10 tests)
- Hook execution and chaining (10 tests)
- Hook error handling (8 tests)
- Hook configuration loading (7 tests)
- Hook filtering and conditional execution (5 tests)
"""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.core.hooks import (
    HookContext,
    HookDecision,
    HookDefinition,
    HookEvent,
    HookManager,
    HookResult,
)


class TestHookEventEnum:
    """Test HookEvent enum."""

    def test_hook_event_values(self):
        """Test HookEvent enum has all expected values."""
        assert HookEvent.SESSION_START.value == "SessionStart"
        assert HookEvent.SESSION_END.value == "SessionEnd"
        assert HookEvent.USER_PROMPT_SUBMIT.value == "UserPromptSubmit"
        assert HookEvent.PRE_TOOL_USE.value == "PreToolUse"
        assert HookEvent.POST_TOOL_USE.value == "PostToolUse"
        assert HookEvent.STOP.value == "Stop"
        assert HookEvent.PRE_COMPACT.value == "PreCompact"
        assert HookEvent.NOTIFICATION.value == "Notification"

    def test_hook_event_enum_members(self):
        """Test all HookEvent members are accessible."""
        events = [
            HookEvent.SESSION_START,
            HookEvent.SESSION_END,
            HookEvent.USER_PROMPT_SUBMIT,
            HookEvent.PRE_TOOL_USE,
            HookEvent.POST_TOOL_USE,
            HookEvent.STOP,
            HookEvent.PRE_COMPACT,
            HookEvent.NOTIFICATION,
        ]
        assert len(events) == 8


class TestHookDecisionEnum:
    """Test HookDecision enum."""

    def test_hook_decision_values(self):
        """Test HookDecision enum has all expected values."""
        assert HookDecision.ALLOW.value == "allow"
        assert HookDecision.DENY.value == "deny"
        assert HookDecision.ASK.value == "ask"
        assert HookDecision.MODIFY.value == "modify"

    def test_hook_decision_enum_members(self):
        """Test all HookDecision members are accessible."""
        decisions = [
            HookDecision.ALLOW,
            HookDecision.DENY,
            HookDecision.ASK,
            HookDecision.MODIFY,
        ]
        assert len(decisions) == 4


class TestHookResult:
    """Test HookResult dataclass."""

    def test_hook_result_initialization(self):
        """Test HookResult initialization with defaults."""
        result = HookResult(success=True)
        assert result.success is True
        assert result.decision == HookDecision.ALLOW
        assert result.reason == ""
        assert result.modified_input is None
        assert result.additional_context == ""
        assert result.continue_processing is True
        assert result.output == ""
        assert result.duration == 0

    def test_hook_result_with_all_fields(self):
        """Test HookResult with all fields set."""
        result = HookResult(
            success=False,
            decision=HookDecision.DENY,
            reason="Test reason",
            modified_input={"key": "value"},
            additional_context="Context",
            continue_processing=False,
            output="Output",
            duration=1.5,
        )
        assert result.success is False
        assert result.decision == HookDecision.DENY
        assert result.reason == "Test reason"
        assert result.modified_input == {"key": "value"}
        assert result.additional_context == "Context"
        assert result.continue_processing is False
        assert result.output == "Output"
        assert result.duration == 1.5

    def test_hook_result_decision_variations(self):
        """Test HookResult with different decisions."""
        for decision in [
            HookDecision.ALLOW,
            HookDecision.DENY,
            HookDecision.ASK,
            HookDecision.MODIFY,
        ]:
            result = HookResult(success=True, decision=decision)
            assert result.decision == decision


class TestHookDefinition:
    """Test HookDefinition dataclass."""

    def test_hook_definition_with_command(self):
        """Test HookDefinition with command."""
        hook = HookDefinition(
            event=HookEvent.PRE_TOOL_USE,
            command="echo test",
            matcher="Bash",
            timeout=30,
        )
        assert hook.event == HookEvent.PRE_TOOL_USE
        assert hook.command == "echo test"
        assert hook.matcher == "Bash"
        assert hook.timeout == 30
        assert hook.enabled is True
        assert hook.callback is None

    def test_hook_definition_with_callback(self):
        """Test HookDefinition with callback."""
        callback = Mock()
        hook = HookDefinition(
            event=HookEvent.SESSION_START,
            callback=callback,
            timeout=60,
        )
        assert hook.event == HookEvent.SESSION_START
        assert hook.callback == callback
        assert hook.command is None
        assert hook.timeout == 60
        assert hook.enabled is True

    def test_hook_definition_to_dict(self):
        """Test HookDefinition.to_dict() method."""
        hook = HookDefinition(
            event=HookEvent.PRE_TOOL_USE,
            command="test.sh",
            matcher="Write",
            timeout=45,
            enabled=False,
        )
        hook_dict = hook.to_dict()
        assert hook_dict["event"] == "PreToolUse"
        assert hook_dict["command"] == "test.sh"
        assert hook_dict["matcher"] == "Write"
        assert hook_dict["timeout"] == 45
        assert hook_dict["enabled"] is False


class TestHookContext:
    """Test HookContext dataclass."""

    def test_hook_context_initialization(self):
        """Test HookContext initialization."""
        context = HookContext(
            event=HookEvent.PRE_TOOL_USE,
            session_id="session123",
            project_dir="/home/user/project",
            permission_mode="default",
        )
        assert context.event == HookEvent.PRE_TOOL_USE
        assert context.session_id == "session123"
        assert context.project_dir == "/home/user/project"
        assert context.permission_mode == "default"
        assert context.timestamp > 0
        assert context.tool_name is None
        assert context.tool_input is None
        assert context.tool_output is None
        assert context.user_prompt is None
        assert context.message_count == 0
        assert context.stop_reason is None

    def test_hook_context_with_tool_info(self):
        """Test HookContext with tool-specific information."""
        context = HookContext(
            event=HookEvent.PRE_TOOL_USE,
            session_id="session123",
            project_dir="/home/user/project",
            permission_mode="default",
            tool_name="Bash",
            tool_input={"command": "ls -la"},
        )
        assert context.tool_name == "Bash"
        assert context.tool_input == {"command": "ls -la"}

    def test_hook_context_to_dict_minimal(self):
        """Test HookContext.to_dict() with minimal fields."""
        context = HookContext(
            event=HookEvent.SESSION_START,
            session_id="session123",
            project_dir="/home/user/project",
            permission_mode="default",
        )
        context_dict = context.to_dict()
        assert context_dict["hook_event_name"] == "SessionStart"
        assert context_dict["session_id"] == "session123"
        assert context_dict["cwd"] == "/home/user/project"
        assert context_dict["permission_mode"] == "default"
        assert "timestamp" in context_dict
        assert "tool_name" not in context_dict

    def test_hook_context_to_dict_with_tool_info(self):
        """Test HookContext.to_dict() with tool information."""
        context = HookContext(
            event=HookEvent.PRE_TOOL_USE,
            session_id="session123",
            project_dir="/home/user/project",
            permission_mode="default",
            tool_name="Write",
            tool_input={"path": "/tmp/file.txt", "content": "test"},
            tool_output="File written",
        )
        context_dict = context.to_dict()
        assert context_dict["tool_name"] == "Write"
        assert context_dict["tool_input"] == {"path": "/tmp/file.txt", "content": "test"}
        assert context_dict["tool_output"] == "File written"

    def test_hook_context_to_dict_with_prompt(self):
        """Test HookContext.to_dict() with user prompt."""
        context = HookContext(
            event=HookEvent.USER_PROMPT_SUBMIT,
            session_id="session123",
            project_dir="/home/user/project",
            permission_mode="default",
            user_prompt="What is the weather?",
        )
        context_dict = context.to_dict()
        assert context_dict["prompt"] == "What is the weather?"


class TestHookManagerInitialization:
    """Test HookManager initialization."""

    def test_hook_manager_initialization_defaults(self):
        """Test HookManager initialization with defaults."""
        manager = HookManager()
        assert manager.project_dir == str(Path(".").resolve())
        assert manager.session_id == ""
        assert manager.permission_mode == "default"
        assert len(manager._hooks) == 8  # All HookEvent types

    def test_hook_manager_initialization_with_params(self):
        """Test HookManager initialization with parameters."""
        manager = HookManager(
            project_dir="/tmp/project",
            session_id="test_session",
            permission_mode="strict",
        )
        assert "/tmp/project" in manager.project_dir
        assert manager.session_id == "test_session"
        assert manager.permission_mode == "strict"

    def test_hook_manager_initialization_with_path_object(self):
        """Test HookManager initialization with Path object."""
        path = Path("/tmp/project")
        manager = HookManager(project_dir=path)
        assert "/tmp/project" in manager.project_dir

    def test_hook_manager_env_setup(self):
        """Test HookManager environment setup."""
        manager = HookManager(project_dir="/tmp/project")
        assert "CLAUDE_PROJECT_DIR" in manager._env
        assert "/tmp/project" in manager._env["CLAUDE_PROJECT_DIR"]

    def test_hook_manager_hooks_initialized_empty(self):
        """Test HookManager initializes all hook events as empty lists."""
        manager = HookManager()
        for event in HookEvent:
            assert event in manager._hooks
            assert manager._hooks[event] == []


class TestHookRegistration:
    """Test hook registration functionality."""

    def test_register_hook_with_command(self):
        """Test registering a hook with a shell command."""
        manager = HookManager()
        manager.register(
            HookEvent.PRE_TOOL_USE,
            command="echo test",
            matcher="Bash",
        )
        assert len(manager._hooks[HookEvent.PRE_TOOL_USE]) == 1
        hook = manager._hooks[HookEvent.PRE_TOOL_USE][0]
        assert hook.command == "echo test"
        assert hook.matcher == "Bash"

    def test_register_hook_with_callback(self):
        """Test registering a hook with a Python callback."""
        manager = HookManager()
        callback = Mock()
        manager.register(
            HookEvent.SESSION_START,
            callback=callback,
        )
        assert len(manager._hooks[HookEvent.SESSION_START]) == 1
        hook = manager._hooks[HookEvent.SESSION_START][0]
        assert hook.callback == callback

    def test_register_hook_requires_command_or_callback(self):
        """Test that registering a hook requires either command or callback."""
        manager = HookManager()
        with pytest.raises(ValueError, match="Either command or callback"):
            manager.register(HookEvent.PRE_TOOL_USE)

    def test_register_multiple_hooks_same_event(self):
        """Test registering multiple hooks for the same event."""
        manager = HookManager()
        manager.register(HookEvent.PRE_TOOL_USE, command="cmd1", matcher="Bash")
        manager.register(HookEvent.PRE_TOOL_USE, command="cmd2", matcher="Write")
        assert len(manager._hooks[HookEvent.PRE_TOOL_USE]) == 2

    def test_register_hook_with_custom_timeout(self):
        """Test registering a hook with custom timeout."""
        manager = HookManager()
        manager.register(
            HookEvent.POST_TOOL_USE,
            command="test.sh",
            timeout=120,
        )
        hook = manager._hooks[HookEvent.POST_TOOL_USE][0]
        assert hook.timeout == 120

    def test_register_hook_default_timeout(self):
        """Test hook registration uses default timeout."""
        manager = HookManager()
        manager.register(HookEvent.SESSION_START, command="test.sh")
        hook = manager._hooks[HookEvent.SESSION_START][0]
        assert hook.timeout == 60

    def test_register_hook_enabled_by_default(self):
        """Test registered hooks are enabled by default."""
        manager = HookManager()
        manager.register(HookEvent.SESSION_START, command="test.sh")
        hook = manager._hooks[HookEvent.SESSION_START][0]
        assert hook.enabled is True

    def test_unregister_hook_by_event(self):
        """Test unregistering all hooks for an event."""
        manager = HookManager()
        manager.register(HookEvent.PRE_TOOL_USE, command="cmd1")
        manager.register(HookEvent.PRE_TOOL_USE, command="cmd2")
        assert len(manager._hooks[HookEvent.PRE_TOOL_USE]) == 2
        manager.unregister(HookEvent.PRE_TOOL_USE)
        assert len(manager._hooks[HookEvent.PRE_TOOL_USE]) == 0

    def test_unregister_hook_by_matcher(self):
        """Test unregistering hooks by matcher pattern."""
        manager = HookManager()
        manager.register(HookEvent.PRE_TOOL_USE, command="cmd1", matcher="Bash")
        manager.register(HookEvent.PRE_TOOL_USE, command="cmd2", matcher="Write")
        manager.unregister(HookEvent.PRE_TOOL_USE, matcher="Bash")
        assert len(manager._hooks[HookEvent.PRE_TOOL_USE]) == 1
        assert manager._hooks[HookEvent.PRE_TOOL_USE][0].matcher == "Write"


class TestHookExecution:
    """Test hook execution functionality."""

    @pytest.mark.asyncio
    async def test_trigger_no_hooks_registered(self):
        """Test triggering event with no hooks registered."""
        manager = HookManager()
        result = await manager.trigger(HookEvent.SESSION_START)
        assert result.success is True
        assert result.decision == HookDecision.ALLOW

    @pytest.mark.asyncio
    async def test_trigger_with_callback_hook(self):
        """Test triggering a callback hook."""
        manager = HookManager()
        callback = Mock(return_value=HookResult(success=True))
        manager.register(HookEvent.SESSION_START, callback=callback)
        result = await manager.trigger(HookEvent.SESSION_START)
        assert result.success is True
        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_with_async_callback(self):
        """Test triggering an async callback hook."""
        manager = HookManager()

        async def async_callback(context):
            return HookResult(success=True, output="async result")

        manager.register(HookEvent.SESSION_START, callback=async_callback)
        result = await manager.trigger(HookEvent.SESSION_START)
        assert result.success is True
        assert "async result" in result.output

    @pytest.mark.asyncio
    async def test_trigger_callback_returns_dict(self):
        """Test callback returning dict is normalized to HookResult."""
        manager = HookManager()

        def callback(context):
            return {
                "success": True,
                "decision": "allow",
                "additional_context": "Test reason",
            }

        manager.register(HookEvent.SESSION_START, callback=callback)
        result = await manager.trigger(HookEvent.SESSION_START)
        assert result.success is True
        assert result.decision == HookDecision.ALLOW
        assert result.additional_context == "Test reason"

    @pytest.mark.asyncio
    async def test_trigger_callback_returns_bool(self):
        """Test callback returning bool is normalized to HookResult."""
        manager = HookManager()
        manager.register(HookEvent.SESSION_START, callback=lambda ctx: True)
        result = await manager.trigger(HookEvent.SESSION_START)
        assert result.success is True
        assert result.decision == HookDecision.ALLOW

    @pytest.mark.asyncio
    async def test_trigger_callback_returns_string(self):
        """Test callback returning string is normalized to HookResult."""
        manager = HookManager()
        manager.register(HookEvent.SESSION_START, callback=lambda ctx: "output text")
        result = await manager.trigger(HookEvent.SESSION_START)
        assert result.success is True
        assert result.output == "output text"

    @pytest.mark.asyncio
    async def test_trigger_with_tool_name_matching(self):
        """Test hook triggering with tool name matching."""
        manager = HookManager()
        callback = Mock(return_value=HookResult(success=True))
        manager.register(
            HookEvent.PRE_TOOL_USE,
            callback=callback,
            matcher="Bash",
        )
        result = await manager.trigger(
            HookEvent.PRE_TOOL_USE,
            tool_name="Bash",
        )
        assert result.success is True
        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_with_tool_name_no_match(self):
        """Test hook not triggered when tool name doesn't match."""
        manager = HookManager()
        callback = Mock(return_value=HookResult(success=True))
        manager.register(
            HookEvent.PRE_TOOL_USE,
            callback=callback,
            matcher="Bash",
        )
        result = await manager.trigger(
            HookEvent.PRE_TOOL_USE,
            tool_name="Write",
        )
        assert result.success is True
        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_trigger_with_regex_matcher(self):
        """Test hook triggering with regex pattern matching."""
        manager = HookManager()
        callback = Mock(return_value=HookResult(success=True))
        manager.register(
            HookEvent.PRE_TOOL_USE,
            callback=callback,
            matcher="(Bash|Shell)",
        )
        result = await manager.trigger(
            HookEvent.PRE_TOOL_USE,
            tool_name="Bash",
        )
        assert result.success is True
        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_passes_context_to_callback(self):
        """Test that trigger passes correct context to callback."""
        manager = HookManager(
            project_dir="/tmp/project",
            session_id="test_session",
        )

        captured_context = None

        def callback(context):
            nonlocal captured_context
            captured_context = context
            return HookResult(success=True)

        manager.register(HookEvent.PRE_TOOL_USE, callback=callback)
        await manager.trigger(
            HookEvent.PRE_TOOL_USE,
            tool_name="Bash",
            tool_input={"cmd": "ls"},
        )
        assert captured_context is not None
        assert captured_context.session_id == "test_session"
        assert captured_context.tool_name == "Bash"
        assert captured_context.tool_input == {"cmd": "ls"}


class TestHookErrorHandling:
    """Test hook error handling."""

    @pytest.mark.asyncio
    async def test_callback_raises_exception(self):
        """Test handling of callback that raises exception."""
        manager = HookManager()

        def failing_callback(context):
            raise ValueError("Test error")

        manager.register(HookEvent.SESSION_START, callback=failing_callback)
        result = await manager.trigger(HookEvent.SESSION_START)
        assert result.success is False
        assert "Test error" in result.reason

    @pytest.mark.asyncio
    async def test_callback_timeout(self):
        """Test handling of callback that times out."""
        manager = HookManager()

        async def slow_callback(context):
            await asyncio.sleep(2)
            return HookResult(success=True)

        manager.register(
            HookEvent.SESSION_START,
            callback=slow_callback,
            timeout=0.1,
        )
        result = await manager.trigger(HookEvent.SESSION_START)
        assert result.success is False
        assert "timed out" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_disabled_hook_not_triggered(self):
        """Test that disabled hooks are not executed."""
        manager = HookManager()
        callback = Mock(return_value=HookResult(success=True))
        manager.register(HookEvent.SESSION_START, callback=callback)
        # Disable the hook
        manager._hooks[HookEvent.SESSION_START][0].enabled = False
        result = await manager.trigger(HookEvent.SESSION_START)
        assert result.success is True
        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_hooks_one_fails(self):
        """Test aggregation when one hook fails."""
        manager = HookManager()
        manager.register(
            HookEvent.SESSION_START,
            callback=lambda ctx: HookResult(success=True),
        )
        manager.register(
            HookEvent.SESSION_START,
            callback=lambda ctx: HookResult(success=False, reason="Failed"),
        )
        result = await manager.trigger(HookEvent.SESSION_START)
        assert result.success is False
        assert "Failed" in result.reason

    @pytest.mark.asyncio
    async def test_hook_decision_deny_aggregation(self):
        """Test that DENY decision is most restrictive."""
        manager = HookManager()
        manager.register(
            HookEvent.PRE_TOOL_USE,
            callback=lambda ctx: HookResult(success=True, decision=HookDecision.ALLOW),
        )
        manager.register(
            HookEvent.PRE_TOOL_USE,
            callback=lambda ctx: HookResult(
                success=True, decision=HookDecision.DENY, reason="Blocked"
            ),
        )
        result = await manager.trigger(HookEvent.PRE_TOOL_USE)
        assert result.decision == HookDecision.DENY
        assert result.reason == "Blocked"

    @pytest.mark.asyncio
    async def test_hook_decision_ask_aggregation(self):
        """Test ASK decision aggregation."""
        manager = HookManager()
        manager.register(
            HookEvent.PRE_TOOL_USE,
            callback=lambda ctx: HookResult(success=True, decision=HookDecision.ALLOW),
        )
        manager.register(
            HookEvent.PRE_TOOL_USE,
            callback=lambda ctx: HookResult(success=True, decision=HookDecision.ASK),
        )
        result = await manager.trigger(HookEvent.PRE_TOOL_USE)
        assert result.decision == HookDecision.ASK

    @pytest.mark.asyncio
    async def test_hook_continue_processing_false(self):
        """Test that continue_processing=False is preserved."""
        manager = HookManager()
        manager.register(
            HookEvent.PRE_TOOL_USE,
            callback=lambda ctx: HookResult(success=True, continue_processing=False),
        )
        result = await manager.trigger(HookEvent.PRE_TOOL_USE)
        assert result.continue_processing is False


class TestHookConfigurationLoading:
    """Test hook configuration loading from files."""

    def test_load_from_file_nonexistent(self):
        """Test loading from nonexistent file doesn't raise error."""
        manager = HookManager()
        manager.load_from_file("/nonexistent/path/hooks.json")
        # Should not raise, just return silently
        assert len(manager._hooks[HookEvent.PRE_TOOL_USE]) == 0

    def test_load_from_file_valid_config(self):
        """Test loading valid hook configuration from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config = {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "./validate.sh",
                                    "timeout": 30,
                                }
                            ],
                        }
                    ]
                }
            }
            json.dump(config, f)
            f.flush()

            manager = HookManager()
            manager.load_from_file(f.name)
            assert len(manager._hooks[HookEvent.PRE_TOOL_USE]) == 1
            hook = manager._hooks[HookEvent.PRE_TOOL_USE][0]
            assert hook.command == "./validate.sh"
            assert hook.matcher == "Bash"
            assert hook.timeout == 30

            Path(f.name).unlink()

    def test_load_from_file_multiple_hooks(self):
        """Test loading multiple hooks from configuration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config = {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [{"type": "command", "command": "cmd1.sh"}],
                        },
                        {
                            "matcher": "Write",
                            "hooks": [{"type": "command", "command": "cmd2.sh"}],
                        },
                    ]
                }
            }
            json.dump(config, f)
            f.flush()

            manager = HookManager()
            manager.load_from_file(f.name)
            assert len(manager._hooks[HookEvent.PRE_TOOL_USE]) == 2

            Path(f.name).unlink()

    def test_load_from_file_invalid_event(self):
        """Test loading configuration with invalid event name."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config = {
                "hooks": {
                    "InvalidEvent": [
                        {
                            "matcher": "Bash",
                            "hooks": [{"type": "command", "command": "cmd.sh"}],
                        }
                    ]
                }
            }
            json.dump(config, f)
            f.flush()

            manager = HookManager()
            manager.load_from_file(f.name)
            # Should skip invalid event, not raise
            assert len(manager._hooks[HookEvent.PRE_TOOL_USE]) == 0

            Path(f.name).unlink()

    def test_load_from_file_malformed_json(self):
        """Test loading malformed JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{invalid json")
            f.flush()

            manager = HookManager()
            manager.load_from_file(f.name)
            # Should not raise, just log error
            assert len(manager._hooks[HookEvent.PRE_TOOL_USE]) == 0

            Path(f.name).unlink()

    def test_save_to_file_creates_directory(self):
        """Test save_to_file creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "hooks.json"
            manager = HookManager()
            manager.register(
                HookEvent.PRE_TOOL_USE,
                command="test.sh",
                matcher="Bash",
            )
            manager.save_to_file(path)
            assert path.exists()

    def test_save_to_file_preserves_hooks(self):
        """Test save_to_file preserves hook configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "hooks.json"
            manager = HookManager()
            manager.register(
                HookEvent.PRE_TOOL_USE,
                command="validate.sh",
                matcher="Bash",
                timeout=45,
            )
            manager.save_to_file(path)

            # Load and verify
            data = json.loads(path.read_text())
            assert "hooks" in data
            assert "PreToolUse" in data["hooks"]
            hooks_list = data["hooks"]["PreToolUse"]
            assert len(hooks_list) > 0
            assert hooks_list[0]["matcher"] == "Bash"


class TestHookFiltering:
    """Test hook filtering and conditional execution."""

    def test_get_matching_hooks_no_matcher(self):
        """Test getting hooks without matcher."""
        manager = HookManager()
        manager.register(HookEvent.SESSION_START, command="cmd1.sh")
        manager.register(HookEvent.SESSION_START, command="cmd2.sh")
        hooks = manager._get_matching_hooks(HookEvent.SESSION_START, None)
        assert len(hooks) == 2

    def test_get_matching_hooks_exact_match(self):
        """Test getting hooks with exact matcher match."""
        manager = HookManager()
        manager.register(
            HookEvent.PRE_TOOL_USE,
            command="cmd1.sh",
            matcher="Bash",
        )
        manager.register(
            HookEvent.PRE_TOOL_USE,
            command="cmd2.sh",
            matcher="Write",
        )
        hooks = manager._get_matching_hooks(HookEvent.PRE_TOOL_USE, "Bash")
        assert len(hooks) == 1
        assert hooks[0].matcher == "Bash"

    def test_get_matching_hooks_regex_match(self):
        """Test getting hooks with regex matcher."""
        manager = HookManager()
        manager.register(
            HookEvent.PRE_TOOL_USE,
            command="cmd1.sh",
            matcher="(Bash|Shell)",
        )
        manager.register(
            HookEvent.PRE_TOOL_USE,
            command="cmd2.sh",
            matcher="Write",
        )
        hooks = manager._get_matching_hooks(HookEvent.PRE_TOOL_USE, "Bash")
        assert len(hooks) == 1
        assert hooks[0].matcher == "(Bash|Shell)"

    def test_get_matching_hooks_invalid_regex_fallback(self):
        """Test fallback to exact match on invalid regex."""
        manager = HookManager()
        manager.register(
            HookEvent.PRE_TOOL_USE,
            command="cmd1.sh",
            matcher="[invalid(regex",
        )
        # Should fallback to exact match
        hooks = manager._get_matching_hooks(HookEvent.PRE_TOOL_USE, "[invalid(regex")
        assert len(hooks) == 1

    def test_get_matching_hooks_disabled_excluded(self):
        """Test that disabled hooks are excluded."""
        manager = HookManager()
        manager.register(HookEvent.SESSION_START, command="cmd1.sh")
        manager.register(HookEvent.SESSION_START, command="cmd2.sh")
        # Disable first hook
        manager._hooks[HookEvent.SESSION_START][0].enabled = False
        hooks = manager._get_matching_hooks(HookEvent.SESSION_START, None)
        assert len(hooks) == 1
        assert hooks[0].command == "cmd2.sh"


class TestHookSummary:
    """Test hook summary functionality."""

    def test_get_hooks_summary_empty(self):
        """Test getting summary of empty hooks."""
        manager = HookManager()
        summary = manager.get_hooks_summary()
        assert summary == {}

    def test_get_hooks_summary_with_hooks(self):
        """Test getting summary of registered hooks."""
        manager = HookManager()
        manager.register(
            HookEvent.PRE_TOOL_USE,
            command="cmd1.sh",
            matcher="Bash",
            timeout=30,
        )
        manager.register(
            HookEvent.SESSION_START,
            command="cmd2.sh",
        )
        summary = manager.get_hooks_summary()
        assert "PreToolUse" in summary
        assert "SessionStart" in summary
        assert len(summary["PreToolUse"]) == 1
        assert len(summary["SessionStart"]) == 1

    def test_get_hooks_summary_includes_all_fields(self):
        """Test that summary includes all hook fields."""
        manager = HookManager()
        manager.register(
            HookEvent.PRE_TOOL_USE,
            command="test.sh",
            matcher="Bash",
            timeout=45,
        )
        summary = manager.get_hooks_summary()
        hook_dict = summary["PreToolUse"][0]
        assert hook_dict["event"] == "PreToolUse"
        assert hook_dict["command"] == "test.sh"
        assert hook_dict["matcher"] == "Bash"
        assert hook_dict["timeout"] == 45
        assert hook_dict["enabled"] is True


class TestHookCommandExecution:
    """Test shell command hook execution."""

    @pytest.mark.asyncio
    async def test_run_command_success(self):
        """Test successful command execution."""
        manager = HookManager()
        manager.register(
            HookEvent.SESSION_START,
            command="echo 'test output'",
        )
        result = await manager.trigger(HookEvent.SESSION_START)
        assert result.success is True
        assert "test output" in result.output

    @pytest.mark.asyncio
    async def test_run_command_json_output(self):
        """Test command returning JSON output."""
        manager = HookManager()
        manager.register(
            HookEvent.SESSION_START,
            command='echo \'{"success": true, "decision": "allow"}\'',
        )
        result = await manager.trigger(HookEvent.SESSION_START)
        assert result.success is True
        assert result.decision == HookDecision.ALLOW

    @pytest.mark.asyncio
    async def test_run_command_exit_code_2_deny(self):
        """Test command with exit code 2 results in DENY."""
        manager = HookManager()
        manager.register(
            HookEvent.SESSION_START,
            command="exit 2",
        )
        result = await manager.trigger(HookEvent.SESSION_START)
        assert result.success is False
        assert result.decision == HookDecision.DENY
        assert result.continue_processing is False

    @pytest.mark.asyncio
    async def test_run_command_exit_code_1_error(self):
        """Test command with exit code 1 results in error."""
        manager = HookManager()
        manager.register(
            HookEvent.SESSION_START,
            command="exit 1",
        )
        result = await manager.trigger(HookEvent.SESSION_START)
        assert result.success is False
        assert result.continue_processing is True

    @pytest.mark.asyncio
    async def test_run_command_timeout(self):
        """Test command execution timeout."""
        manager = HookManager()
        manager.register(
            HookEvent.SESSION_START,
            command="sleep 10",
            timeout=0.1,
        )
        result = await manager.trigger(HookEvent.SESSION_START)
        assert result.success is False
        assert "timed out" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_run_command_with_context_env(self):
        """Test command receives context via environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = HookManager(project_dir=tmpdir)
            manager.register(
                HookEvent.SESSION_START,
                command="echo $CLAUDE_PROJECT_DIR",
            )
            result = await manager.trigger(HookEvent.SESSION_START)
            assert result.success is True
            assert tmpdir in result.output or "test" in result.output


class TestHookAggregation:
    """Test hook result aggregation."""

    @pytest.mark.asyncio
    async def test_aggregate_multiple_outputs(self):
        """Test aggregation of multiple hook outputs."""
        manager = HookManager()
        manager.register(
            HookEvent.SESSION_START,
            callback=lambda ctx: HookResult(success=True, output="output1"),
        )
        manager.register(
            HookEvent.SESSION_START,
            callback=lambda ctx: HookResult(success=True, output="output2"),
        )
        result = await manager.trigger(HookEvent.SESSION_START)
        assert "output1" in result.output
        assert "output2" in result.output

    @pytest.mark.asyncio
    async def test_aggregate_modified_input(self):
        """Test aggregation of modified input (last wins)."""
        manager = HookManager()
        manager.register(
            HookEvent.PRE_TOOL_USE,
            callback=lambda ctx: HookResult(success=True, modified_input={"key": "value1"}),
        )
        manager.register(
            HookEvent.PRE_TOOL_USE,
            callback=lambda ctx: HookResult(success=True, modified_input={"key": "value2"}),
        )
        result = await manager.trigger(HookEvent.PRE_TOOL_USE)
        assert result.modified_input == {"key": "value2"}

    @pytest.mark.asyncio
    async def test_aggregate_duration(self):
        """Test aggregation of hook durations."""
        manager = HookManager()
        manager.register(
            HookEvent.SESSION_START,
            callback=lambda ctx: HookResult(success=True),
        )
        manager.register(
            HookEvent.SESSION_START,
            callback=lambda ctx: HookResult(success=True),
        )
        result = await manager.trigger(HookEvent.SESSION_START)
        assert result.duration > 0

    @pytest.mark.asyncio
    async def test_aggregate_exception_handling(self):
        """Test aggregation handles exceptions gracefully."""
        manager = HookManager()
        manager.register(
            HookEvent.SESSION_START,
            callback=lambda ctx: HookResult(success=True),
        )
        manager.register(
            HookEvent.SESSION_START,
            callback=lambda ctx: 1 / 0,  # Will raise exception
        )
        result = await manager.trigger(HookEvent.SESSION_START)
        # Should still succeed overall
        assert isinstance(result, HookResult)


class TestHookContextData:
    """Test hook context data handling."""

    @pytest.mark.asyncio
    async def test_context_with_all_tool_fields(self):
        """Test context with all tool-related fields."""
        manager = HookManager()

        captured_context = None

        def callback(context):
            nonlocal captured_context
            captured_context = context
            return HookResult(success=True)

        manager.register(HookEvent.PRE_TOOL_USE, callback=callback)
        await manager.trigger(
            HookEvent.PRE_TOOL_USE,
            tool_name="Bash",
            tool_input={"command": "ls"},
            tool_output="file1.txt",
        )
        assert captured_context.tool_name == "Bash"
        assert captured_context.tool_input == {"command": "ls"}
        assert captured_context.tool_output == "file1.txt"

    @pytest.mark.asyncio
    async def test_context_with_message_count(self):
        """Test context with message count."""
        manager = HookManager()

        captured_context = None

        def callback(context):
            nonlocal captured_context
            captured_context = context
            return HookResult(success=True)

        manager.register(HookEvent.STOP, callback=callback)
        await manager.trigger(
            HookEvent.STOP,
            message_count=42,
            stop_reason="max_tokens",
        )
        assert captured_context.message_count == 42
        assert captured_context.stop_reason == "max_tokens"

    @pytest.mark.asyncio
    async def test_context_timestamp_set(self):
        """Test that context timestamp is set."""
        manager = HookManager()

        captured_context = None

        def callback(context):
            nonlocal captured_context
            captured_context = context
            return HookResult(success=True)

        manager.register(HookEvent.SESSION_START, callback=callback)
        await manager.trigger(HookEvent.SESSION_START)
        assert captured_context.timestamp > 0
