"""Comprehensive tests for hooks.py"""

import asyncio
import json
import time

import pytest

from src.core.hooks import (
    HookContext,
    HookDecision,
    HookDefinition,
    HookEvent,
    HookManager,
    HookResult,
)


class TestHookEvent:
    """Tests for HookEvent enum."""

    def test_all_events_defined(self):
        """Test that all hook events are defined."""
        assert HookEvent.SESSION_START.value == "SessionStart"
        assert HookEvent.SESSION_END.value == "SessionEnd"
        assert HookEvent.USER_PROMPT_SUBMIT.value == "UserPromptSubmit"
        assert HookEvent.PRE_TOOL_USE.value == "PreToolUse"
        assert HookEvent.POST_TOOL_USE.value == "PostToolUse"
        assert HookEvent.STOP.value == "Stop"
        assert HookEvent.PRE_COMPACT.value == "PreCompact"
        assert HookEvent.NOTIFICATION.value == "Notification"

    def test_event_count(self):
        """Test expected number of events."""
        assert len(HookEvent) == 8


class TestHookDecision:
    """Tests for HookDecision enum."""

    def test_all_decisions_defined(self):
        """Test that all decisions are defined."""
        assert HookDecision.ALLOW.value == "allow"
        assert HookDecision.DENY.value == "deny"
        assert HookDecision.ASK.value == "ask"
        assert HookDecision.MODIFY.value == "modify"

    def test_decision_count(self):
        """Test expected number of decisions."""
        assert len(HookDecision) == 4


class TestHookResult:
    """Tests for HookResult."""

    def test_creation_minimal(self):
        """Test creating minimal hook result."""
        result = HookResult(success=True)
        assert result.success is True
        assert result.decision == HookDecision.ALLOW
        assert result.reason == ""
        assert result.modified_input is None
        assert result.continue_processing is True

    def test_creation_with_deny(self):
        """Test creating result with deny decision."""
        result = HookResult(success=True, decision=HookDecision.DENY, reason="Not allowed")
        assert result.decision == HookDecision.DENY
        assert result.reason == "Not allowed"

    def test_creation_with_modification(self):
        """Test creating result with modifications."""
        modified = {"key": "new_value"}
        result = HookResult(
            success=True,
            decision=HookDecision.MODIFY,
            modified_input=modified,
            additional_context="Modified input",
        )
        assert result.decision == HookDecision.MODIFY
        assert result.modified_input == modified
        assert result.additional_context == "Modified input"

    def test_creation_with_output(self):
        """Test creating result with output."""
        result = HookResult(success=True, output="Hook output", duration=1.5)
        assert result.output == "Hook output"
        assert result.duration == 1.5

    def test_failed_result(self):
        """Test creating failed result."""
        result = HookResult(success=False, reason="Hook failed", continue_processing=False)
        assert result.success is False
        assert result.continue_processing is False


class TestHookDefinition:
    """Tests for HookDefinition."""

    def test_creation_with_command(self):
        """Test creating hook with shell command."""
        hook = HookDefinition(event=HookEvent.PRE_TOOL_USE, command="echo 'test'", timeout=30)
        assert hook.event == HookEvent.PRE_TOOL_USE
        assert hook.command == "echo 'test'"
        assert hook.timeout == 30
        assert hook.enabled is True

    def test_creation_with_matcher(self):
        """Test creating hook with matcher."""
        hook = HookDefinition(
            event=HookEvent.PRE_TOOL_USE, matcher="write_.*", command="validate.sh"
        )
        assert hook.matcher == "write_.*"

    def test_creation_with_callback(self):
        """Test creating hook with Python callback."""

        def my_callback(ctx):
            return HookResult(success=True)

        hook = HookDefinition(event=HookEvent.SESSION_START, callback=my_callback)
        assert hook.callback == my_callback
        assert hook.command is None

    def test_to_dict(self):
        """Test converting hook to dictionary."""
        hook = HookDefinition(
            event=HookEvent.POST_TOOL_USE,
            matcher="test_tool",
            command="log.sh",
            timeout=45,
            enabled=False,
        )
        data = hook.to_dict()
        assert data["event"] == "PostToolUse"
        assert data["matcher"] == "test_tool"
        assert data["command"] == "log.sh"
        assert data["timeout"] == 45
        assert data["enabled"] is False

    def test_to_dict_excludes_callback(self):
        """Test that to_dict doesn't include callback."""

        def callback(ctx):
            pass

        hook = HookDefinition(event=HookEvent.STOP, callback=callback)
        data = hook.to_dict()
        assert "callback" not in data

    def test_default_timeout(self):
        """Test default timeout value."""
        hook = HookDefinition(event=HookEvent.SESSION_END)
        assert hook.timeout == 60

    def test_disabled_hook(self):
        """Test creating disabled hook."""
        hook = HookDefinition(event=HookEvent.PRE_COMPACT, command="test.sh", enabled=False)
        assert hook.enabled is False


class TestHookContext:
    """Tests for HookContext."""

    def test_creation_basic(self):
        """Test creating basic hook context."""
        ctx = HookContext(
            event=HookEvent.SESSION_START,
            session_id="session_123",
            project_dir="/test/project",
            permission_mode="ask",
        )
        assert ctx.event == HookEvent.SESSION_START
        assert ctx.session_id == "session_123"
        assert ctx.project_dir == "/test/project"
        assert ctx.permission_mode == "ask"
        assert ctx.tool_name is None

    def test_creation_with_tool_info(self):
        """Test creating context with tool information."""
        tool_input = {"file": "test.py", "content": "code"}
        ctx = HookContext(
            event=HookEvent.PRE_TOOL_USE,
            session_id="session_456",
            project_dir="/project",
            permission_mode="allow",
            tool_name="write",
            tool_input=tool_input,
        )
        assert ctx.tool_name == "write"
        assert ctx.tool_input == tool_input

    def test_creation_with_tool_output(self):
        """Test creating context with tool output."""
        ctx = HookContext(
            event=HookEvent.POST_TOOL_USE,
            session_id="session_789",
            project_dir="/project",
            permission_mode="allow",
            tool_name="read",
            tool_output="file contents",
        )
        assert ctx.tool_output == "file contents"

    def test_timestamp_auto_generated(self):
        """Test that timestamp is auto-generated."""
        before = time.time()
        ctx = HookContext(
            event=HookEvent.NOTIFICATION,
            session_id="test",
            project_dir="/test",
            permission_mode="allow",
        )
        after = time.time()
        assert before <= ctx.timestamp <= after

    def test_all_optional_fields_none(self):
        """Test that optional fields default to None."""
        ctx = HookContext(
            event=HookEvent.SESSION_END,
            session_id="test",
            project_dir="/test",
            permission_mode="deny",
        )
        assert ctx.tool_name is None
        assert ctx.tool_input is None
        assert ctx.tool_output is None


class TestHookIntegration:
    """Integration tests for hook system."""

    def test_hook_workflow_allow(self):
        """Test complete hook workflow with allow decision."""
        # Define hook
        hook = HookDefinition(
            event=HookEvent.PRE_TOOL_USE, matcher="write", command="validate.sh"
        )

        # Create context
        ctx = HookContext(
            event=HookEvent.PRE_TOOL_USE,
            session_id="test_session",
            project_dir="/project",
            permission_mode="ask",
            tool_name="write",
            tool_input={"path": "test.py"},
        )

        # Create result
        result = HookResult(success=True, decision=HookDecision.ALLOW)

        assert hook.event == ctx.event
        assert result.decision == HookDecision.ALLOW
        assert result.continue_processing is True

    def test_hook_workflow_deny(self):
        """Test hook workflow with deny decision."""
        HookDefinition(event=HookEvent.PRE_TOOL_USE, matcher="write")

        HookContext(
            event=HookEvent.PRE_TOOL_USE,
            session_id="test",
            project_dir="/project",
            permission_mode="ask",
            tool_name="write",
        )

        result = HookResult(success=True, decision=HookDecision.DENY, reason="Deletion not allowed")

        assert result.decision == HookDecision.DENY
        assert result.reason == "Deletion not allowed"

    def test_hook_workflow_modify(self):
        """Test hook workflow with modification."""
        HookDefinition(event=HookEvent.PRE_TOOL_USE, matcher=".*")

        original_input = {"path": "test.py", "content": "old"}
        modified_input = {"path": "test.py", "content": "new"}

        ctx = HookContext(
            event=HookEvent.PRE_TOOL_USE,
            session_id="test",
            project_dir="/project",
            permission_mode="allow",
            tool_name="write",
            tool_input=original_input,
        )

        result = HookResult(
            success=True,
            decision=HookDecision.MODIFY,
            modified_input=modified_input,
            additional_context="Content sanitized",
        )

        assert result.modified_input != ctx.tool_input
        assert result.modified_input == modified_input


class TestHookContextToDict:
    def test_to_dict_includes_tool_fields(self):
        ctx = HookContext(
            event=HookEvent.PRE_TOOL_USE,
            session_id="s1",
            project_dir="/p",
            permission_mode="allow",
            tool_name="Bash",
            tool_input={"cmd": "ls"},
            tool_output="output",
            user_prompt="prompt",
            message_count=5,
            stop_reason="end_turn",
        )
        d = ctx.to_dict()
        assert d["tool_name"] == "Bash"
        assert d["tool_input"] == {"cmd": "ls"}
        assert d["tool_output"] == "output"
        assert d["prompt"] == "prompt"
        assert d["message_count"] == 5
        assert d["stop_reason"] == "end_turn"


class TestHookManager:
    def test_register_requires_command_or_callback(self):
        mgr = HookManager()
        with pytest.raises(ValueError):
            mgr.register(HookEvent.SESSION_START)

    def test_unregister_specific_matcher(self):
        mgr = HookManager()
        mgr.register(HookEvent.PRE_TOOL_USE, command="validate.sh", matcher="Bash")
        mgr.register(HookEvent.PRE_TOOL_USE, command="other.sh", matcher="Write")
        mgr.unregister(HookEvent.PRE_TOOL_USE, matcher="Bash")
        assert len(mgr._hooks[HookEvent.PRE_TOOL_USE]) == 1

    def test_unregister_all(self):
        mgr = HookManager()
        mgr.register(HookEvent.PRE_TOOL_USE, command="a.sh", matcher="Bash")
        mgr.unregister(HookEvent.PRE_TOOL_USE)
        assert len(mgr._hooks[HookEvent.PRE_TOOL_USE]) == 0

    @pytest.mark.asyncio
    async def test_trigger_no_hooks_returns_allow(self):
        mgr = HookManager()
        result = await mgr.trigger(HookEvent.SESSION_START)
        assert result.success is True
        assert result.decision == HookDecision.ALLOW

    @pytest.mark.asyncio
    async def test_trigger_callback_returns_bool(self):
        mgr = HookManager()

        def deny_all(ctx):
            return False

        mgr.register(HookEvent.PRE_TOOL_USE, callback=deny_all, matcher="Bash")
        result = await mgr.trigger(HookEvent.PRE_TOOL_USE, tool_name="Bash")
        assert result.decision == HookDecision.DENY

    @pytest.mark.asyncio
    async def test_trigger_callback_returns_dict(self):
        mgr = HookManager()

        def modify_hook(ctx):
            return {
                "success": True,
                "decision": "modify",
                "modified_input": {"cmd": "sanitized"},
                "reason": "Sanitized",
            }

        mgr.register(HookEvent.PRE_TOOL_USE, callback=modify_hook, matcher="Bash")
        result = await mgr.trigger(HookEvent.PRE_TOOL_USE, tool_name="Bash", tool_input={"cmd": "rm -rf"})
        assert result.modified_input == {"cmd": "sanitized"}

    @pytest.mark.asyncio
    async def test_trigger_async_callback(self):
        mgr = HookManager()

        async def async_hook(ctx):
            return HookResult(success=True, output="async_result")

        mgr.register(HookEvent.POST_TOOL_USE, callback=async_hook, matcher="Write")
        result = await mgr.trigger(HookEvent.POST_TOOL_USE, tool_name="Write")
        assert result.success is True
        assert "async_result" in result.output

    @pytest.mark.asyncio
    async def test_trigger_callback_exception(self):
        mgr = HookManager()

        def bad_hook(ctx):
            raise RuntimeError("Hook crashed")

        mgr.register(HookEvent.SESSION_START, callback=bad_hook)
        result = await mgr.trigger(HookEvent.SESSION_START)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_trigger_disabled_hook_skipped(self):
        mgr = HookManager()
        call_count = 0

        def counting_hook(ctx):
            nonlocal call_count
            call_count += 1
            return True

        mgr.register(HookEvent.SESSION_START, callback=counting_hook)
        mgr._hooks[HookEvent.SESSION_START][0].enabled = False
        await mgr.trigger(HookEvent.SESSION_START)
        assert call_count == 0

    @pytest.mark.asyncio
    async def test_trigger_matcher_no_tool_name_skips(self):
        mgr = HookManager()

        def hook(ctx):
            return True

        mgr.register(HookEvent.PRE_TOOL_USE, callback=hook, matcher="Bash")
        result = await mgr.trigger(HookEvent.PRE_TOOL_USE, tool_name=None)
        assert result.success is True
        assert result.decision == HookDecision.ALLOW

    @pytest.mark.asyncio
    async def test_trigger_invalid_regex_falls_back_to_exact(self):
        mgr = HookManager()

        def hook(ctx):
            return HookResult(success=True, decision=HookDecision.DENY)

        mgr.register(HookEvent.PRE_TOOL_USE, callback=hook, matcher="[invalid")
        result = await mgr.trigger(HookEvent.PRE_TOOL_USE, tool_name="[invalid")
        assert result.decision == HookDecision.DENY

    def test_save_and_load_hooks(self, tmp_path):
        mgr = HookManager()
        mgr.register(HookEvent.PRE_TOOL_USE, command="validate.sh", matcher="Bash")
        hooks_file = tmp_path / "hooks.json"
        mgr.save_to_file(hooks_file)
        assert hooks_file.exists()

        mgr2 = HookManager()
        mgr2.load_from_file(hooks_file)
        assert len(mgr2._hooks[HookEvent.PRE_TOOL_USE]) == 1

    def test_load_from_nonexistent_file(self, tmp_path):
        mgr = HookManager()
        mgr.load_from_file(tmp_path / "nope.json")

    def test_load_from_file_invalid_event(self, tmp_path):
        hooks_file = tmp_path / "hooks.json"
        hooks_file.write_text(json.dumps({"hooks": {"UnknownEvent": []}}))
        mgr = HookManager()
        mgr.load_from_file(hooks_file)

    @pytest.mark.asyncio
    async def test_run_hook_no_handler(self):
        mgr = HookManager()
        hook = HookDefinition(event=HookEvent.SESSION_START, command=None, callback=None, matcher=None)
        hook.command = None
        hook.callback = None
        result = await mgr._run_hook(hook, HookContext(
            event=HookEvent.SESSION_START, session_id="s", project_dir="/p", permission_mode="allow"
        ))
        assert result.success is False

    def test_get_hooks_summary(self):
        mgr = HookManager()
        mgr.register(HookEvent.SESSION_START, command="echo hi")
        summary = mgr.get_hooks_summary()
        assert "SessionStart" in summary
