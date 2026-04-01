"""Comprehensive tests for hooks.py"""

import time

from src.core.hooks import HookContext, HookDecision, HookDefinition, HookEvent, HookResult


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
