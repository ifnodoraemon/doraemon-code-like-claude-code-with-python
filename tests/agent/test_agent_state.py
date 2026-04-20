"""
Tests for src/agent/state.py - Agent State Management

Comprehensive tests for AgentState: message management, tool tracking,
token estimation, compression, checkpoints, serialization.
"""

import pytest

from src.agent.state import AgentState
from src.agent.types import Message, ToolCall


class TestAgentStateInit:
    def test_default_initialization(self):
        state = AgentState()
        assert state.messages == []
        assert state.tool_history == []
        assert state.mode == "build"
        assert state.max_turns == 100
        assert state.turn_count == 0
        assert state.max_context_tokens == 128000
        assert state.estimated_tokens == 0
        assert state.max_messages == 50
        assert state.goal is None
        assert state.is_finished is False
        assert state.status == "idle"
        assert state.user_input is None
        assert state.last_response is None
        assert state.last_error is None
        assert state.metadata == {}

    def test_custom_initialization(self):
        state = AgentState(mode="plan", max_turns=50, max_context_tokens=64000)
        assert state.mode == "plan"
        assert state.max_turns == 50
        assert state.max_context_tokens == 64000

    def test_from_dict_basic(self):
        data = {
            "mode": "plan",
            "max_turns": 25,
            "turn_count": 10,
            "max_context_tokens": 32000,
            "goal": "Refactor code",
            "is_finished": True,
            "status": "finished",
            "metadata": {"key": "value"},
        }
        state = AgentState.from_dict(data)
        assert state.mode == "plan"
        assert state.max_turns == 25
        assert state.turn_count == 10
        assert state.max_context_tokens == 32000
        assert state.goal == "Refactor code"
        assert state.is_finished is True
        assert state.status == "finished"
        assert state.metadata == {"key": "value"}

    def test_from_dict_defaults(self):
        state = AgentState.from_dict({})
        assert state.mode == "build"
        assert state.max_turns == 100
        assert state.turn_count == 0
        assert state.is_finished is False
        assert state.status == "idle"


class TestMessageManagement:
    def test_add_message(self):
        state = AgentState()
        msg = Message(role="user", content="Hello")
        state.add_message(msg)
        assert len(state.messages) == 1
        assert state.messages[0].role == "user"
        assert state.messages[0].content == "Hello"

    def test_add_user_message(self):
        state = AgentState()
        state.add_user_message("Hello")
        assert len(state.messages) == 1
        assert state.messages[0].role == "user"
        assert state.messages[0].content == "Hello"
        assert state.user_input == "Hello"

    def test_add_assistant_message(self):
        state = AgentState()
        state.add_assistant_message(content="Hi there")
        assert len(state.messages) == 1
        assert state.messages[0].role == "assistant"
        assert state.messages[0].content == "Hi there"
        assert state.last_response == "Hi there"

    def test_add_assistant_message_with_tool_calls(self):
        state = AgentState()
        state.add_assistant_message(
            content="Using tool",
            tool_calls=[{"name": "read", "arguments": {}}],
        )
        assert state.messages[0].tool_calls == [{"name": "read", "arguments": {}}]

    def test_add_assistant_message_with_provider_items(self):
        state = AgentState()
        state.add_assistant_message(
            content="Thinking...",
            provider_items=[{"type": "reasoning", "id": "rs_1"}],
        )
        assert state.messages[0].provider_items == [{"type": "reasoning", "id": "rs_1"}]

    def test_add_assistant_message_with_thought(self):
        state = AgentState()
        state.add_assistant_message(
            content="Result",
            thought="I analyzed the file",
        )
        assert state.messages[0].thought == "I analyzed the file"

    def test_add_tool_result(self):
        state = AgentState()
        state.add_tool_result("call_123", "read", "file content")
        assert len(state.messages) == 1
        assert state.messages[0].role == "tool"
        assert state.messages[0].tool_call_id == "call_123"
        assert state.messages[0].name == "read"
        assert state.messages[0].content == "file content"

    def test_add_tool_call(self):
        state = AgentState()
        tc = ToolCall(id="1", name="read", arguments={"path": "/f"}, result="data")
        state.add_tool_call(tc)
        assert len(state.tool_history) == 1
        assert state.tool_history[0].name == "read"

    def test_multiple_messages(self):
        state = AgentState()
        state.add_user_message("Hello")
        state.add_assistant_message("Hi")
        state.add_user_message("How are you?")
        assert len(state.messages) == 3
        assert state.messages[0].role == "user"
        assert state.messages[1].role == "assistant"
        assert state.messages[2].role == "user"

    def test_add_assistant_message_updates_last_response(self):
        state = AgentState()
        state.add_assistant_message(content="first")
        assert state.last_response == "first"
        state.add_assistant_message(content="second")
        assert state.last_response == "second"

    def test_add_assistant_message_last_response_none_when_no_content(self):
        state = AgentState()
        state.add_assistant_message(content=None, tool_calls=[{"name": "read"}])
        assert state.last_response is None


class TestTokenEstimation:
    def test_update_token_estimate_empty(self):
        state = AgentState()
        state._update_token_estimate()
        assert state.estimated_tokens == 0

    def test_update_token_estimate_with_content(self):
        state = AgentState()
        state.add_user_message("a" * 100)
        assert state.estimated_tokens == 25

    def test_update_token_estimate_with_tool_calls(self):
        state = AgentState()
        state.add_assistant_message(
            content=None,
            tool_calls=[{"name": "read", "arguments": {"path": "/very/long/path"}}],
        )
        assert state.estimated_tokens > 0

    def test_needs_compression_below_threshold(self):
        state = AgentState(max_context_tokens=1000)
        state.estimated_tokens = 500
        assert state.needs_compression() is False

    def test_needs_compression_above_threshold(self):
        state = AgentState(max_context_tokens=1000)
        state.estimated_tokens = 950
        assert state.needs_compression() is True

    def test_needs_compression_at_exactly_90_percent(self):
        state = AgentState(max_context_tokens=1000)
        state.estimated_tokens = 900
        assert state.needs_compression() is False

    def test_needs_compression_just_above_90_percent(self):
        state = AgentState(max_context_tokens=1000)
        state.estimated_tokens = 901
        assert state.needs_compression() is True


class TestMessageLimitAndCompression:
    def test_message_limit_enforcement(self):
        state = AgentState(max_messages=5)
        for i in range(10):
            state.add_user_message(f"Message {i}")
        assert len(state.messages) <= 5

    def test_keeps_most_recent_messages(self):
        state = AgentState(max_messages=3)
        for i in range(5):
            state.add_user_message(f"Msg {i}")
        contents = [m.content for m in state.messages]
        assert "Msg 4" in contents
        assert "Msg 3" in contents
        assert "Msg 2" in contents
        assert "Msg 1" not in contents

    def test_compressed_summary_created(self):
        state = AgentState(max_messages=3)
        for i in range(5):
            state.add_user_message(f"User message {i}")
        assert state.get_compressed_summary() != ""

    def test_compressed_summary_includes_archived_count(self):
        state = AgentState(max_messages=3)
        for i in range(6):
            state.add_user_message(f"Msg {i}")
        summary = state.get_compressed_summary()
        assert "archived" in summary.lower() or "messages" in summary.lower()

    def test_compressed_summary_includes_user_intents(self):
        state = AgentState(max_messages=3)
        for i in range(6):
            state.add_user_message(f"User intent {i}")
        summary = state.get_compressed_summary()
        assert "User intents" in summary

    def test_compressed_summary_includes_tool_names(self):
        state = AgentState(max_messages=3)
        state.add_user_message("Msg 0")
        state.add_assistant_message(
            content="tool result",
            tool_calls=[{"name": "read", "arguments": {}}],
        )
        state.add_user_message("Msg 1")
        state.add_user_message("Msg 2")
        state.add_user_message("Msg 3")
        summary = state.get_compressed_summary()
        assert "read" in summary

    def test_compress_messages_empty_list(self):
        state = AgentState()
        state._compress_messages([])
        assert state.get_compressed_summary() == ""

    def test_no_compression_when_under_limit(self):
        state = AgentState(max_messages=50)
        for i in range(3):
            state.add_user_message(f"Msg {i}")
        assert state.get_compressed_summary() == ""


class TestRecentMessages:
    def test_get_recent_messages_fewer_than_n(self):
        state = AgentState()
        state.add_user_message("Only one")
        recent = state.get_recent_messages(5)
        assert len(recent) == 1

    def test_get_recent_messages_more_than_n(self):
        state = AgentState()
        for i in range(20):
            state.add_user_message(f"Msg {i}")
        recent = state.get_recent_messages(5)
        assert len(recent) == 5
        assert "Msg 19" in recent[-1].content

    def test_get_recent_messages_default(self):
        state = AgentState()
        for i in range(15):
            state.add_user_message(f"Msg {i}")
        recent = state.get_recent_messages()
        assert len(recent) == 12

    def test_get_history_for_api(self):
        state = AgentState()
        state.add_user_message("Hello")
        state.add_assistant_message("Hi")
        api = state.get_history_for_api()
        assert len(api) == 2
        assert api[0]["role"] == "user"
        assert api[1]["role"] == "assistant"


class TestClearHistory:
    def test_clear_history(self):
        state = AgentState()
        state.add_user_message("Hello")
        state.add_assistant_message("World")
        state.add_tool_call(ToolCall(id="1", name="read", arguments={}, result="ok"))
        state.turn_count = 3
        state.goal = "Implement feature"
        state.is_finished = True
        state.status = "finished"
        state.last_error = "boom"

        state.clear_history()

        assert state.messages == []
        assert state.tool_history == []
        assert state.turn_count == 0
        assert state.estimated_tokens == 0
        assert state.goal is None
        assert state.is_finished is False
        assert state.status == "idle"
        assert state.user_input is None
        assert state.last_response is None
        assert state.last_error is None


class TestGoalAndStatus:
    def test_set_goal(self):
        state = AgentState()
        state.set_goal("Fix the bug")
        assert state.goal == "Fix the bug"
        assert state.is_finished is False
        assert state.status == "running"

    def test_set_goal_resets_finished(self):
        state = AgentState()
        state.is_finished = True
        state.set_goal("New task")
        assert state.is_finished is False

    def test_mark_finished(self):
        state = AgentState()
        state.mark_finished()
        assert state.is_finished is True
        assert state.status == "finished"

    def test_mark_error(self):
        state = AgentState()
        state.mark_error("Disk full")
        assert state.last_error == "Disk full"
        assert state.status == "error"


class TestTurnTracking:
    def test_increment_turn_returns_true_when_under_limit(self):
        state = AgentState(max_turns=5)
        for i in range(4):
            assert state.increment_turn() is True
            assert state.turn_count == i + 1

    def test_increment_turn_returns_false_when_exceeded(self):
        state = AgentState(max_turns=3)
        assert state.increment_turn() is True
        assert state.turn_count == 1
        assert state.increment_turn() is True
        assert state.turn_count == 2
        assert state.increment_turn() is False

    def test_increment_turn_at_boundary(self):
        state = AgentState(max_turns=2)
        assert state.increment_turn() is True
        assert state.increment_turn() is False


class TestToolCallTracking:
    def test_get_tool_call_count_empty(self):
        state = AgentState()
        assert state.get_tool_call_count() == 0

    def test_get_tool_call_count(self):
        state = AgentState()
        state.add_tool_call(ToolCall(id="1", name="read", arguments={}, result="ok"))
        state.add_tool_call(ToolCall(id="2", name="write", arguments={}, result="done"))
        assert state.get_tool_call_count() == 2

    def test_get_successful_tool_calls(self):
        state = AgentState()
        state.add_tool_call(ToolCall(id="1", name="read", arguments={}, result="ok"))
        state.add_tool_call(ToolCall(id="2", name="write", arguments={}, error="failed"))
        successful = state.get_successful_tool_calls()
        assert len(successful) == 1
        assert successful[0].name == "read"

    def test_get_failed_tool_calls(self):
        state = AgentState()
        state.add_tool_call(ToolCall(id="1", name="read", arguments={}, result="ok"))
        state.add_tool_call(ToolCall(id="2", name="write", arguments={}, error="failed"))
        failed = state.get_failed_tool_calls()
        assert len(failed) == 1
        assert failed[0].name == "write"

    def test_all_successful(self):
        state = AgentState()
        state.add_tool_call(ToolCall(id="1", name="a", arguments={}, result="ok"))
        state.add_tool_call(ToolCall(id="2", name="b", arguments={}, result="ok"))
        assert len(state.get_successful_tool_calls()) == 2
        assert len(state.get_failed_tool_calls()) == 0

    def test_all_failed(self):
        state = AgentState()
        state.add_tool_call(ToolCall(id="1", name="a", arguments={}, error="e1"))
        state.add_tool_call(ToolCall(id="2", name="b", arguments={}, error="e2"))
        assert len(state.get_successful_tool_calls()) == 0
        assert len(state.get_failed_tool_calls()) == 2


class TestSerialization:
    def test_to_dict(self):
        state = AgentState(mode="plan", turn_count=5)
        state.add_user_message("Test")
        data = state.to_dict()
        assert data["mode"] == "plan"
        assert data["turn_count"] == 5
        assert data["message_count"] == 1
        assert data["is_finished"] is False
        assert data["status"] == "idle"

    def test_to_dict_with_tool_calls(self):
        state = AgentState()
        state.add_tool_call(ToolCall(id="1", name="read", arguments={}, result="ok"))
        data = state.to_dict()
        assert data["tool_call_count"] == 1

    def test_to_dict_with_goal(self):
        state = AgentState()
        state.goal = "My goal"
        data = state.to_dict()
        assert data["goal"] == "My goal"

    def test_from_dict_round_trip(self):
        state = AgentState(mode="plan", max_turns=50, turn_count=10)
        state.goal = "Test goal"
        state.is_finished = True
        state.status = "finished"
        data = state.to_dict()
        restored = AgentState.from_dict(data)
        assert restored.mode == "plan"
        assert restored.turn_count == 10
        assert restored.goal == "Test goal"
        assert restored.is_finished is True


class TestCheckpoint:
    def test_create_checkpoint(self):
        state = AgentState(mode="plan", goal="test goal")
        state.add_user_message("Hello")
        state.add_tool_call(ToolCall(id="1", name="read", arguments={}, result="data"))

        checkpoint = state.create_checkpoint()

        assert "messages" in checkpoint
        assert "tool_history" in checkpoint
        assert "state" in checkpoint
        assert len(checkpoint["messages"]) == 1
        assert len(checkpoint["tool_history"]) == 1
        assert checkpoint["state"]["mode"] == "plan"
        assert checkpoint["state"]["goal"] == "test goal"

    def test_restore_checkpoint(self):
        state = AgentState(mode="plan", goal="test goal")
        state.add_user_message("Hello")
        state.add_tool_call(ToolCall(id="1", name="read", arguments={}, result="data"))
        state.turn_count = 5

        checkpoint = state.create_checkpoint()

        new_state = AgentState()
        new_state.restore_checkpoint(checkpoint)

        assert new_state.mode == "plan"
        assert new_state.goal == "test goal"
        assert new_state.turn_count == 5

    def test_restore_checkpoint_preserves_status(self):
        state = AgentState()
        state.status = "running"
        state.is_finished = False

        checkpoint = state.create_checkpoint()

        new_state = AgentState()
        new_state.restore_checkpoint(checkpoint)

        assert new_state.status == "running"
        assert new_state.is_finished is False

    def test_restore_checkpoint_with_missing_state(self):
        state = AgentState()
        checkpoint = {"messages": [], "tool_history": []}
        state.restore_checkpoint(checkpoint)
        assert state.mode == "build"
        assert state.status == "idle"

    def test_checkpoint_message_format(self):
        state = AgentState()
        state.add_user_message("Test message")
        checkpoint = state.create_checkpoint()
        assert checkpoint["messages"][0]["role"] == "user"
        assert checkpoint["messages"][0]["content"] == "Test message"

    def test_checkpoint_tool_history_format(self):
        state = AgentState()
        tc = ToolCall(id="1", name="read", arguments={"path": "/f"}, result="data")
        state.add_tool_call(tc)
        checkpoint = state.create_checkpoint()
        th = checkpoint["tool_history"][0]
        assert th["id"] == "1"
        assert th["name"] == "read"
        assert th["arguments"] == {"path": "/f"}
