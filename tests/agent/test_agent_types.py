"""
Tests for src/agent/types.py - Core Data Structures

Comprehensive tests for all data types: AgentStatus, ActionType,
Message, ToolCall, Observation, Thought, Action, AgentResult, ToolDefinition.
"""

from src.agent.types import (
    Action,
    ActionType,
    AgentResult,
    AgentStatus,
    Message,
    Observation,
    Thought,
    ToolCall,
    ToolDefinition,
)


class TestAgentStatus:
    def test_all_status_values(self):
        assert AgentStatus.IDLE.value == "idle"
        assert AgentStatus.RUNNING.value == "running"
        assert AgentStatus.THINKING.value == "thinking"
        assert AgentStatus.ACTING.value == "acting"
        assert AgentStatus.WAITING.value == "waiting"
        assert AgentStatus.FINISHED.value == "finished"
        assert AgentStatus.ERROR.value == "error"

    def test_enum_membership(self):
        assert len(AgentStatus) == 7
        all_values = {s.value for s in AgentStatus}
        assert "idle" in all_values
        assert "error" in all_values

    def test_enum_identity(self):
        assert AgentStatus.IDLE is AgentStatus.IDLE
        assert AgentStatus.RUNNING is not AgentStatus.FINISHED


class TestActionType:
    def test_all_action_type_values(self):
        assert ActionType.TOOL_CALL.value == "tool_call"
        assert ActionType.RESPOND.value == "respond"
        assert ActionType.ASK_USER.value == "ask_user"
        assert ActionType.ERROR.value == "error"
        assert ActionType.FINISH.value == "finish"

    def test_enum_count(self):
        assert len(ActionType) == 5


class TestMessage:
    def test_minimal_message(self):
        msg = Message(role="user")
        assert msg.role == "user"
        assert msg.content is None
        assert msg.provider_items is None
        assert msg.tool_calls is None
        assert msg.tool_call_id is None
        assert msg.name is None
        assert msg.thought is None

    def test_full_message(self):
        msg = Message(
            role="assistant",
            content="hello",
            provider_items=[{"type": "reasoning"}],
            tool_calls=[{"name": "read"}],
            tool_call_id="call_1",
            name="read",
            thought="I should read the file",
        )
        assert msg.role == "assistant"
        assert msg.content == "hello"
        assert msg.provider_items == [{"type": "reasoning"}]
        assert msg.tool_calls == [{"name": "read"}]
        assert msg.tool_call_id == "call_1"
        assert msg.name == "read"
        assert msg.thought == "I should read the file"

    def test_to_api_format_minimal(self):
        msg = Message(role="user", content="hi")
        result = msg.to_api_format()
        assert result == {"role": "user", "content": "hi"}

    def test_to_api_format_only_includes_non_none(self):
        msg = Message(role="assistant", content="hi")
        result = msg.to_api_format()
        assert "provider_items" not in result
        assert "tool_calls" not in result
        assert "tool_call_id" not in result
        assert "name" not in result
        assert "thought" not in result

    def test_to_api_format_with_all_fields(self):
        msg = Message(
            role="tool",
            content="file content",
            tool_call_id="call_1",
            name="read",
        )
        result = msg.to_api_format()
        assert result["role"] == "tool"
        assert result["content"] == "file content"
        assert result["tool_call_id"] == "call_1"
        assert result["name"] == "read"

    def test_to_api_format_preserves_empty_string_content(self):
        msg = Message(role="assistant", content="")
        result = msg.to_api_format()
        assert result["content"] == ""

    def test_to_api_format_with_provider_items(self):
        msg = Message(
            role="assistant",
            content="result",
            provider_items=[{"type": "reasoning", "id": "rs_1"}],
        )
        result = msg.to_api_format()
        assert result["provider_items"] == [{"type": "reasoning", "id": "rs_1"}]

    def test_to_api_format_with_thought(self):
        msg = Message(role="assistant", content="hi", thought="reasoning step")
        result = msg.to_api_format()
        assert result["thought"] == "reasoning step"

    def test_to_api_format_with_tool_calls(self):
        msg = Message(
            role="assistant",
            content=None,
            tool_calls=[{"id": "c1", "name": "read", "arguments": {}}],
        )
        result = msg.to_api_format()
        assert result["tool_calls"] == [{"id": "c1", "name": "read", "arguments": {}}]

    def test_to_api_format_content_none_excluded(self):
        msg = Message(role="assistant", content=None)
        result = msg.to_api_format()
        assert "content" not in result

    def test_message_roles(self):
        for role in ("user", "assistant", "system", "tool"):
            msg = Message(role=role, content=f"{role} msg")
            assert msg.to_api_format()["role"] == role


class TestToolCall:
    def test_minimal_tool_call(self):
        tc = ToolCall(id="1", name="read", arguments={"path": "/test"})
        assert tc.id == "1"
        assert tc.name == "read"
        assert tc.arguments == {"path": "/test"}
        assert tc.result is None
        assert tc.error is None
        assert tc.duration == 0.0

    def test_full_tool_call(self):
        tc = ToolCall(
            id="2",
            name="write",
            arguments={"path": "/out", "content": "data"},
            result="written successfully",
            error=None,
            duration=1.5,
        )
        assert tc.result == "written successfully"
        assert tc.duration == 1.5

    def test_tool_call_with_error(self):
        tc = ToolCall(
            id="3",
            name="delete",
            arguments={},
            result=None,
            error="Permission denied",
            duration=0.2,
        )
        assert tc.error == "Permission denied"
        assert tc.result is None

    def test_to_dict(self):
        tc = ToolCall(
            id="tc1",
            name="read",
            arguments={"path": "/file.py"},
            result="content",
            error=None,
            duration=0.5,
        )
        d = tc.to_dict()
        assert d == {
            "id": "tc1",
            "name": "read",
            "arguments": {"path": "/file.py"},
            "result": "content",
            "error": None,
            "duration": 0.5,
        }

    def test_to_dict_with_error(self):
        tc = ToolCall(
            id="tc2",
            name="bad_tool",
            arguments={},
            result=None,
            error="Tool crashed",
            duration=0.1,
        )
        d = tc.to_dict()
        assert d["error"] == "Tool crashed"
        assert d["result"] is None

    def test_to_dict_defaults(self):
        tc = ToolCall(id="x", name="y", arguments={})
        d = tc.to_dict()
        assert d["result"] is None
        assert d["error"] is None
        assert d["duration"] == 0.0


class TestObservation:
    def test_default_observation(self):
        obs = Observation()
        assert obs.user_input is None
        assert obs.tool_results == []
        assert obs.errors == []
        assert obs.context == {}

    def test_observation_with_values(self):
        tc = ToolCall(id="1", name="read", arguments={}, result="data")
        obs = Observation(
            user_input="Hello",
            tool_results=[tc],
            errors=["file not found"],
            context={"turn": 1},
        )
        assert obs.user_input == "Hello"
        assert len(obs.tool_results) == 1
        assert obs.tool_results[0].name == "read"
        assert obs.errors == ["file not found"]
        assert obs.context == {"turn": 1}

    def test_observation_independent_default_factories(self):
        obs1 = Observation()
        obs2 = Observation()
        obs1.tool_results.append(ToolCall(id="1", name="a", arguments={}))
        obs1.errors.append("err")
        obs1.context["key"] = "val"
        assert obs2.tool_results == []
        assert obs2.errors == []
        assert obs2.context == {}


class TestThought:
    def test_default_thought(self):
        t = Thought()
        assert t.reasoning == ""
        assert t.provider_items == []
        assert t.tool_calls == []
        assert t.response is None
        assert t.is_finished is False

    def test_thought_with_values(self):
        t = Thought(
            reasoning="I need to read the file first",
            provider_items=[{"type": "reasoning"}],
            tool_calls=[{"name": "read", "arguments": {}}],
            response="Let me check",
            is_finished=False,
        )
        assert t.reasoning == "I need to read the file first"
        assert len(t.provider_items) == 1
        assert len(t.tool_calls) == 1
        assert t.response == "Let me check"
        assert t.is_finished is False

    def test_thought_finished(self):
        t = Thought(is_finished=True, response="Task complete")
        assert t.is_finished is True

    def test_independent_default_factories(self):
        t1 = Thought()
        t2 = Thought()
        t1.tool_calls.append({"name": "x"})
        t1.provider_items.append({"type": "y"})
        assert t2.tool_calls == []
        assert t2.provider_items == []


class TestAction:
    def test_tool_call_classmethod(self):
        action = Action.tool_call("read", {"path": "/test"}, "call_1")
        assert action.type == ActionType.TOOL_CALL
        assert action.tool_name == "read"
        assert action.tool_args == {"path": "/test"}
        assert action.tool_call_id == "call_1"
        assert action.response is None
        assert action.error is None

    def test_respond_classmethod(self):
        action = Action.respond("Hello!")
        assert action.type == ActionType.RESPOND
        assert action.response == "Hello!"
        assert action.tool_name is None
        assert action.tool_args is None

    def test_ask_user_classmethod(self):
        action = Action.ask_user("What file?")
        assert action.type == ActionType.ASK_USER
        assert action.response == "What file?"
        assert action.tool_name is None

    def test_finish_classmethod_with_response(self):
        action = Action.finish("All done")
        assert action.type == ActionType.FINISH
        assert action.response == "All done"

    def test_finish_classmethod_without_response(self):
        action = Action.finish()
        assert action.type == ActionType.FINISH
        assert action.response is None

    def test_direct_construction(self):
        action = Action(type=ActionType.ERROR, error="Something broke")
        assert action.type == ActionType.ERROR
        assert action.error == "Something broke"

    def test_to_dict_tool_call(self):
        action = Action.tool_call("read", {"path": "/f"}, "c1")
        d = action.to_dict()
        assert d["type"] == "tool_call"
        assert d["tool_name"] == "read"
        assert d["tool_args"] == {"path": "/f"}
        assert d["tool_call_id"] == "c1"

    def test_to_dict_respond(self):
        action = Action.respond("Done")
        d = action.to_dict()
        assert d["type"] == "respond"
        assert d["response"] == "Done"
        assert "tool_name" not in d
        assert "tool_args" not in d

    def test_to_dict_finish(self):
        action = Action.finish("Complete")
        d = action.to_dict()
        assert d["type"] == "finish"
        assert d["response"] == "Complete"

    def test_to_dict_with_error(self):
        action = Action(type=ActionType.ERROR, error="crash")
        d = action.to_dict()
        assert d["type"] == "error"
        assert d["error"] == "crash"

    def test_to_dict_excludes_none_fields(self):
        action = Action.respond("hello")
        d = action.to_dict()
        assert "tool_name" not in d
        assert "tool_args" not in d
        assert "tool_call_id" not in d
        assert "error" not in d

    def test_to_dict_empty_response_excluded(self):
        action = Action(type=ActionType.RESPOND, response=None)
        d = action.to_dict()
        assert "response" not in d


class TestAgentResult:
    def test_minimal_result(self):
        r = AgentResult(success=True)
        assert r.success is True
        assert r.response is None
        assert r.tool_calls == []
        assert r.messages == []
        assert r.tokens_used == 0
        assert r.duration == 0.0
        assert r.error is None
        assert r.metadata == {}

    def test_full_result(self):
        tc = ToolCall(id="1", name="read", arguments={}, result="data")
        msg = Message(role="user", content="hi")
        r = AgentResult(
            success=True,
            response="Done",
            tool_calls=[tc],
            messages=[msg],
            tokens_used=150,
            duration=2.5,
            error=None,
            metadata={"turns": 3},
        )
        assert r.success is True
        assert r.response == "Done"
        assert len(r.tool_calls) == 1
        assert len(r.messages) == 1
        assert r.tokens_used == 150
        assert r.duration == 2.5
        assert r.metadata == {"turns": 3}

    def test_failed_result(self):
        r = AgentResult(success=False, error="Something failed")
        assert r.success is False
        assert r.error == "Something failed"

    def test_to_dict(self):
        tc = ToolCall(id="1", name="read", arguments={}, result="content")
        r = AgentResult(
            success=True,
            response="Done",
            tool_calls=[tc],
            tokens_used=100,
            duration=1.0,
        )
        d = r.to_dict()
        assert d["success"] is True
        assert d["response"] == "Done"
        assert len(d["tool_calls"]) == 1
        assert d["tool_calls"][0]["name"] == "read"
        assert d["tokens_used"] == 100
        assert d["duration"] == 1.0
        assert d["error"] is None
        assert d["metadata"] == {}

    def test_to_dict_with_error(self):
        r = AgentResult(success=False, error="boom")
        d = r.to_dict()
        assert d["success"] is False
        assert d["error"] == "boom"

    def test_to_dict_serializes_tool_calls(self):
        tc1 = ToolCall(id="a", name="read", arguments={"p": "1"}, result="r1")
        tc2 = ToolCall(id="b", name="write", arguments={"p": "2"}, error="fail")
        r = AgentResult(success=True, tool_calls=[tc1, tc2])
        d = r.to_dict()
        assert d["tool_calls"][0]["id"] == "a"
        assert d["tool_calls"][1]["error"] == "fail"

    def test_independent_default_factories(self):
        r1 = AgentResult(success=True)
        r2 = AgentResult(success=True)
        r1.tool_calls.append(ToolCall(id="1", name="x", arguments={}))
        r1.messages.append(Message(role="user", content="hi"))
        r1.metadata["key"] = "val"
        assert r2.tool_calls == []
        assert r2.messages == []
        assert r2.metadata == {}


class TestToolDefinition:
    def test_basic_tool(self):
        tool = ToolDefinition(
            name="read",
            description="Read a file",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        )
        assert tool.name == "read"
        assert tool.description == "Read a file"
        assert tool.sensitive is False

    def test_sensitive_tool(self):
        tool = ToolDefinition(
            name="write",
            description="Write a file",
            parameters={},
            sensitive=True,
        )
        assert tool.sensitive is True

    def test_to_api_format(self):
        tool = ToolDefinition(
            name="search",
            description="Search codebase",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        api = tool.to_api_format()
        assert api["type"] == "function"
        assert api["function"]["name"] == "search"
        assert api["function"]["description"] == "Search codebase"
        assert api["function"]["parameters"]["required"] == ["query"]

    def test_to_api_format_structure(self):
        tool = ToolDefinition(name="run", description="Run command", parameters={})
        api = tool.to_api_format()
        assert "type" in api
        assert "function" in api
        assert "name" in api["function"]
        assert "description" in api["function"]
        assert "parameters" in api["function"]

    def test_to_api_format_sensitive_tool(self):
        tool = ToolDefinition(
            name="delete",
            description="Delete a file",
            parameters={},
            sensitive=True,
        )
        api = tool.to_api_format()
        assert api["function"]["name"] == "delete"
        assert "sensitive" not in api["function"]
