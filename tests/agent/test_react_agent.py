"""
Tests for src/agent/react.py - ReAct Agent

Comprehensive tests for ReActAgent, TraceInterface, and BaseAgent:
observe, think, act, execute_tool, run, run_stream, _parse_llm_response,
_build_messages, _get_system_prompt, _execute_tool_with_permission,
_compress_context, _execute_tools_parallel, and related methods.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.base import (
    AgentError,
    BaseAgent,
    ContextOverflowError,
    MaxTurnsExceededError,
    ToolExecutionError,
    ToolNotFoundError,
)
from src.agent.react import ReActAgent, TraceInterface
from src.agent.state import AgentState
from src.agent.types import (
    Action,
    ActionType,
    AgentResult,
    Message,
    Observation,
    Thought,
    ToolCall,
    ToolDefinition,
)


class TestTraceInterface:
    def test_tool_call_noop(self):
        trace = TraceInterface()
        trace.tool_call("read", {"path": "/f"}, "data", 0.5, None)

    def test_llm_call_noop(self):
        trace = TraceInterface()
        trace.llm_call("gpt-4", [], {}, 100, 50, 1.2)

    def test_event_noop(self):
        trace = TraceInterface()
        trace.event("error", "tool_failed", {"name": "read"})


class TestBaseAgentAbstract:
    def test_base_agent_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseAgent()

    def test_base_agent_subclass_must_implement_observe(self):
        class PartialAgent(BaseAgent):
            async def think(self, obs):
                pass

            async def act(self, thought):
                pass

            async def execute_tool(self, name, args):
                pass

            async def run(self, input, **kwargs):
                pass

            async def run_stream(self, input, **kwargs):
                pass

            async def ask_user(self, question, options=None):
                pass

        with pytest.raises(TypeError):
            PartialAgent()


class TestBaseAgentErrors:
    def test_agent_error_hierarchy(self):
        assert issubclass(ToolNotFoundError, AgentError)
        assert issubclass(ToolExecutionError, AgentError)
        assert issubclass(MaxTurnsExceededError, AgentError)
        assert issubclass(ContextOverflowError, AgentError)

    def test_agent_error_is_exception(self):
        assert issubclass(AgentError, Exception)

    def test_max_turns_exceeded_message(self):
        err = MaxTurnsExceededError("Exceeded max turns (10)")
        assert "10" in str(err)


class TestReActAgentInit:
    @pytest.fixture
    def mock_llm(self):
        llm = AsyncMock()
        llm.chat = AsyncMock(return_value=MagicMock(content="Done!", tool_calls=None))
        return llm

    @pytest.fixture
    def tools(self):
        return [
            ToolDefinition(
                name="read",
                description="Read a file",
                parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            ),
            ToolDefinition(
                name="write",
                description="Write a file",
                parameters={},
                sensitive=True,
            ),
        ]

    @pytest.fixture
    def agent(self, mock_llm, tools):
        return ReActAgent(llm_client=mock_llm, tools=tools, max_turns=10)

    def test_initialization(self, agent, tools):
        assert len(agent.tools) == 2
        assert agent.is_sensitive_tool("write") is True
        assert agent.is_sensitive_tool("read") is False
        assert agent.max_turns == 10
        assert agent.timeout == 300.0

    def test_register_tool(self, agent):
        new_tool = ToolDefinition(
            name="delete", description="Delete", parameters={}, sensitive=True
        )
        agent.register_tool(new_tool)
        assert len(agent.tools) == 3
        assert agent.is_sensitive_tool("delete") is True

    def test_set_trace(self, agent):
        trace = TraceInterface()
        agent.set_trace(trace)
        assert agent._trace is trace

    def test_set_trace_none(self, agent):
        agent.set_trace(None)
        assert agent._trace is None

    def test_set_tool_executor(self, agent):
        executor = AsyncMock()
        agent.set_tool_executor(executor)
        assert agent._tool_executor is executor

    def test_custom_state(self, mock_llm):
        state = AgentState(mode="plan")
        agent = ReActAgent(llm_client=mock_llm, state=state)
        assert agent.state.mode == "plan"

    def test_permission_callback(self, mock_llm):
        callback = AsyncMock(return_value="yes")
        agent = ReActAgent(
            llm_client=mock_llm,
            permission_callback=callback,
        )
        assert agent.permission_callback is callback


class TestReActAgentObserve:
    @pytest.fixture
    def agent(self):
        llm = AsyncMock()
        return ReActAgent(llm_client=llm, tools=[])

    @pytest.mark.asyncio
    async def test_observe_empty_state(self, agent):
        obs = await agent.observe()
        assert obs.user_input is None
        assert obs.tool_results == []
        assert obs.errors == []
        assert obs.context["turn_count"] == 0

    @pytest.mark.asyncio
    async def test_observe_with_user_input(self, agent):
        agent.state.add_user_message("Hello")
        obs = await agent.observe()
        assert obs.user_input == "Hello"

    @pytest.mark.asyncio
    async def test_observe_with_tool_history(self, agent):
        tc = ToolCall(id="1", name="read", arguments={}, result="file content")
        agent.state.add_tool_call(tc)
        obs = await agent.observe()
        assert len(obs.tool_results) == 1
        assert obs.tool_results[0].name == "read"

    @pytest.mark.asyncio
    async def test_observe_with_tool_error(self, agent):
        tc = ToolCall(id="1", name="read", arguments={}, error="File not found")
        agent.state.add_tool_call(tc)
        obs = await agent.observe()
        assert "File not found" in obs.errors

    @pytest.mark.asyncio
    async def test_observe_context_includes_goal(self, agent):
        agent.state.goal = "Fix bug"
        obs = await agent.observe()
        assert obs.context["goal"] == "Fix bug"

    @pytest.mark.asyncio
    async def test_observe_context_includes_turn_count(self, agent):
        agent.state.turn_count = 5
        obs = await agent.observe()
        assert obs.context["turn_count"] == 5

    @pytest.mark.asyncio
    async def test_observe_limits_recent_tool_results(self, agent):
        for i in range(10):
            agent.state.add_tool_call(
                ToolCall(id=str(i), name=f"tool_{i}", arguments={}, result=f"result_{i}")
            )
        obs = await agent.observe()
        assert len(obs.tool_results) <= 15

    @pytest.mark.asyncio
    async def test_observe_goal_relevant_tool_results(self, agent):
        agent.state.goal = "analyze database"
        agent.state.add_tool_call(
            ToolCall(id="old1", name="search", arguments={}, result="database schema found")
        )
        agent.state.add_tool_call(
            ToolCall(id="recent1", name="read", arguments={}, result="current file")
        )
        obs = await agent.observe()
        assert any(tc.name == "read" for tc in obs.tool_results)


class TestReActAgentThink:
    @pytest.fixture
    def mock_llm(self):
        llm = AsyncMock()
        llm.chat = AsyncMock(
            return_value=MagicMock(
                content="I will help you",
                tool_calls=None,
                thought="reasoning",
                provider_items=None,
            )
        )
        return llm

    @pytest.fixture
    def agent(self, mock_llm):
        return ReActAgent(llm_client=mock_llm, tools=[], max_turns=10)

    @pytest.mark.asyncio
    async def test_think_returns_thought(self, agent, mock_llm):
        agent.state.add_user_message("Hello")
        thought = await agent.think(Observation(user_input="Hello"))
        assert isinstance(thought, Thought)
        assert thought.response == "I will help you"
        assert thought.is_finished is True

    @pytest.mark.asyncio
    async def test_think_with_tool_calls(self, agent, mock_llm):
        mock_llm.chat.return_value = MagicMock(
            content=None,
            tool_calls=[
                MagicMock(
                    id="call_1",
                    function=MagicMock(name="read", arguments='{"path": "/f"}'),
                )
            ],
        )
        agent.state.add_user_message("Read file")
        thought = await agent.think(Observation(user_input="Read file"))
        assert len(thought.tool_calls) == 1
        assert thought.is_finished is False

    @pytest.mark.asyncio
    async def test_think_surfaces_provider_thought(self, agent, mock_llm):
        mock_llm.chat.return_value = MagicMock(
            content="Done!",
            thought="I should respond directly.",
            tool_calls=None,
            provider_items=None,
        )
        agent.state.add_user_message("Hello")
        thought = await agent.think(Observation(user_input="Hello"))
        assert thought.reasoning == "I should respond directly."

    @pytest.mark.asyncio
    async def test_think_timeout(self, mock_llm):
        async def slow_chat(*args, **kwargs):
            await asyncio.sleep(10)
            return MagicMock(content="late", tool_calls=None)

        mock_llm.chat = slow_chat
        agent = ReActAgent(llm_client=mock_llm, tools=[], timeout=0.1)
        agent.state.add_user_message("Test")
        with pytest.raises(TimeoutError, match="timed out"):
            await agent.think(Observation(user_input="Test"))

    @pytest.mark.asyncio
    async def test_think_runtime_error(self, agent, mock_llm):
        mock_llm.chat.side_effect = ValueError("Bad input")
        agent.state.add_user_message("Hello")
        with pytest.raises(RuntimeError, match="Error in thinking"):
            await agent.think(Observation(user_input="Hello"))


class TestReActAgentAct:
    @pytest.fixture
    def agent(self):
        llm = AsyncMock()
        return ReActAgent(llm_client=llm, tools=[])

    @pytest.mark.asyncio
    async def test_act_respond_when_finished(self, agent):
        thought = Thought(reasoning="Done", response="All complete", is_finished=True)
        action = await agent.act(thought)
        assert action.type == ActionType.RESPOND
        assert action.response == "All complete"

    @pytest.mark.asyncio
    async def test_act_respond_when_no_tool_calls(self, agent):
        thought = Thought(reasoning="No tools", response="I can answer directly")
        action = await agent.act(thought)
        assert action.type == ActionType.RESPOND

    @pytest.mark.asyncio
    async def test_act_tool_call(self, agent):
        thought = Thought(
            reasoning="Need to read file",
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "read",
                    "arguments": {"path": "/test"},
                }
            ],
        )
        action = await agent.act(thought)
        assert action.type == ActionType.TOOL_CALL
        assert action.tool_name == "read"
        assert action.tool_args == {"path": "/test"}
        assert action.tool_call_id == "call_1"

    @pytest.mark.asyncio
    async def test_act_tool_call_with_function_format(self, agent):
        thought = Thought(
            reasoning="Using function format",
            tool_calls=[
                {
                    "id": "call_2",
                    "function": {"name": "write", "arguments": {"path": "/out"}},
                }
            ],
        )
        action = await agent.act(thought)
        assert action.type == ActionType.TOOL_CALL
        assert action.tool_name == "write"
        assert action.tool_args == {"path": "/out"}

    @pytest.mark.asyncio
    async def test_act_parses_string_arguments(self, agent):
        thought = Thought(
            reasoning="String args",
            tool_calls=[
                {
                    "id": "call_3",
                    "name": "read",
                    "arguments": '{"path": "/test"}',
                }
            ],
        )
        action = await agent.act(thought)
        assert action.tool_args == {"path": "/test"}

    @pytest.mark.asyncio
    async def test_act_handles_invalid_json_arguments(self, agent):
        thought = Thought(
            reasoning="Bad JSON",
            tool_calls=[
                {
                    "id": "call_4",
                    "name": "read",
                    "arguments": "{invalid json}",
                }
            ],
        )
        action = await agent.act(thought)
        assert action.tool_args == {}

    @pytest.mark.asyncio
    async def test_act_generates_id_when_missing(self, agent):
        thought = Thought(
            reasoning="No ID",
            tool_calls=[{"name": "read", "arguments": {}}],
        )
        action = await agent.act(thought)
        assert action.tool_call_id is not None
        assert len(action.tool_call_id) > 0

    @pytest.mark.asyncio
    async def test_act_only_processes_first_tool_call(self, agent):
        thought = Thought(
            reasoning="Multiple tools",
            tool_calls=[
                {"id": "c1", "name": "read", "arguments": {}},
                {"id": "c2", "name": "write", "arguments": {}},
            ],
        )
        action = await agent.act(thought)
        assert action.tool_name == "read"

    @pytest.mark.asyncio
    async def test_act_default_response_when_no_content(self, agent):
        thought = Thought(is_finished=True, response=None)
        action = await agent.act(thought)
        assert action.response == "Task completed."


class TestReActAgentExecuteTool:
    @pytest.fixture
    def agent(self):
        llm = AsyncMock()
        tools = [
            ToolDefinition(name="read", description="Read", parameters={}),
            ToolDefinition(name="write", description="Write", parameters={}, sensitive=True),
        ]
        return ReActAgent(llm_client=llm, tools=tools)

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, agent):
        result, error = await agent.execute_tool("unknown", {})
        assert result == ""
        assert "not found" in error

    @pytest.mark.asyncio
    async def test_execute_tool_with_executor(self, agent):
        executor = AsyncMock(return_value=("file content", None))
        agent.set_tool_executor(executor)
        result, error = await agent.execute_tool("read", {"path": "/f"})
        assert result == "file content"
        assert error is None

    @pytest.mark.asyncio
    async def test_execute_tool_without_executor(self, agent):
        result, error = await agent.execute_tool("read", {"path": "/f"})
        assert result == ""
        assert "No tool executor" in error

    @pytest.mark.asyncio
    async def test_execute_tool_with_error(self, agent):
        executor = AsyncMock(return_value=("", "Permission denied"))
        agent.set_tool_executor(executor)
        result, error = await agent.execute_tool("write", {"path": "/f"})
        assert error == "Permission denied"

    @pytest.mark.asyncio
    async def test_execute_tool_traces(self, agent):
        trace = TraceInterface()
        trace.tool_call = MagicMock()
        agent.set_trace(trace)
        executor = AsyncMock(return_value=("data", None))
        agent.set_tool_executor(executor)
        await agent.execute_tool("read", {"path": "/f"})
        trace.tool_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_unknown_tool_traces(self, agent):
        trace = TraceInterface()
        trace.tool_call = MagicMock()
        agent.set_trace(trace)
        await agent.execute_tool("missing", {})
        trace.tool_call.assert_called_once()
        call_args = trace.tool_call.call_args
        assert "not found" in call_args[0][4]


class TestReActAgentPermission:
    @pytest.fixture
    def agent(self):
        llm = AsyncMock()
        tools = [
            ToolDefinition(name="read", description="Read", parameters={}),
            ToolDefinition(name="write", description="Write", parameters={}, sensitive=True),
        ]
        return ReActAgent(llm_client=llm, tools=tools)

    @pytest.mark.asyncio
    async def test_check_permission_non_sensitive(self, agent):
        allowed = await agent.check_permission("read", {"path": "/f"})
        assert allowed is True

    @pytest.mark.asyncio
    async def test_check_permission_allowed(self, agent):
        agent.permission_callback = AsyncMock(return_value="yes")
        allowed = await agent.check_permission("write", {"path": "/f"})
        assert allowed is True

    @pytest.mark.asyncio
    async def test_check_permission_denied(self, agent):
        agent.permission_callback = AsyncMock(return_value="no")
        allowed = await agent.check_permission("write", {"path": "/f"})
        assert allowed is False

    @pytest.mark.asyncio
    async def test_execute_with_permission_denied(self, agent):
        agent.permission_callback = AsyncMock(return_value="no")
        result, error = await agent._execute_tool_with_permission("write", {"path": "/f"})
        assert result == ""
        assert "Permission denied" in error

    @pytest.mark.asyncio
    async def test_execute_with_permission_allowed(self, agent):
        agent.permission_callback = AsyncMock(return_value="yes")
        executor = AsyncMock(return_value=("written", None))
        agent.set_tool_executor(executor)
        result, error = await agent._execute_tool_with_permission("write", {"path": "/f"})
        assert result == "written"
        assert error is None

    @pytest.mark.asyncio
    async def test_ask_user_with_callback(self, agent):
        agent.permission_callback = AsyncMock(return_value="yes")
        result = await agent.ask_user("Allow this?", ["yes", "no"])
        assert result == "yes"

    @pytest.mark.asyncio
    async def test_ask_user_without_callback(self, agent):
        result = await agent.ask_user("Allow this?")
        assert result == "yes"


class TestParseLlmResponse:
    @pytest.fixture
    def agent(self):
        llm = AsyncMock()
        return ReActAgent(llm_client=llm, tools=[])

    def test_parse_object_with_content(self, agent):
        response = MagicMock(content="Hello!", tool_calls=None)
        result = agent._parse_llm_response(response)
        assert result["content"] == "Hello!"
        assert result["tool_calls"] == []
        assert result["is_finished"] is True

    def test_parse_object_with_tool_calls(self, agent):
        func = MagicMock()
        func.name = "read"
        func.arguments = '{"path": "/f"}'
        tc = MagicMock(id="call_1", function=func)
        response = MagicMock(content=None, tool_calls=[tc])
        result = agent._parse_llm_response(response)
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "read"
        assert result["is_finished"] is False

    def test_parse_dict_response(self, agent):
        response = {
            "content": "Here's the answer",
            "tool_calls": [],
        }
        result = agent._parse_llm_response(response)
        assert result["content"] == "Here's the answer"
        assert result["is_finished"] is True

    def test_parse_dict_with_tool_calls(self, agent):
        response = {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "read",
                    "arguments": {"path": "/f"},
                }
            ],
        }
        result = agent._parse_llm_response(response)
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "read"

    def test_parse_dict_with_function_format(self, agent):
        response = {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_2",
                    "function": {"name": "write", "arguments": '{"path": "/out"}'},
                }
            ],
        }
        result = agent._parse_llm_response(response)
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "write"

    def test_parse_string_response(self, agent):
        result = agent._parse_llm_response("Just a string")
        assert result["content"] == "Just a string"
        assert result["tool_calls"] == []
        assert result["is_finished"] is True

    def test_parse_object_preserves_provider_items(self, agent):
        response = MagicMock(
            content="result",
            tool_calls=None,
            provider_items=[{"type": "reasoning"}],
            thought=None,
        )
        result = agent._parse_llm_response(response)
        assert result["provider_items"] == [{"type": "reasoning"}]

    def test_parse_dict_preserves_thought(self, agent):
        response = {
            "content": "done",
            "thought": "my reasoning",
            "tool_calls": [],
        }
        result = agent._parse_llm_response(response)
        assert result["thought"] == "my reasoning"

    def test_parse_tool_call_with_thought_signature(self, agent):
        response = {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "thought_signature": "sig_123",
                    "function": {"name": "write", "arguments": '{"path":"a.py"}'},
                }
            ],
        }
        result = agent._parse_llm_response(response)
        assert result["tool_calls"][0]["thought_signature"] == "sig_123"

    def test_parse_object_tool_call_with_dict_style(self, agent):
        tc_dict = {
            "id": "call_d1",
            "name": "search",
            "arguments": '{"query": "test"}',
        }
        response = MagicMock(content=None, tool_calls=[tc_dict])
        result = agent._parse_llm_response(response)
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "search"

    def test_parse_skips_tool_calls_without_name(self, agent):
        response = {
            "content": None,
            "tool_calls": [
                {"id": "call_1", "arguments": {}},
                {"id": "call_2", "name": "read", "arguments": {}},
            ],
        }
        result = agent._parse_llm_response(response)
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "read"

    def test_parse_tool_call_arguments_from_function(self, agent):
        response = {
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {"name": "read", "arguments": '{"path":"/f"}'},
                }
            ],
        }
        result = agent._parse_llm_response(response)
        assert result["tool_calls"][0]["arguments"] == '{"path":"/f"}'

    def test_parse_generates_uuid_for_missing_id(self, agent):
        response = {
            "content": None,
            "tool_calls": [{"name": "read", "arguments": {}}],
        }
        result = agent._parse_llm_response(response)
        assert result["tool_calls"][0]["id"] is not None


class TestBuildMessages:
    @pytest.fixture
    def agent(self):
        llm = AsyncMock()
        return ReActAgent(llm_client=llm, tools=[])

    def test_build_messages_includes_system(self, agent):
        agent.state.add_user_message("Hello")
        msgs = agent._build_messages(Observation(user_input="Hello"))
        assert msgs[0]["role"] == "system"

    def test_build_messages_includes_state_messages(self, agent):
        agent.state.add_user_message("Hello")
        agent.state.add_assistant_message("Hi")
        msgs = agent._build_messages(Observation(user_input="Hello"))
        assert len(msgs) == 3
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

    def test_build_messages_keeps_assistant_tool_call_before_tool_result(self, agent):
        for i in range(5):
            agent.state.add_user_message(f"user-{i}")
        agent.state.add_assistant_message(
            content=None,
            tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path":"README.md"}'},
                }
            ],
        )
        agent.state.add_tool_result("c1", "read", "contents")
        msgs = agent._build_messages(Observation(user_input="next"))
        tool_idx = next(i for i, m in enumerate(msgs) if m["role"] == "tool")
        assert tool_idx > 0
        assert msgs[tool_idx - 1]["role"] == "assistant"


class TestSystemPrompt:
    @pytest.fixture
    def mock_llm(self):
        return AsyncMock()

    def test_build_mode_prompt(self, mock_llm):
        agent = ReActAgent(llm_client=mock_llm, state=AgentState(mode="build"))
        prompt = agent._get_system_prompt()
        assert "coding agent" in prompt.lower() or "tool" in prompt.lower()

    def test_plan_mode_prompt(self, mock_llm):
        agent = ReActAgent(llm_client=mock_llm, state=AgentState(mode="plan"))
        prompt = agent._get_system_prompt()
        assert "plan" in prompt.lower()

    def test_custom_system_prompt(self, mock_llm):
        agent = ReActAgent(llm_client=mock_llm, state=AgentState(mode="build"))
        agent._custom_system_prompt = "Custom prompt here"
        prompt = agent._get_system_prompt()
        assert prompt == "Custom prompt here"


class TestReActAgentRun:
    @pytest.fixture
    def mock_llm(self):
        llm = AsyncMock()
        llm.chat = AsyncMock(return_value=MagicMock(content="Hello!", tool_calls=None))
        return llm

    @pytest.fixture
    def agent(self, mock_llm):
        return ReActAgent(
            llm_client=mock_llm,
            tools=[
                ToolDefinition(name="read", description="Read", parameters={}),
            ],
        )

    @pytest.mark.asyncio
    async def test_run_simple(self, agent, mock_llm):
        result = await agent.run("Say hello")
        assert result.success is True
        assert result.response == "Hello!"

    @pytest.mark.asyncio
    async def test_run_with_tool(self, agent, mock_llm):
        mock_llm.chat.side_effect = [
            MagicMock(
                content=None,
                tool_calls=[
                    MagicMock(
                        id="call_1",
                        function=MagicMock(name="read", arguments='{"path": "/test"}'),
                    )
                ],
            ),
            MagicMock(content="Done!", tool_calls=None),
        ]
        agent.set_tool_executor(lambda n, a: ("file content", None))
        result = await agent.run("Read /test")
        assert result.success is True
        assert len(result.tool_calls) == 1

    @pytest.mark.asyncio
    async def test_run_max_turns_exceeded(self, mock_llm):
        agent = ReActAgent(
            llm_client=mock_llm,
            tools=[ToolDefinition(name="read", description="Read", parameters={})],
            max_turns=2,
        )
        mock_llm.chat.return_value = MagicMock(
            content=None,
            tool_calls=[MagicMock(id="1", function=MagicMock(name="read", arguments="{}"))],
        )
        agent.set_tool_executor(lambda n, a: ("result", None))
        result = await agent.run("Keep reading")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_run_timeout(self, mock_llm):
        agent = ReActAgent(
            llm_client=mock_llm,
            tools=[],
            timeout=0.01,
        )
        with patch("src.agent.react.time.time", side_effect=[0.0, 100.0]):
            mock_llm.chat.return_value = MagicMock(content="Done", tool_calls=None)
            result = await agent.run("Test")
            assert result.success is False

    @pytest.mark.asyncio
    async def test_run_sets_goal(self, agent):
        await agent.run("Test goal")
        assert agent.state.goal == "Test goal"

    @pytest.mark.asyncio
    async def test_run_adds_user_message(self, agent):
        await agent.run("User input")
        assert any(m.role == "user" and m.content == "User input" for m in agent.state.messages)

    @pytest.mark.asyncio
    async def test_run_compression_triggered(self, agent, mock_llm):
        agent.state.needs_compression = MagicMock(return_value=True)
        agent._compress_context = AsyncMock()
        await agent.run("Test")
        agent._compress_context.assert_awaited()

    @pytest.mark.asyncio
    async def test_run_error_handling(self, agent, mock_llm):
        mock_llm.chat.side_effect = ValueError("LLM error")
        result = await agent.run("Test")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_run_parallel_tool_calls(self, agent, mock_llm):
        mock_llm.chat.side_effect = [
            MagicMock(
                content=None,
                tool_calls=[
                    MagicMock(id="1", function=MagicMock(name="read", arguments='{"path": "a"}')),
                    MagicMock(id="2", function=MagicMock(name="read", arguments='{"path": "b"}')),
                ],
            ),
            MagicMock(content="Done!", tool_calls=None),
        ]
        agent.set_tool_executor(lambda n, a: ("content", None))
        result = await agent.run("Read both")
        assert result.success is True
        assert len(result.tool_calls) == 2


class TestReActAgentRunStream:
    @pytest.fixture
    def mock_llm(self):
        llm = AsyncMock()
        llm.chat = AsyncMock(return_value=MagicMock(content="Done!", tool_calls=None))
        return llm

    @pytest.fixture
    def agent(self, mock_llm):
        return ReActAgent(
            llm_client=mock_llm,
            tools=[
                ToolDefinition(name="read", description="Read", parameters={}),
            ],
        )

    @pytest.mark.asyncio
    async def test_stream_events(self, agent, mock_llm):
        events = []
        async for event in agent.run_stream("Test"):
            events.append(event)
        assert any(e["type"] == "start" for e in events)
        assert any(e["type"] == "done" for e in events)

    @pytest.mark.asyncio
    async def test_stream_response_event(self, agent, mock_llm):
        events = []
        async for event in agent.run_stream("Test"):
            events.append(event)
        response_events = [e for e in events if e["type"] == "response"]
        assert len(response_events) == 1
        assert response_events[0]["content"] == "Done!"

    @pytest.mark.asyncio
    async def test_stream_tool_call_events(self, agent, mock_llm):
        mock_llm.chat.side_effect = [
            MagicMock(
                content=None,
                tool_calls=[
                    MagicMock(id="1", function=MagicMock(name="read", arguments='{"path": "a"}')),
                    MagicMock(id="2", function=MagicMock(name="read", arguments='{"path": "b"}')),
                ],
            ),
            MagicMock(content="Done", tool_calls=None),
        ]
        agent.set_tool_executor(lambda n, a: ("content", None))
        events = []
        async for event in agent.run_stream("Test"):
            events.append(event)
        tool_events = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_events) == 2

    @pytest.mark.asyncio
    async def test_stream_thinking_event(self, agent, mock_llm):
        mock_llm.chat.return_value = MagicMock(
            content="Done",
            tool_calls=None,
            thought="My reasoning",
        )
        events = []
        async for event in agent.run_stream("Test"):
            events.append(event)
        thinking_events = [e for e in events if e["type"] == "thinking"]
        assert len(thinking_events) >= 1

    @pytest.mark.asyncio
    async def test_stream_timeout(self, agent, monkeypatch):
        timestamps = iter([0.0, 1.0])
        monkeypatch.setattr("src.agent.react.time.time", lambda: next(timestamps))
        agent.timeout = 0.5
        events = []
        async for event in agent.run_stream("Test"):
            events.append(event)
        assert any(event["type"] == "error" and "timeout" in event["error"] for event in events)
        assert events[-1]["type"] == "done"

    @pytest.mark.asyncio
    async def test_stream_compression(self, agent, mock_llm):
        agent.state.needs_compression = MagicMock(return_value=True)
        agent._compress_context = AsyncMock()
        async for _ in agent.run_stream("Test"):
            pass
        agent._compress_context.assert_awaited()

    @pytest.mark.asyncio
    async def test_stream_max_turns(self, mock_llm):
        agent = ReActAgent(
            llm_client=mock_llm,
            tools=[ToolDefinition(name="read", description="Read", parameters={})],
            max_turns=2,
        )
        mock_llm.chat.return_value = MagicMock(
            content=None,
            tool_calls=[MagicMock(id="1", function=MagicMock(name="read", arguments="{}"))],
        )
        agent.set_tool_executor(lambda n, a: ("result", None))
        events = []
        async for event in agent.run_stream("Test"):
            events.append(event)
        error_events = [e for e in events if e["type"] == "error"]
        assert len(error_events) >= 1


class TestCompressContext:
    @pytest.fixture
    def mock_llm(self):
        llm = AsyncMock()
        llm.chat = AsyncMock(return_value=MagicMock(content="Compressed summary", tool_calls=None))
        return llm

    @pytest.fixture
    def agent(self, mock_llm):
        return ReActAgent(llm_client=mock_llm, tools=[])

    @pytest.mark.asyncio
    async def test_compress_context_reduces_messages(self, agent, mock_llm):
        for i in range(20):
            agent.state.add_user_message(f"Message {i}")
        original_count = len(agent.state.messages)
        await agent._compress_context()
        assert len(agent.state.messages) < original_count or len(agent.state.messages) <= 13

    @pytest.mark.asyncio
    async def test_compress_context_adds_summary_message(self, agent, mock_llm):
        for i in range(20):
            agent.state.add_user_message(f"Message {i}")
        await agent._compress_context()
        summary_msgs = [
            m
            for m in agent.state.messages
            if m.role == "system" and m.content and "Semantic Context" in m.content
        ]
        assert len(summary_msgs) == 1

    @pytest.mark.asyncio
    async def test_compress_context_skips_when_few_messages(self, agent, mock_llm):
        agent.state.add_user_message("Only one")
        await agent._compress_context()
        mock_llm.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_compress_context_handles_llm_error(self, agent, mock_llm):
        mock_llm.chat.side_effect = RuntimeError("LLM failed")
        for i in range(20):
            agent.state.add_user_message(f"Msg {i}")
        await agent._compress_context()
        assert len(agent.state.messages) == 20


class TestExecuteToolsParallel:
    @pytest.fixture
    def mock_llm(self):
        return AsyncMock()

    @pytest.fixture
    def agent(self, mock_llm):
        tools = [
            ToolDefinition(name="read", description="Read", parameters={}),
            ToolDefinition(name="write", description="Write", parameters={}, sensitive=True),
        ]
        agent = ReActAgent(llm_client=mock_llm, tools=tools)
        agent.permission_callback = AsyncMock(return_value="yes")
        return agent

    @pytest.mark.asyncio
    async def test_parallel_execution(self, agent):
        executor = AsyncMock(
            side_effect=[
                ("result_a", None),
                ("result_b", None),
            ]
        )
        agent.set_tool_executor(executor)
        tool_calls = [
            {"name": "read", "arguments": {"path": "a"}},
            {"name": "read", "arguments": {"path": "b"}},
        ]
        results = await agent._execute_tools_parallel(tool_calls)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_sequential_for_write_tool(self, agent):
        executor = AsyncMock(
            side_effect=[
                ("written", None),
                ("read_result", None),
            ]
        )
        agent.set_tool_executor(executor)
        tool_calls = [
            {"name": "write", "arguments": {"path": "a"}},
            {"name": "read", "arguments": {"path": "b"}},
        ]
        results = await agent._execute_tools_parallel(tool_calls)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_parallel_with_string_args(self, agent):
        executor = AsyncMock(return_value=("result", None))
        agent.set_tool_executor(executor)
        tool_calls = [
            {"name": "read", "arguments": '{"path": "a"}'},
        ]
        results = await agent._execute_tools_parallel(tool_calls)
        assert len(results) == 1


class TestBuildResult:
    @pytest.fixture
    def mock_llm(self):
        return AsyncMock()

    @pytest.fixture
    def agent(self, mock_llm):
        agent = ReActAgent(llm_client=mock_llm, tools=[])
        agent._start_time = 100.0
        return agent

    def test_build_result_success(self, agent, monkeypatch):
        monkeypatch.setattr("src.agent.react.time.time", lambda: 102.0)
        agent.state.status = "finished"
        agent.state.last_response = "Done"
        result = agent._build_result()
        assert result.success is True
        assert result.response == "Done"
        assert result.duration == 2.0

    def test_build_result_with_error(self, agent, monkeypatch):
        monkeypatch.setattr("src.agent.react.time.time", lambda: 101.0)
        result = agent._build_result(error="Something failed")
        assert result.success is False
        assert result.error == "Something failed"

    def test_build_result_when_state_has_error(self, agent, monkeypatch):
        monkeypatch.setattr("src.agent.react.time.time", lambda: 101.0)
        agent.state.status = "error"
        agent.state.last_error = "State error"
        result = agent._build_result()
        assert result.success is False
        assert result.error == "State error"

    def test_build_result_metadata(self, agent, monkeypatch):
        monkeypatch.setattr("src.agent.react.time.time", lambda: 101.0)
        agent.state.turn_count = 5
        agent.state.add_tool_call(ToolCall(id="1", name="read", arguments={}))
        result = agent._build_result()
        assert result.metadata["turn_count"] == 5
        assert result.metadata["tool_call_count"] == 1

    def test_build_result_no_start_time(self, agent):
        agent._start_time = None
        result = agent._build_result()
        assert result.duration == 0


class TestEstimateTokens:
    @pytest.fixture
    def agent(self):
        llm = AsyncMock()
        return ReActAgent(llm_client=llm, tools=[])

    def test_empty_messages(self, agent):
        assert agent._estimate_tokens([]) == 0

    def test_with_content(self, agent):
        msgs = [{"content": "a" * 100}]
        assert agent._estimate_tokens(msgs) == 25

    def test_with_multiple_messages(self, agent):
        msgs = [{"content": "hello"}, {"content": "world"}]
        assert agent._estimate_tokens(msgs) > 0

    def test_with_no_content_key(self, agent):
        msgs = [{}]
        assert agent._estimate_tokens(msgs) == 0


class TestCallLlm:
    @pytest.fixture
    def mock_llm(self):
        llm = AsyncMock()
        llm.chat = AsyncMock(return_value=MagicMock(content="response", tool_calls=None))
        return llm

    @pytest.fixture
    def agent(self, mock_llm):
        return ReActAgent(llm_client=mock_llm, tools=[])

    @pytest.mark.asyncio
    async def test_call_llm_with_chat(self, agent, mock_llm):
        result = await agent._call_llm([{"role": "user", "content": "Hi"}], [])
        assert result["content"] == "response"
        mock_llm.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_llm_with_ainvoke(self, mock_llm):
        del mock_llm.chat
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="invoked", tool_calls=None))
        agent = ReActAgent(llm_client=mock_llm, tools=[])
        result = await agent._call_llm([{"role": "user", "content": "Hi"}], [])
        assert result["content"] == "invoked"

    @pytest.mark.asyncio
    async def test_call_llm_with_callable(self, mock_llm):
        async def my_callable(messages, tools=None):
            return MagicMock(content="called", tool_calls=None)

        agent = ReActAgent(llm_client=my_callable, tools=[])
        result = await agent._call_llm([{"role": "user", "content": "Hi"}], [])
        assert result["content"] == "called"

    @pytest.mark.asyncio
    async def test_call_llm_no_valid_method(self, agent):
        bad_llm = SimpleNamespace()
        agent.llm = bad_llm
        with pytest.raises(ValueError, match="chat, ainvoke, or be callable"):
            await agent._call_llm([{"role": "user", "content": "Hi"}], [])

    @pytest.mark.asyncio
    async def test_call_llm_traces(self, agent, mock_llm):
        trace = TraceInterface()
        trace.llm_call = MagicMock()
        agent.set_trace(trace)
        await agent._call_llm([{"role": "user", "content": "Hi"}], [])
        trace.llm_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_llm_trace_on_error(self, agent, mock_llm):
        trace = TraceInterface()
        trace.event = MagicMock()
        agent.set_trace(trace)
        mock_llm.chat.side_effect = ValueError("bad")
        with pytest.raises(ValueError):
            await agent._call_llm([{"role": "user", "content": "Hi"}], [])
        trace.event.assert_called_once()
        assert trace.event.call_args[0][0] == "error"


class TestReActAgentReset:
    @pytest.fixture
    def mock_llm(self):
        return AsyncMock()

    def test_reset_clears_state(self, mock_llm):
        agent = ReActAgent(llm_client=mock_llm, tools=[])
        agent.state.add_user_message("Test")
        agent.state.turn_count = 5
        agent.reset()
        assert len(agent.state.messages) == 0
        assert agent.state.turn_count == 0
        assert agent.state.is_finished is False


class TestGetSummary:
    @pytest.fixture
    def mock_llm(self):
        return AsyncMock()

    def test_get_summary(self, mock_llm):
        agent = ReActAgent(
            llm_client=mock_llm,
            tools=[
                ToolDefinition(name="read", description="Read", parameters={}),
                ToolDefinition(name="write", description="Write", parameters={}, sensitive=True),
            ],
        )
        summary = agent.get_summary()
        assert summary["tools_registered"] == 2
        assert "write" in summary["sensitive_tools"]
