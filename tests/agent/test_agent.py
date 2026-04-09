"""
Agent Module Tests

Tests for the standard agent abstraction.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent import (
    Action,
    ActionType,
    AgentResult,
    AgentState,
    DoraemonAgent,
    Observation,
    ReActAgent,
    Thought,
    ToolCall,
    ToolDefinition,
    create_doraemon_agent,
)
from src.agent.adapter import _collect_modified_paths
from src.core.home import Trace
from src.core.tasks import TaskManager, TaskStatus
from src.host.tools import LazyToolFunction


class TestAgentState:
    """Tests for AgentState."""

    def test_initial_state(self):
        """State should initialize with defaults."""
        state = AgentState()
        assert state.messages == []
        assert state.tool_history == []
        assert state.mode == "build"
        assert state.is_finished is False
        assert state.status == "idle"

    def test_add_user_message(self):
        """Should add user message correctly."""
        state = AgentState()
        state.add_user_message("Hello")

        assert len(state.messages) == 1
        assert state.messages[0].role == "user"
        assert state.messages[0].content == "Hello"
        assert state.user_input == "Hello"

    def test_add_assistant_message(self):
        """Should add assistant message correctly."""
        state = AgentState()
        state.add_assistant_message(
            content="Hi there",
            tool_calls=[{"name": "test", "arguments": {}}],
        )

        assert len(state.messages) == 1
        assert state.messages[0].role == "assistant"
        assert state.messages[0].content == "Hi there"

    def test_add_tool_result(self):
        """Should add tool result correctly."""
        state = AgentState()
        state.add_tool_result("call_123", "read", "file content")

        assert len(state.messages) == 1
        assert state.messages[0].role == "tool"
        assert state.messages[0].name == "read"

    def test_increment_turn(self):
        """Should increment turn counter."""
        state = AgentState(max_turns=3)

        assert state.increment_turn() is True
        assert state.turn_count == 1
        assert state.increment_turn() is True
        assert state.turn_count == 2
        assert state.increment_turn() is False  # Exceeded

    def test_needs_compression(self):
        """Should detect when compression needed."""
        state = AgentState(max_context_tokens=1000)
        state.estimated_tokens = 500
        assert state.needs_compression() is False

        state.estimated_tokens = 950
        assert state.needs_compression() is True

    def test_get_recent_messages(self):
        """Should get recent N messages."""
        state = AgentState()
        for i in range(20):
            state.add_user_message(f"Message {i}")

        recent = state.get_recent_messages(5)
        assert len(recent) == 5
        assert "Message 19" in recent[-1].content

    def test_tool_call_tracking(self):
        """Should track tool calls correctly."""
        state = AgentState()
        state.add_tool_call(
            ToolCall(
                id="1",
                name="read",
                arguments={},
                result="success",
            )
        )
        state.add_tool_call(
            ToolCall(
                id="2",
                name="write",
                arguments={},
                error="failed",
            )
        )

        assert state.get_tool_call_count() == 2
        assert len(state.get_successful_tool_calls()) == 1
        assert len(state.get_failed_tool_calls()) == 1

    def test_checkpoint(self):
        """Should create and restore checkpoints."""
        state = AgentState(mode="plan", goal="test")
        state.add_user_message("Hello")

        checkpoint = state.create_checkpoint()

        new_state = AgentState()
        new_state.restore_checkpoint(checkpoint)

        assert new_state.mode == "plan"
        assert new_state.goal == "test"

    def test_clear_history_resets_runtime_fields(self):
        """Clearing history should also reset runtime-facing state."""
        state = AgentState(mode="build")
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


class TestToolDefinition:
    """Tests for ToolDefinition."""

    def test_to_api_format(self):
        """Should convert to API format."""
        tool = ToolDefinition(
            name="read",
            description="Read a file",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        )

        api_format = tool.to_api_format()

        assert api_format["type"] == "function"
        assert api_format["function"]["name"] == "read"
        assert api_format["function"]["description"] == "Read a file"

    def test_sensitive_tool(self):
        """Should mark sensitive tools."""
        tool = ToolDefinition(
            name="write",
            description="Delete a file",
            parameters={},
            sensitive=True,
        )

        assert tool.sensitive is True


class TestAction:
    """Tests for Action."""

    def test_tool_call_action(self):
        """Should create tool call action."""
        action = Action.tool_call("read", {"path": "/test"}, "call_1")

        assert action.type == ActionType.TOOL_CALL
        assert action.tool_name == "read"
        assert action.tool_args == {"path": "/test"}

    def test_respond_action(self):
        """Should create respond action."""
        action = Action.respond("Hello!")

        assert action.type == ActionType.RESPOND
        assert action.response == "Hello!"

    def test_finish_action(self):
        """Should create finish action."""
        action = Action.finish("Done")

        assert action.type == ActionType.FINISH
        assert action.response == "Done"


class TestReActAgent:
    """Tests for ReActAgent."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM client."""
        llm = AsyncMock()
        llm.chat = AsyncMock(
            return_value=MagicMock(
                content="Hello!",
                tool_calls=None,
            )
        )
        return llm

    @pytest.fixture
    def tools(self):
        """Create test tools."""
        return [
            ToolDefinition(
                name="read",
                description="Read a file",
                parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            ),
            ToolDefinition(
                name="write",
                description="Write a file",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                },
                sensitive=True,
            ),
        ]

    @pytest.fixture
    def agent(self, mock_llm, tools):
        """Create test agent."""
        return ReActAgent(
            llm_client=mock_llm,
            tools=tools,
            max_turns=10,
        )

    def test_agent_initialization(self, agent, tools):
        """Should initialize agent correctly."""
        assert len(agent.tools) == 2
        assert agent.is_sensitive_tool("write") is True
        assert agent.is_sensitive_tool("read") is False

    def test_register_tool(self, agent):
        """Should register new tools."""
        new_tool = ToolDefinition(
            name="delete",
            description="Delete",
            parameters={},
            sensitive=True,
        )
        agent.register_tool(new_tool)

        assert len(agent.tools) == 3
        assert agent.is_sensitive_tool("delete") is True

    @pytest.mark.asyncio
    async def test_observe(self, agent):
        """Should observe environment correctly."""
        agent.state.add_user_message("Test")

        observation = await agent.observe()

        assert observation.user_input == "Test"
        assert observation.context["goal"] is None

    @pytest.mark.asyncio
    async def test_think(self, agent, mock_llm):
        """Should think and return thought."""
        agent.state.add_user_message("Hello")

        thought = await agent.think(Observation(user_input="Hello"))

        assert thought.response == "Hello!"
        assert thought.is_finished is True

    def test_parse_llm_response_preserves_thought_signature(self, agent):
        """Should preserve Gemini thought signatures on tool calls."""
        response = {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "thought_signature": "sig_123",
                    "function": {
                        "name": "write",
                        "arguments": '{"path":"add.py","content":"def add(a, b):\\n    return a + b\\n"}',
                    },
                }
            ],
        }

        parsed = agent._parse_llm_response(response)

        assert parsed["tool_calls"][0]["thought_signature"] == "sig_123"

    @pytest.mark.asyncio
    async def test_act_respond(self, agent):
        """Should act with respond when no tool calls."""
        thought = Thought(
            reasoning="No tools needed",
            response="Done!",
            is_finished=True,
        )

        action = await agent.act(thought)

        assert action.type == ActionType.RESPOND
        assert action.response == "Done!"

    @pytest.mark.asyncio
    async def test_act_tool_call(self, agent):
        """Should act with tool call when tools needed."""
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

    @pytest.mark.asyncio
    async def test_run_simple(self, agent, mock_llm):
        """Should run simple task."""
        result = await agent.run("Say hello")

        assert result.success is True
        assert result.response == "Hello!"

    @pytest.mark.asyncio
    async def test_run_with_tool(self, agent, mock_llm):
        """Should run task with tool call."""
        mock_llm.chat.side_effect = [
            MagicMock(
                content=None,
                tool_calls=[
                    MagicMock(
                        id="call_1", function=MagicMock(name="read", arguments='{"path": "/test"}')
                    )
                ],
            ),
            MagicMock(content="Done!", tool_calls=None),
        ]

        agent.set_tool_executor(lambda name, args: ("file content", None))

        result = await agent.run("Read /test")

        assert result.success is True
        assert len(result.tool_calls) == 1

    @pytest.mark.asyncio
    async def test_max_turns_exceeded(self, mock_llm, tools):
        """Should handle max turns exceeded."""
        agent = ReActAgent(
            llm_client=mock_llm,
            tools=tools,
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
    async def test_permission_check(self, agent, mock_llm):
        """Should check permission for sensitive tools."""
        agent.permission_callback = AsyncMock(return_value="yes")

        allowed = await agent.check_permission("write", {"path": "/test"})

        assert allowed is True

    @pytest.mark.asyncio
    async def test_permission_denied(self, agent, mock_llm):
        """Should deny permission."""
        agent.permission_callback = AsyncMock(return_value="no")

        allowed = await agent.check_permission("write", {"path": "/test"})

        assert allowed is False

    @pytest.mark.asyncio
    async def test_stream_events(self, agent, mock_llm):
        """Should stream events correctly."""
        mock_llm.chat.return_value = MagicMock(
            content="Done!",
            tool_calls=None,
        )

        events = []
        async for event in agent.run_stream("Test"):
            events.append(event)

        assert any(e["type"] == "start" for e in events)
        assert any(e["type"] == "done" for e in events)
        assert agent.state.messages[-1].role == "assistant"
        assert agent.state.messages[-1].content == "Done!"

    @pytest.mark.asyncio
    async def test_stream_run_triggers_context_compression(self, agent, mock_llm):
        """Streaming runs should apply the same compression path as non-streaming runs."""
        mock_llm.chat.return_value = MagicMock(content="Done!", tool_calls=None)
        agent.state.needs_compression = MagicMock(return_value=True)
        agent._compress_context = AsyncMock()

        async for _event in agent.run_stream("Test"):
            pass

        agent._compress_context.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stream_run_emits_timeout_error(self, agent, monkeypatch):
        """Streaming runs should honor the configured timeout."""
        timestamps = iter([0.0, 1.0])
        monkeypatch.setattr("src.agent.react.time.time", lambda: next(timestamps))
        agent.timeout = 0.5

        events = []
        async for event in agent.run_stream("Test"):
            events.append(event)

        assert any(event["type"] == "error" and "timeout" in event["error"] for event in events)
        assert events[-1]["type"] == "done"
        assert events[-1]["result"]["success"] is False
        assert "timeout" in events[-1]["result"]["error"]
        assert events[-1]["result"]["metadata"]["status"] == "error"

    def test_reset(self, agent):
        """Should reset agent state."""
        agent.state.add_user_message("Test")
        agent.state.turn_count = 5

        agent.reset()

        assert len(agent.state.messages) == 0
        assert agent.state.turn_count == 0
        assert agent.state.is_finished is False


class TestUnifiedWriteTracking:
    def test_collect_modified_paths_includes_move_destination(self):
        tool_calls = [
            ToolCall(
                id="call_1",
                name="write",
                arguments={"path": "old.txt", "operation": "move", "destination": "new.txt"},
            )
        ]

        assert _collect_modified_paths(tool_calls) == ["old.txt", "new.txt"]


class TestDoraemonPrompts:
    """Prompt wording should stay agentic and avoid workflow scripts."""

    def test_build_prompt_avoids_workflow_wording(self):
        agent = DoraemonAgent(
            llm_client=AsyncMock(),
            tool_registry=SimpleNamespace(get_tool_names=lambda: [], _tools={}),
            state=AgentState(mode="build"),
        )

        prompt = agent._get_system_prompt()

        assert "WORKFLOW:" not in prompt
        assert "OPERATING PRINCIPLES:" in prompt

    def test_plan_prompt_avoids_step_by_step_wording(self):
        agent = DoraemonAgent(
            llm_client=AsyncMock(),
            tool_registry=SimpleNamespace(get_tool_names=lambda: [], _tools={}),
            state=AgentState(mode="plan"),
        )

        prompt = agent._get_system_prompt()

        assert "step-by-step plan" not in prompt
        assert "concrete implementation strategy" in prompt

    @pytest.mark.asyncio
    async def test_checkpoint_paths_include_move_destination(self):
        checkpoints = SimpleNamespace(snapshot=MagicMock())
        agent = DoraemonAgent(
            llm_client=AsyncMock(),
            tool_registry=SimpleNamespace(get_tool_names=lambda: [], _tools={}),
            hooks=None,
            checkpoints=checkpoints,
            skills=None,
            enable_trace=False,
        )

        await agent._create_checkpoint(
            "write",
            {"path": "old.txt", "operation": "move", "destination": "new.txt"},
        )

        assert [call.args[0] for call in checkpoints.snapshot.call_args_list] == [
            "old.txt",
            "new.txt",
        ]


class TestAgentIntegration:
    """Integration tests for agent system."""

    @pytest.mark.asyncio
    async def test_full_react_loop(self):
        """Test complete ReAct loop."""
        llm = AsyncMock()

        llm.chat.side_effect = [
            MagicMock(
                content=None,
                tool_calls=[
                    MagicMock(
                        id="1", function=MagicMock(name="read", arguments='{"path": "/file.py"}')
                    )
                ],
            ),
            MagicMock(
                content=None,
                tool_calls=[
                    MagicMock(
                        id="2",
                        function=MagicMock(
                            name="write", arguments='{"path": "/new.py", "content": "test"}'
                        ),
                    )
                ],
            ),
            MagicMock(content="Task complete!", tool_calls=None),
        ]

        tool_results = {
            "read": ("original content", None),
            "write": ("written successfully", None),
        }

        agent = ReActAgent(
            llm_client=llm,
            tools=[
                ToolDefinition(name="read", description="Read", parameters={}),
                ToolDefinition(name="write", description="Write", parameters={}, sensitive=True),
            ],
        )

        agent.set_tool_executor(lambda n, a: tool_results.get(n, ("", "unknown tool")))
        agent.permission_callback = AsyncMock(return_value="yes")

        result = await agent.run("Copy /file.py to /new.py")

        assert result.success is True
        assert len(result.tool_calls) == 2
        assert result.response == "Task complete!"

    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test error recovery in agent."""
        llm = AsyncMock()

        llm.chat.side_effect = [
            MagicMock(
                content=None,
                tool_calls=[
                    MagicMock(id="1", function=MagicMock(name="read", arguments='{"path": "/bad"}'))
                ],
            ),
            MagicMock(
                content=None,
                tool_calls=[
                    MagicMock(
                        id="2", function=MagicMock(name="read", arguments='{"path": "/good"}')
                    )
                ],
            ),
            MagicMock(content="Done", tool_calls=None),
        ]

        call_count = 0

        def executor(name, args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ("", "File not found")
            return ("content", None)

        agent = ReActAgent(
            llm_client=llm, tools=[ToolDefinition(name="read", description="Read", parameters={})]
        )
        agent.set_tool_executor(executor)

        result = await agent.run("Read file")

        assert result.success is True


class TestAgentStateSerialization:
    """Tests for state serialization."""

    def test_to_dict(self):
        """Should serialize to dict."""
        state = AgentState(mode="plan", turn_count=5)
        state.add_user_message("Test")

        data = state.to_dict()

        assert data["mode"] == "plan"
        assert data["turn_count"] == 5
        assert data["message_count"] == 1

    def test_from_dict(self):
        """Should deserialize from dict."""
        data = {
            "mode": "build",
            "turn_count": 3,
            "is_finished": True,
            "status": "finished",
        }

        state = AgentState.from_dict(data)

        assert state.mode == "build"
        assert state.turn_count == 3
        assert state.is_finished is True


class TestAgentResult:
    """Tests for AgentResult."""

    def test_to_dict(self):
        """Should serialize to dict."""
        result = AgentResult(
            success=True,
            response="Done",
            tool_calls=[
                ToolCall(id="1", name="read", arguments={}, result="content"),
            ],
            tokens_used=100,
            duration=1.5,
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["response"] == "Done"
        assert len(data["tool_calls"]) == 1
        assert data["tokens_used"] == 100


class TestDoraemonAgent:
    """Tests for DoraemonAgent."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM client."""
        llm = AsyncMock()
        llm.chat = AsyncMock(
            return_value=MagicMock(
                content="Done!",
                tool_calls=None,
            )
        )
        return llm

    @pytest.fixture
    def mock_registry(self):
        """Create mock tool registry."""
        registry = MagicMock()
        registry.get_tool_names = MagicMock(return_value=["read", "write"])
        registry._tools = {
            "read": ToolDefinition(name="read", description="Read file", parameters={}),
            "write": ToolDefinition(
                name="write", description="Write file", parameters={}, sensitive=True
            ),
        }
        registry.call_tool = AsyncMock(return_value="file content")
        return registry

    @pytest.fixture
    def mock_hooks(self):
        """Create mock hook manager."""
        hook_mgr = AsyncMock()
        hook_mgr.trigger = AsyncMock()
        return hook_mgr

    def test_create_doraemon_agent(self, mock_llm, mock_registry):
        """Should create DoraemonAgent via factory."""
        agent = create_doraemon_agent(
            llm_client=mock_llm,
            tool_registry=mock_registry,
            mode="build",
        )

        assert isinstance(agent, DoraemonAgent)
        assert agent.state.mode == "build"
        assert len(agent.tools) == 2

    def test_skips_unavailable_lazy_tools(self, mock_llm):
        """Unavailable lazy tools should not be exposed to the model."""
        broken_lazy = LazyToolFunction("missing.module", "browser_click")
        try:
            broken_lazy._load()
        except ImportError:
            pass

        registry = MagicMock()
        registry.get_tool_names = MagicMock(return_value=["read", "browser_click"])
        registry._tools = {
            "read": SimpleNamespace(
                name="read",
                description="Read file",
                parameters={"type": "object", "properties": {"path": {"type": "string"}}},
                sensitive=False,
                function=lambda path: path,
            ),
            "browser_click": SimpleNamespace(
                name="browser_click",
                description="Broken browser tool",
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": True,
                },
                sensitive=False,
                function=broken_lazy,
            ),
        }

        agent = create_doraemon_agent(
            llm_client=mock_llm,
            tool_registry=registry,
            mode="build",
        )

        assert [tool.name for tool in agent.tools] == ["read"]

    @pytest.mark.asyncio
    async def test_doraemon_agent_run(self, mock_llm, mock_registry):
        """Should run DoraemonAgent."""
        agent = create_doraemon_agent(
            llm_client=mock_llm,
            tool_registry=mock_registry,
            mode="build",
        )

        result = await agent.run("Hello")

        assert result.success is True
        assert result.response == "Done!"

    @pytest.mark.asyncio
    async def test_doraemon_agent_run_records_trace_run_id(self, mock_llm, mock_registry):
        """Agent turns should carry orchestration run identity into trace metadata."""
        trace = Trace("session_test")
        agent = create_doraemon_agent(
            llm_client=mock_llm,
            tool_registry=mock_registry,
            mode="build",
            enable_trace=True,
            trace=trace,
        )

        result = await agent.run("Hello", trace_run_id="run-42")

        assert result.success is True
        assert trace.events[0].type == "turn_start"
        assert trace.events[0].data["run_id"] == "run-42"

    @pytest.mark.asyncio
    async def test_doraemon_agent_with_hooks(self, mock_llm, mock_registry, mock_hooks):
        """Should trigger hooks during execution."""
        agent = create_doraemon_agent(
            llm_client=mock_llm,
            tool_registry=mock_registry,
            mode="build",
            hooks=mock_hooks,
        )

        await agent.run("Hello")

        mock_hooks.trigger.assert_called()

    @pytest.mark.asyncio
    async def test_doraemon_agent_tool_execution(self, mock_llm, mock_registry):
        """Should execute tools from registry."""
        mock_llm.chat.side_effect = [
            MagicMock(
                content=None,
                tool_calls=[
                    MagicMock(
                        id="1", function=MagicMock(name="read", arguments='{"path": "/test"}')
                    )
                ],
            ),
            MagicMock(content="Done", tool_calls=None),
        ]

        agent = create_doraemon_agent(
            llm_client=mock_llm,
            tool_registry=mock_registry,
            mode="build",
        )

        result = await agent.run("Read /test")

        assert result.success is True
        assert len(result.tool_calls) == 1
        mock_registry.call_tool.assert_called_once()

    @pytest.mark.asyncio
    async def test_doraemon_agent_tool_trace_includes_run_id(self, mock_llm, mock_registry):
        """Tool-call trace events should include the active orchestration run ID."""
        mock_llm.chat.side_effect = [
            MagicMock(
                content=None,
                tool_calls=[
                    MagicMock(
                        id="1", function=MagicMock(name="read", arguments='{"path": "/test"}')
                    )
                ],
            ),
            MagicMock(content="Done", tool_calls=None),
        ]
        trace = Trace("session_test")
        agent = create_doraemon_agent(
            llm_client=mock_llm,
            tool_registry=mock_registry,
            mode="build",
            enable_trace=True,
            trace=trace,
        )

        result = await agent.run("Read /test", trace_run_id="run-99")

        tool_event = next(event for event in trace.events if event.type == "tool_call")
        assert result.success is True
        assert tool_event.data["run_id"] == "run-99"

    @pytest.mark.asyncio
    async def test_doraemon_agent_policy_blocks_write_in_plan_mode(self, mock_llm):
        """Plan mode should reject write execution through shared tool policy."""
        registry = MagicMock()
        registry.get_tool_names = MagicMock(return_value=["read", "write"])
        registry._tools = {
            "read": SimpleNamespace(
                name="read",
                description="Read file",
                parameters={},
                sensitive=False,
                source="built_in",
                metadata={"capability_group": "read"},
            ),
            "write": SimpleNamespace(
                name="write",
                description="Write file",
                parameters={},
                sensitive=True,
                source="built_in",
                metadata={"capability_group": "edit"},
            ),
        }
        registry.get_tool_policy.side_effect = lambda name, mode=None, active_mcp_extensions=None: {
            "read": {
                "tool_name": "read",
                "visible": True,
                "requires_approval": False,
            },
            "write": {
                "tool_name": "write",
                "visible": mode != "plan",
                "requires_approval": True,
            },
        }[name]
        registry.call_tool = AsyncMock(return_value="written")

        agent = create_doraemon_agent(
            llm_client=mock_llm,
            tool_registry=registry,
            mode="plan",
        )

        result, error = await agent.execute_tool("write", {"path": "a.py", "content": "x"})

        assert result == ""
        assert "not available in plan mode" in error
        registry.call_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_doraemon_agent_does_not_persist_runtime_task_for_direct_turns(
        self, mock_llm, tmp_path
    ):
        """Direct turns should not pollute the persistent orchestration task graph."""
        task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
        registry = MagicMock()
        registry.get_tool_names = MagicMock(return_value=["read"])
        registry._tools = {
            "read": SimpleNamespace(
                name="read",
                description="Read file",
                parameters={},
                sensitive=False,
                source="built_in",
                metadata={"capability_group": "read"},
            )
        }
        registry.call_tool = AsyncMock(return_value="content")

        agent = create_doraemon_agent(
            llm_client=mock_llm,
            tool_registry=registry,
            mode="build",
            task_manager=task_manager,
        )

        result = await agent.run("Inspect README and summarize")

        assert result.success is True
        assert task_manager.list_tasks() == []

    @pytest.mark.asyncio
    async def test_doraemon_agent_can_opt_in_to_runtime_task_persistence(self, mock_llm, tmp_path):
        """Explicit runtime-task persistence remains available for callers that need it."""
        task_manager = TaskManager(storage_path=tmp_path / "tasks.json")
        registry = MagicMock()
        registry.get_tool_names = MagicMock(return_value=["read"])
        registry._tools = {
            "read": SimpleNamespace(
                name="read",
                description="Read file",
                parameters={},
                sensitive=False,
                source="built_in",
                metadata={"capability_group": "read"},
            )
        }
        registry.call_tool = AsyncMock(return_value="content")

        agent = create_doraemon_agent(
            llm_client=mock_llm,
            tool_registry=registry,
            mode="build",
            task_manager=task_manager,
        )

        result = await agent.run("Inspect README and summarize", create_runtime_task=True)

        assert result.success is True
        tasks = task_manager.list_tasks()
        assert len(tasks) == 1
        assert tasks[0].status == TaskStatus.COMPLETED
        assert tasks[0].assigned_agent is None
        assert tasks[0].description == "Inspect README and summarize"


class TestAgentAdapter:
    """Tests for agent adapter functions."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM client."""
        llm = AsyncMock()
        llm.chat = AsyncMock(
            return_value=MagicMock(
                content="Response",
                tool_calls=None,
            )
        )
        return llm

    @pytest.fixture
    def mock_registry(self):
        """Create mock tool registry."""
        registry = MagicMock()
        registry.get_tool_names = MagicMock(return_value=["read"])
        registry._tools = {
            "read": ToolDefinition(name="read", description="Read", parameters={}),
        }
        registry.call_tool = AsyncMock(return_value="content")
        return registry

    @pytest.mark.asyncio
    async def test_run_agent_turn(self, mock_llm, mock_registry):
        """Should run agent turn and return result."""
        from src.agent.adapter import run_agent_turn

        result = await run_agent_turn(
            user_input="Hello",
            model_client=mock_llm,
            registry=mock_registry,
            context=None,
            mode="build",
        )

        assert result.success is True
        assert result.response == "Response"

    @pytest.mark.asyncio
    async def test_agent_session(self, mock_llm, mock_registry, tmp_path):
        """Should manage agent session."""
        from src.agent.adapter import AgentSession

        session = AgentSession(
            model_client=mock_llm,
            registry=mock_registry,
            mode="build",
            project_dir=tmp_path,
            enable_trace=False,
        )

        await session.initialize()

        assert session._agent is not None
        assert session._state is not None

        result = await session.turn("Hello")

        assert result.success is True

        session.reset()

        assert session._state is None

    @pytest.mark.asyncio
    async def test_agent_session_mode_change_rebuilds_visible_tools(self, mock_llm, tmp_path):
        """Should rebuild visible tools when mode changes."""
        from src.agent.adapter import AgentSession

        registry = MagicMock()
        registry.get_tool_names = MagicMock(return_value=["read", "write"])
        registry._tool_schemas = {
            "read": {"name": "read", "description": "Read", "parameters": {}},
            "write": {"name": "write", "description": "Write", "parameters": {}},
        }
        registry._tools = {}
        registry._sensitive_tools = set()
        registry.get_tool_policy = MagicMock(
            side_effect=lambda name, mode="build", active_mcp_extensions=None: {
                "visible": not (name == "write" and mode == "plan"),
                "requires_approval": False,
                "sandbox": "workspace-write",
                "audit_level": "full",
                "background_safe": False,
            }
        )

        session = AgentSession(
            model_client=mock_llm,
            registry=registry,
            mode="build",
            project_dir=tmp_path,
        )

        await session.initialize()

        assert {tool.name for tool in session._agent.tools} == {"read", "write"}

        await session.set_mode("plan")

        assert session.mode == "plan"
        assert session._state.mode == "plan"
        assert {tool.name for tool in session._agent.tools} == {"read"}

    @pytest.mark.asyncio
    async def test_agent_session_mode_change_rebuilds_runtime_registry(self, mock_llm, tmp_path):
        """Bootstrapped sessions should rebuild their mode-specific registry on mode changes."""
        from src.agent.adapter import AgentSession

        config_path = tmp_path / "config.json"
        config_path.write_text("{}", encoding="utf-8")

        session = AgentSession(
            model_client=mock_llm,
            registry=None,
            mode="plan",
            project_dir=tmp_path,
            config_path=config_path,
            enable_trace=False,
        )

        await session.initialize()

        assert "write" not in session.registry.get_tool_names()
        assert {tool.name for tool in session._agent.tools} == {
            "ask_user",
            "memory_get",
            "memory_list",
            "memory_put",
            "memory_search",
            "read",
            "search",
            "task",
            "web_fetch",
            "web_search",
        }

        await session.set_mode("build")

        assert "write" in session.registry.get_tool_names()
        assert "run" in session.registry.get_tool_names()
        assert "write" in {tool.name for tool in session._agent.tools}
        assert "run" in {tool.name for tool in session._agent.tools}

    @pytest.mark.asyncio
    async def test_agent_session_persists_and_restores_messages(self, mock_llm, mock_registry, tmp_path):
        """Should persist session messages and restore them by session ID."""
        from src.agent.adapter import AgentSession

        session = AgentSession(
            model_client=mock_llm,
            registry=mock_registry,
            mode="build",
            project_dir=tmp_path,
            enable_trace=False,
        )

        await session.initialize()
        session_id = session.session_id
        result = await session.turn("Hello")
        assert result.success is True
        await session.aclose()

        resumed = AgentSession(
            model_client=mock_llm,
            registry=mock_registry,
            mode="build",
            project_dir=tmp_path,
            session_id=session_id,
            enable_trace=False,
        )
        await resumed.initialize()

        assert resumed.session_id == session_id
        assert [message.role for message in resumed._state.messages] == ["user", "assistant"]
        assert resumed._state.messages[0].content == "Hello"
        assert resumed._state.messages[1].content == "Response"

    @pytest.mark.asyncio
    async def test_spawn_worker_session_reuses_initialized_runtime(self, mock_llm, mock_registry, tmp_path):
        """Should spawn worker sessions even after the parent session is initialized."""
        from src.agent.adapter import AgentSession
        from src.core.session import SessionManager

        session = AgentSession(
            model_client=mock_llm,
            registry=mock_registry,
            mode="build",
            project_dir=tmp_path,
            enable_trace=False,
        )
        await session.initialize()

        worker = await session.spawn_worker_session(enable_trace=False, worker_role="inspect")

        assert worker is not None
        assert worker.worker_role == "inspect"
        assert worker._agent is not None
        assert worker._session_manager is None
        assert worker._session_record is None

        session_manager = SessionManager(tmp_path / ".agent" / "sessions")
        listed = session_manager.list_sessions(project="default")
        assert len(listed) == 1
        assert listed[0].id == session.session_id
        await worker.aclose()

    @pytest.mark.asyncio
    async def test_agent_session_mode_change_transfers_model_ownership(self, tmp_path):
        """Mode changes should preserve model ownership so shutdown still closes the client."""
        from src.agent.adapter import AgentSession
        from src.runtime.bootstrap import ProjectContext, RuntimeBootstrap

        class DummyModel:
            def __init__(self):
                self.closed = 0

            async def close(self):
                self.closed += 1

        class DummyRegistry:
            def __init__(self):
                self._mcp_clients = []

            def get_tool_names(self):
                return ["read"]

        session = AgentSession(
            model_client=None,
            registry=None,
            mode="build",
            project_dir=tmp_path,
            enable_trace=False,
            persist_session=False,
        )
        model = DummyModel()
        first_runtime = RuntimeBootstrap(
            context=ProjectContext("default", "build", tmp_path, None, [], ["read"], []),
            model_client=model,
            registry=DummyRegistry(),
            hooks=object(),
            checkpoints=object(),
            skills=object(),
            task_manager=object(),
            owns_model_client=True,
            owns_registry=True,
        )
        second_runtime = RuntimeBootstrap(
            context=ProjectContext("default", "plan", tmp_path, None, [], ["read"], []),
            model_client=model,
            registry=DummyRegistry(),
            hooks=object(),
            checkpoints=object(),
            skills=object(),
            task_manager=object(),
            owns_model_client=False,
            owns_registry=True,
        )

        async def fake_bootstrap_runtime(*, registry=None):
            assert registry is None
            return second_runtime

        session._runtime = first_runtime
        session._apply_runtime(first_runtime)
        session._state = MagicMock()
        session._agent = object()
        session._bootstrap_runtime = fake_bootstrap_runtime
        session._rebuild_agent = MagicMock()

        await session.set_mode("plan")
        await session.aclose()

        assert second_runtime.owns_model_client is True
        assert model.closed == 1

    @pytest.mark.asyncio
    async def test_agent_session_reinitialize_after_reset_reuses_existing_runtime(self, tmp_path):
        """Reset sessions should reuse the current runtime instead of bootstrapping a replacement."""
        from src.agent.adapter import AgentSession
        from src.runtime.bootstrap import ProjectContext, RuntimeBootstrap

        session = AgentSession(
            model_client=None,
            registry=None,
            mode="build",
            project_dir=tmp_path,
            enable_trace=False,
            persist_session=False,
        )
        runtime = RuntimeBootstrap(
            context=ProjectContext("default", "build", tmp_path, None, [], ["read"], []),
            model_client=object(),
            registry=MagicMock(),
            hooks=object(),
            checkpoints=object(),
            skills=object(),
            task_manager=object(),
            owns_model_client=True,
            owns_registry=True,
        )

        def read(path: str) -> str:
            return path

        runtime.registry.get_tool_names.return_value = ["read"]
        runtime.registry._tool_schemas = {
            "read": {"name": "read", "description": "Read", "parameters": {}}
        }
        runtime.registry._tools = {}

        session._runtime = runtime
        session._apply_runtime(runtime)
        session._trace = None

        async def fail_bootstrap_runtime(*, registry=None):
            raise AssertionError("bootstrap_runtime should not be called after reset")

        session._bootstrap_runtime = fail_bootstrap_runtime

        session.reset()
        await session.initialize()

        assert session._runtime is runtime
        assert session._agent is not None

    def test_agent_session_close_uses_aclose_for_sync_callers(self, tmp_path):
        """Synchronous close should delegate to aclose so resources are released."""
        from src.agent.adapter import AgentSession

        session = AgentSession(
            model_client=None,
            registry=None,
            mode="build",
            project_dir=tmp_path,
            enable_trace=False,
            persist_session=False,
        )
        called = {"aclose": 0}

        async def fake_aclose():
            called["aclose"] += 1
            return None

        session.aclose = fake_aclose
        session.close()

        assert called["aclose"] == 1

    @pytest.mark.asyncio
    async def test_agent_session_reinitialize_after_close_drops_owned_resources(self, tmp_path):
        """Closed sessions should not reuse owned model/registry instances on reinitialize."""
        from src.agent.adapter import AgentSession
        from src.runtime.bootstrap import ProjectContext, RuntimeBootstrap

        class DummyModel:
            def __init__(self):
                self.closed = 0

            async def close(self):
                self.closed += 1

        class DummyRegistry:
            def __init__(self):
                self._mcp_clients = []

            def get_tool_names(self):
                return ["read"]

        session = AgentSession(
            model_client=None,
            registry=None,
            mode="build",
            project_dir=tmp_path,
            enable_trace=False,
            persist_session=False,
        )
        first_runtime = RuntimeBootstrap(
            context=ProjectContext("default", "build", tmp_path, None, [], ["read"], []),
            model_client=DummyModel(),
            registry=DummyRegistry(),
            hooks=object(),
            checkpoints=object(),
            skills=object(),
            task_manager=object(),
            owns_model_client=True,
            owns_registry=True,
        )
        second_runtime = RuntimeBootstrap(
            context=ProjectContext("default", "build", tmp_path, None, [], ["read"], []),
            model_client=DummyModel(),
            registry=DummyRegistry(),
            hooks=object(),
            checkpoints=object(),
            skills=object(),
            task_manager=object(),
            owns_model_client=True,
            owns_registry=True,
        )

        bootstrap_calls: list[tuple[object | None, object | None]] = []

        async def fake_bootstrap_runtime(*, registry=None):
            bootstrap_calls.append((registry, session.model_client))
            return first_runtime if len(bootstrap_calls) == 1 else second_runtime

        session._bootstrap_runtime = fake_bootstrap_runtime
        session._initialize_trace = MagicMock()
        session._rebuild_agent = lambda: setattr(session, "_agent", object())

        await session.initialize()
        await session.aclose()
        await session.initialize()

        assert bootstrap_calls[0] == (None, None)
        assert bootstrap_calls[1] == (None, None)
        assert first_runtime.model_client.closed == 1
        assert session._runtime is second_runtime

    @pytest.mark.asyncio
    async def test_agent_session_initialize_failure_does_not_persist_empty_session(
        self, mock_llm, tmp_path
    ):
        """Failed runtime initialization should not create an empty persisted session."""
        from src.agent.adapter import AgentSession
        from src.core.session import SessionManager

        session = AgentSession(
            model_client=mock_llm,
            registry=None,
            mode="build",
            project_dir=tmp_path,
            enable_trace=False,
        )

        async def fail_bootstrap_runtime(*, registry=None):
            raise RuntimeError("bootstrap failed")

        session._bootstrap_runtime = fail_bootstrap_runtime

        with pytest.raises(RuntimeError, match="bootstrap failed"):
            await session.initialize()

        session_manager = SessionManager(tmp_path / ".agent" / "sessions")
        assert session_manager.list_sessions(project="default") == []

    @pytest.mark.asyncio
    async def test_agent_session_reset_clears_persisted_name_for_next_topic(
        self, mock_llm, mock_registry, tmp_path
    ):
        """Reset sessions should derive a fresh title from the next conversation."""
        from src.agent.adapter import AgentSession
        from src.core.session import SessionManager

        session = AgentSession(
            model_client=mock_llm,
            registry=mock_registry,
            mode="build",
            project_dir=tmp_path,
            enable_trace=False,
        )

        await session.initialize()
        await session.turn("First topic")
        session_id = session.session_id

        session.reset()
        await session.initialize()
        await session.turn("Second topic")
        await session.aclose()

        session_manager = SessionManager(tmp_path / ".agent" / "sessions")
        persisted = session_manager.load_session(session_id)
        assert persisted is not None
        assert persisted.metadata.name == "Second topic"
