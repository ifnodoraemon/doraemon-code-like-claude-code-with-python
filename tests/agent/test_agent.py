"""
Agent Module Tests

Tests for the standard agent abstraction.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent import (
    Action,
    ActionType,
    AgentResult,
    AgentState,
    BaseAgent,
    DoraemonAgent,
    MaxTurnsExceededError,
    Message,
    Observation,
    ReActAgent,
    Thought,
    ToolCall,
    ToolDefinition,
    create_doraemon_agent,
)


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
        state.add_tool_result("call_123", "read_file", "file content")

        assert len(state.messages) == 1
        assert state.messages[0].role == "tool"
        assert state.messages[0].name == "read_file"

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


class TestToolDefinition:
    """Tests for ToolDefinition."""

    def test_to_api_format(self):
        """Should convert to API format."""
        tool = ToolDefinition(
            name="read_file",
            description="Read a file",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        )

        api_format = tool.to_api_format()

        assert api_format["type"] == "function"
        assert api_format["function"]["name"] == "read_file"
        assert api_format["function"]["description"] == "Read a file"

    def test_sensitive_tool(self):
        """Should mark sensitive tools."""
        tool = ToolDefinition(
            name="delete_file",
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

    def test_reset(self, agent):
        """Should reset agent state."""
        agent.state.add_user_message("Test")
        agent.state.turn_count = 5

        agent.reset()

        assert len(agent.state.messages) == 0
        assert agent.state.turn_count == 0
        assert agent.state.is_finished is False


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
    async def test_agent_session(self, mock_llm, mock_registry):
        """Should manage agent session."""
        from src.agent.adapter import AgentSession

        session = AgentSession(
            model_client=mock_llm,
            registry=mock_registry,
            mode="build",
        )

        await session.initialize()

        assert session._agent is not None
        assert session._state is not None

        result = await session.turn("Hello")

        assert result.success is True

        session.reset()

        assert session._state is None

    @pytest.mark.asyncio
    async def test_agent_session_mode_change(self, mock_llm, mock_registry):
        """Should change mode during session."""
        from src.agent.adapter import AgentSession

        session = AgentSession(
            model_client=mock_llm,
            registry=mock_registry,
            mode="build",
        )

        await session.initialize()

        session.set_mode("plan")

        assert session.mode == "plan"
        assert session._state.mode == "plan"
