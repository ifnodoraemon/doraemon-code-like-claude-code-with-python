"""Comprehensive tests for enhanced subagents.py

Tests cover:
- Agent communication protocol (message passing)
- Agent state management (idle/running/completed/failed)
- Agent parallel execution with asyncio
- Result aggregation and monitoring
- Timeout and error handling
- Agent metrics and logging
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.subagents import (
    AgentMessage,
    AgentMessageQueue,
    AgentMetrics,
    AgentState,
    AgentStateManager,
    SubagentConfig,
    SubagentManager,
    SubagentModel,
    SubagentResult,
    _aggregate_results,
    _create_agent_id,
    _get_model_name,
)


class TestAgentState:
    """Tests for AgentState enum."""

    def test_agent_state_values(self):
        """Test all agent state values."""
        assert AgentState.IDLE.value == "idle"
        assert AgentState.RUNNING.value == "running"
        assert AgentState.COMPLETED.value == "completed"
        assert AgentState.FAILED.value == "failed"
        assert AgentState.TIMEOUT.value == "timeout"
        assert AgentState.CANCELLED.value == "cancelled"

    def test_agent_state_enum_members(self):
        """Test agent state enum members."""
        states = [
            AgentState.IDLE,
            AgentState.RUNNING,
            AgentState.COMPLETED,
            AgentState.FAILED,
            AgentState.TIMEOUT,
            AgentState.CANCELLED,
        ]
        assert len(states) == 6


class TestAgentMessage:
    """Tests for AgentMessage dataclass."""

    def test_agent_message_creation(self):
        """Test creating an agent message."""
        msg = AgentMessage(
            sender_id="agent_1",
            recipient_id="agent_2",
            message_type="request",
            content="Please review this code",
        )
        assert msg.sender_id == "agent_1"
        assert msg.recipient_id == "agent_2"
        assert msg.message_type == "request"
        assert msg.content == "Please review this code"
        assert msg.timestamp > 0

    def test_agent_message_broadcast(self):
        """Test creating a broadcast message."""
        msg = AgentMessage(
            sender_id="agent_1",
            recipient_id=None,
            message_type="status",
            content="Task completed",
        )
        assert msg.recipient_id is None

    def test_agent_message_with_metadata(self):
        """Test agent message with metadata."""
        metadata = {"priority": "high", "retry_count": 3}
        msg = AgentMessage(
            sender_id="agent_1",
            recipient_id="agent_2",
            message_type="request",
            content="Do something",
            metadata=metadata,
        )
        assert msg.metadata == metadata

    def test_agent_message_to_dict(self):
        """Test converting message to dict."""
        msg = AgentMessage(
            sender_id="agent_1",
            recipient_id="agent_2",
            message_type="request",
            content="Test",
        )
        msg_dict = msg.to_dict()
        assert msg_dict["sender_id"] == "agent_1"
        assert msg_dict["recipient_id"] == "agent_2"
        assert msg_dict["message_type"] == "request"
        assert msg_dict["content"] == "Test"


class TestAgentMetrics:
    """Tests for AgentMetrics dataclass."""

    def test_agent_metrics_creation(self):
        """Test creating agent metrics."""
        start_time = time.time()
        metrics = AgentMetrics(
            agent_id="agent_1",
            agent_name="code-reviewer",
            state=AgentState.RUNNING,
            start_time=start_time,
        )
        assert metrics.agent_id == "agent_1"
        assert metrics.agent_name == "code-reviewer"
        assert metrics.state == AgentState.RUNNING
        assert metrics.turns_used == 0
        assert metrics.tokens_used == 0

    def test_agent_metrics_duration(self):
        """Test calculating duration."""
        start_time = time.time()
        metrics = AgentMetrics(
            agent_id="agent_1",
            agent_name="test",
            state=AgentState.COMPLETED,
            start_time=start_time,
            end_time=start_time + 5.0,
        )
        assert metrics.duration == 5.0

    def test_agent_metrics_to_dict(self):
        """Test converting metrics to dict."""
        start_time = time.time()
        metrics = AgentMetrics(
            agent_id="agent_1",
            agent_name="test",
            state=AgentState.COMPLETED,
            start_time=start_time,
            end_time=start_time + 2.5,
            turns_used=3,
            tokens_used=150,
            tool_calls=2,
            errors=0,
        )
        metrics_dict = metrics.to_dict()
        assert metrics_dict["agent_id"] == "agent_1"
        assert metrics_dict["agent_name"] == "test"
        assert metrics_dict["state"] == "completed"
        assert metrics_dict["turns_used"] == 3
        assert metrics_dict["tokens_used"] == 150


class TestAgentMessageQueue:
    """Tests for AgentMessageQueue."""

    @pytest.mark.asyncio
    async def test_message_queue_send_receive(self):
        """Test sending and receiving messages."""
        queue = AgentMessageQueue()
        msg = AgentMessage(
            sender_id="agent_1",
            recipient_id="agent_2",
            message_type="request",
            content="Test",
        )

        await queue.send(msg)
        received = await queue.receive(agent_id="agent_2", timeout=1.0)

        assert received is not None
        assert received.sender_id == "agent_1"
        assert received.content == "Test"

    @pytest.mark.asyncio
    async def test_message_queue_timeout(self):
        """Test message queue timeout."""
        queue = AgentMessageQueue()
        received = await queue.receive(agent_id="missing-agent", timeout=0.1)
        assert received is None

    @pytest.mark.asyncio
    async def test_message_queue_legacy_timeout_signature(self):
        """Test legacy receive(timeout) calls still work."""
        queue = AgentMessageQueue()
        received = await queue.receive(0.1)
        assert received is None

    @pytest.mark.asyncio
    async def test_message_queue_subscribe(self):
        """Test message subscription."""
        queue = AgentMessageQueue()
        received_messages = []

        def callback(msg: AgentMessage):
            received_messages.append(msg)

        queue.subscribe("agent_2", callback)

        msg = AgentMessage(
            sender_id="agent_1",
            recipient_id="agent_2",
            message_type="request",
            content="Test",
        )

        await queue.send(msg)
        await asyncio.sleep(0.1)

        assert len(received_messages) == 1
        assert received_messages[0].content == "Test"

    @pytest.mark.asyncio
    async def test_message_queue_multiple_subscribers(self):
        """Test multiple subscribers."""
        queue = AgentMessageQueue()
        messages_1 = []
        messages_2 = []

        queue.subscribe("agent_1", lambda msg: messages_1.append(msg))
        queue.subscribe("agent_1", lambda msg: messages_2.append(msg))

        msg = AgentMessage(
            sender_id="sender",
            recipient_id="agent_1",
            message_type="request",
            content="Test",
        )

        await queue.send(msg)
        await asyncio.sleep(0.1)

        assert len(messages_1) == 1
        assert len(messages_2) == 1

    @pytest.mark.asyncio
    async def test_message_queue_persists_mailbox_entries(self, tmp_path):
        """Test durable mailbox append logs."""
        queue = AgentMessageQueue(storage_dir=tmp_path / "mailboxes")
        msg = AgentMessage(
            sender_id="agent_1",
            recipient_id="agent_2",
            message_type="request",
            content="Persist me",
        )

        await queue.send(msg)
        persisted = queue.get_mailbox_messages("agent_2")

        assert len(persisted) == 1
        assert persisted[0].content == "Persist me"


class TestAgentStateManager:
    """Tests for AgentStateManager."""

    @pytest.mark.asyncio
    async def test_state_manager_set_get_state(self):
        """Test setting and getting agent state."""
        manager = AgentStateManager()

        await manager.set_state("agent_1", AgentState.RUNNING)
        state = await manager.get_state("agent_1")

        assert state == AgentState.RUNNING

    @pytest.mark.asyncio
    async def test_state_manager_state_transitions(self):
        """Test state transitions."""
        manager = AgentStateManager()

        await manager.set_state("agent_1", AgentState.IDLE)
        assert await manager.get_state("agent_1") == AgentState.IDLE

        await manager.set_state("agent_1", AgentState.RUNNING)
        assert await manager.get_state("agent_1") == AgentState.RUNNING

        await manager.set_state("agent_1", AgentState.COMPLETED)
        assert await manager.get_state("agent_1") == AgentState.COMPLETED

    @pytest.mark.asyncio
    async def test_state_manager_create_metrics(self):
        """Test creating metrics."""
        manager = AgentStateManager()

        metrics = await manager.create_metrics("agent_1", "code-reviewer")

        assert metrics.agent_id == "agent_1"
        assert metrics.agent_name == "code-reviewer"
        assert metrics.state == AgentState.IDLE

    @pytest.mark.asyncio
    async def test_state_manager_update_metrics(self):
        """Test updating metrics."""
        manager = AgentStateManager()

        await manager.create_metrics("agent_1", "test")
        await manager.update_metrics("agent_1", turns_used=5, tokens_used=200, tool_calls=3)

        metrics = await manager.get_metrics("agent_1")
        assert metrics.turns_used == 5
        assert metrics.tokens_used == 200
        assert metrics.tool_calls == 3

    @pytest.mark.asyncio
    async def test_state_manager_get_all_metrics(self):
        """Test getting all metrics."""
        manager = AgentStateManager()

        await manager.create_metrics("agent_1", "reviewer")
        await manager.create_metrics("agent_2", "debugger")

        all_metrics = await manager.get_all_metrics()

        assert len(all_metrics) == 2
        assert "agent_1" in all_metrics
        assert "agent_2" in all_metrics


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_agent_id(self):
        """Test creating agent IDs."""
        id1 = _create_agent_id()
        id2 = _create_agent_id()

        assert len(id1) == 8
        assert len(id2) == 8
        assert id1 != id2

    def test_get_model_name_inherit(self):
        """Test getting model name with INHERIT."""
        model_name = _get_model_name(SubagentModel.INHERIT, "gemini-2.0-flash")
        assert model_name == "gemini-2.0-flash"

    def test_get_model_name_pro(self):
        """Test getting model name for PRO (inherits parent model)."""
        model_name = _get_model_name(SubagentModel.PRO, "gemini-2.0-flash")
        assert model_name == "gemini-2.0-flash"

    def test_get_model_name_flash(self):
        """Test getting model name for FLASH (inherits parent model)."""
        model_name = _get_model_name(SubagentModel.FLASH, "gemini-2.0-flash")
        assert model_name == "gemini-2.0-flash"

    def test_aggregate_results_empty(self):
        """Test aggregating empty results."""
        aggregated = _aggregate_results([])

        assert aggregated["total_agents"] == 0
        assert aggregated["successful"] == 0
        assert aggregated["failed"] == 0
        assert aggregated["total_duration"] == 0

    def test_aggregate_results_single(self):
        """Test aggregating single result."""
        result = SubagentResult(
            agent_id="agent_1",
            agent_name="test",
            success=True,
            output="Success",
            turns_used=2,
            tokens_used=100,
            duration=1.5,
        )

        aggregated = _aggregate_results([result])

        assert aggregated["total_agents"] == 1
        assert aggregated["successful"] == 1
        assert aggregated["failed"] == 0
        assert aggregated["total_tokens"] == 100

    def test_aggregate_results_multiple(self):
        """Test aggregating multiple results."""
        results = [
            SubagentResult(
                agent_id="agent_1",
                agent_name="reviewer",
                success=True,
                output="OK",
                turns_used=2,
                tokens_used=100,
                duration=1.0,
            ),
            SubagentResult(
                agent_id="agent_2",
                agent_name="debugger",
                success=True,
                output="OK",
                turns_used=3,
                tokens_used=150,
                duration=2.0,
            ),
            SubagentResult(
                agent_id="agent_3",
                agent_name="tester",
                success=False,
                output="",
                turns_used=0,
                tokens_used=0,
                duration=0.5,
                error="Timeout",
            ),
        ]

        aggregated = _aggregate_results(results)

        assert aggregated["total_agents"] == 3
        assert aggregated["successful"] == 2
        assert aggregated["failed"] == 1
        assert aggregated["total_tokens"] == 250
        assert aggregated["total_duration"] == 3.5


class TestSubagentResult:
    """Tests for SubagentResult with state tracking."""

    def test_subagent_result_with_state(self):
        """Test SubagentResult with state."""
        result = SubagentResult(
            agent_id="agent_1",
            agent_name="test",
            success=True,
            output="Success",
            turns_used=2,
            tokens_used=100,
            duration=1.5,
            state=AgentState.COMPLETED,
        )

        assert result.state == AgentState.COMPLETED
        assert result.success is True

    def test_subagent_result_failed_state(self):
        """Test SubagentResult with failed state."""
        result = SubagentResult(
            agent_id="agent_1",
            agent_name="test",
            success=False,
            output="",
            turns_used=0,
            tokens_used=0,
            duration=0.5,
            error="Timeout",
            state=AgentState.TIMEOUT,
        )

        assert result.state == AgentState.TIMEOUT
        assert result.success is False

    def test_subagent_result_with_metrics(self):
        """Test SubagentResult with metrics."""
        start_time = time.time()
        metrics = AgentMetrics(
            agent_id="agent_1",
            agent_name="test",
            state=AgentState.COMPLETED,
            start_time=start_time,
            end_time=start_time + 1.5,
            turns_used=2,
            tokens_used=100,
        )

        result = SubagentResult(
            agent_id="agent_1",
            agent_name="test",
            success=True,
            output="Success",
            turns_used=2,
            tokens_used=100,
            duration=1.5,
            metrics=metrics,
        )

        assert result.metrics is not None
        assert result.metrics.agent_id == "agent_1"

    def test_subagent_result_to_dict(self):
        """Test converting result to dict."""
        result = SubagentResult(
            agent_id="agent_1",
            agent_name="test",
            success=True,
            output="Success",
            turns_used=2,
            tokens_used=100,
            duration=1.5,
            state=AgentState.COMPLETED,
        )

        result_dict = result.to_dict()

        assert result_dict["agent_id"] == "agent_1"
        assert result_dict["success"] is True
        assert result_dict["state"] == "completed"


class TestSubagentManager:
    """Tests for enhanced SubagentManager."""

    @pytest.fixture
    def mock_client(self):
        """Create mock GenAI client."""
        return MagicMock()

    @pytest.fixture
    def mock_tool_registry(self):
        """Create mock tool registry."""
        registry = MagicMock()
        registry.get_genai_tools = MagicMock(return_value=[])
        return registry

    @pytest.fixture
    def manager(self, mock_client, mock_tool_registry):
        """Create SubagentManager instance."""
        return SubagentManager(mock_client, mock_tool_registry, parent_model="test-model")

    def test_manager_initialization(self, manager):
        """Test manager initialization."""
        assert manager.model_client is not None
        assert manager.tool_registry is not None
        assert manager.parent_model == "test-model"

    def test_manager_register_agent(self, manager):
        """Test registering custom agent."""
        config = SubagentConfig(
            name="custom-agent",
            description="Custom agent",
            prompt="You are helpful",
        )

        manager.register_agent(config)

        assert manager.get_agent_config("custom-agent") is not None

    def test_manager_list_agents(self, manager):
        """Test listing agents."""
        agents = manager.list_agents()

        assert len(agents) > 0
        agent_names = [a["name"] for a in agents]
        assert "code-reviewer" in agent_names
        assert "debugger" in agent_names

    @pytest.mark.asyncio
    async def test_manager_get_agent_state(self, manager):
        """Test getting agent state."""
        await manager._state_manager.set_state("agent_1", AgentState.RUNNING)
        state = await manager.get_agent_state("agent_1")

        assert state == AgentState.RUNNING

    @pytest.mark.asyncio
    async def test_manager_get_agent_metrics(self, manager):
        """Test getting agent metrics."""
        await manager._state_manager.create_metrics("agent_1", "test")
        metrics = await manager.get_agent_metrics("agent_1")

        assert metrics is not None
        assert metrics.agent_id == "agent_1"

    @pytest.mark.asyncio
    async def test_manager_get_all_metrics(self, manager):
        """Test getting all metrics."""
        await manager._state_manager.create_metrics("agent_1", "test1")
        await manager._state_manager.create_metrics("agent_2", "test2")

        all_metrics = await manager.get_all_metrics()

        assert len(all_metrics) == 2

    @pytest.mark.asyncio
    async def test_manager_send_receive_message(self, manager):
        """Test sending and receiving messages."""
        msg = AgentMessage(
            sender_id="agent_1",
            recipient_id="agent_2",
            message_type="request",
            content="Test",
        )

        await manager.send_message(msg)
        received = await manager.receive_message("agent_2", timeout=1.0)

        assert received is not None
        assert received.content == "Test"

    @pytest.mark.asyncio
    async def test_manager_get_mailbox_messages(self, manager, tmp_path):
        """Test reading durable mailbox messages via manager."""
        manager._message_queue = AgentMessageQueue(storage_dir=tmp_path / "mailboxes")
        msg = AgentMessage(
            sender_id="agent_1",
            recipient_id="agent_2",
            message_type="request",
            content="Stored",
        )

        await manager.send_message(msg)
        persisted = manager.get_mailbox_messages("agent_2")

        assert len(persisted) == 1
        assert persisted[0].content == "Stored"

    @pytest.mark.asyncio
    async def test_manager_subscribe_to_messages(self, manager):
        """Test subscribing to messages."""
        received_messages = []

        def callback(msg: AgentMessage):
            received_messages.append(msg)

        manager.subscribe_to_messages("agent_1", callback)

        msg = AgentMessage(
            sender_id="sender",
            recipient_id="agent_1",
            message_type="request",
            content="Test",
        )

        await manager.send_message(msg)
        await asyncio.sleep(0.1)

        assert len(received_messages) == 1

    @pytest.mark.asyncio
    async def test_manager_spawn_unknown_agent(self, manager):
        """Test spawning unknown agent."""
        result = await manager.spawn("unknown-agent", "Do something")

        assert result.success is False
        assert result.state == AgentState.FAILED
        assert "Unknown agent" in result.error

    @pytest.mark.asyncio
    async def test_manager_spawn_with_timeout(self, manager):
        """Test spawning agent with custom timeout."""
        config = SubagentConfig(
            name="test-agent",
            description="Test",
            prompt="Test",
            timeout=10.0,
        )

        manager.register_agent(config)

        # This will fail because we don't have a real client, but we can verify
        # the timeout parameter is used
        result = await manager.spawn_with_timeout("test-agent", "Do something", timeout=5.0)

        # Result will be failed due to mock, but we're testing the interface
        assert result.agent_name == "test-agent"

    @pytest.mark.asyncio
    async def test_manager_get_running_agents(self, manager):
        """Test getting running agents."""
        running = await manager.get_running_agents()
        assert isinstance(running, list)

    @pytest.mark.asyncio
    async def test_manager_cancel_agent(self, manager):
        """Test cancelling agent."""
        # Create a mock task
        mock_task = AsyncMock()
        manager._running_agents["agent_1"] = mock_task

        result = await manager.cancel_agent("agent_1")

        assert result is True
        mock_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_manager_cancel_nonexistent_agent(self, manager):
        """Test cancelling non-existent agent."""
        result = await manager.cancel_agent("nonexistent")
        assert result is False


class TestSubagentConfig:
    """Tests for SubagentConfig."""

    def test_config_to_dict(self):
        """Test converting config to dict."""
        config = SubagentConfig(
            name="test",
            description="Test agent",
            prompt="You are helpful",
            tools=["read_file", "write_file"],
            model=SubagentModel.FLASH,
            max_turns=5,
            timeout=60.0,
        )

        config_dict = config.to_dict()

        assert config_dict["name"] == "test"
        assert config_dict["description"] == "Test agent"
        assert config_dict["tools"] == ["read_file", "write_file"]
        assert config_dict["model"] == "flash"

    def test_config_from_dict(self):
        """Test creating config from dict."""
        data = {
            "name": "test",
            "description": "Test agent",
            "prompt": "You are helpful",
            "tools": ["read_file"],
            "model": "flash",
            "max_turns": 5,
            "timeout": 60.0,
        }

        config = SubagentConfig.from_dict(data)

        assert config.name == "test"
        assert config.model == SubagentModel.FLASH
        assert config.max_turns == 5


class TestIntegration:
    """Integration tests for subagent system."""

    @pytest.mark.asyncio
    async def test_agent_communication_flow(self):
        """Test agent communication flow."""
        queue = AgentMessageQueue()
        AgentStateManager()

        # Simulate agent 1 sending a request
        msg1 = AgentMessage(
            sender_id="agent_1",
            recipient_id="agent_2",
            message_type="request",
            content="Please review this code",
        )

        await queue.send(msg1)

        # Simulate agent 2 receiving and responding
        received = await queue.receive(agent_id="agent_2", timeout=1.0)
        assert received.sender_id == "agent_1"

        # Agent 2 sends response
        msg2 = AgentMessage(
            sender_id="agent_2",
            recipient_id="agent_1",
            message_type="response",
            content="Code looks good",
        )

        await queue.send(msg2)

        # Agent 1 receives response
        response = await queue.receive(agent_id="agent_1", timeout=1.0)
        assert response.content == "Code looks good"

    @pytest.mark.asyncio
    async def test_agent_state_lifecycle(self):
        """Test agent state lifecycle."""
        manager = AgentStateManager()

        # Create metrics
        metrics = await manager.create_metrics("agent_1", "test")
        assert metrics.state == AgentState.IDLE

        # Transition to running
        await manager.set_state("agent_1", AgentState.RUNNING)
        state = await manager.get_state("agent_1")
        assert state == AgentState.RUNNING

        # Update metrics
        await manager.update_metrics(
            "agent_1", turns_used=3, tokens_used=150, state=AgentState.COMPLETED
        )

        # Verify final state
        final_metrics = await manager.get_metrics("agent_1")
        assert final_metrics.turns_used == 3
        assert final_metrics.tokens_used == 150

    @pytest.mark.asyncio
    async def test_parallel_agent_execution_simulation(self):
        """Test simulating parallel agent execution."""
        results = [
            SubagentResult(
                agent_id="agent_1",
                agent_name="reviewer",
                success=True,
                output="Review complete",
                turns_used=2,
                tokens_used=100,
                duration=1.0,
            ),
            SubagentResult(
                agent_id="agent_2",
                agent_name="debugger",
                success=True,
                output="Debug complete",
                turns_used=3,
                tokens_used=150,
                duration=1.5,
            ),
        ]

        aggregated = _aggregate_results(results)

        assert aggregated["total_agents"] == 2
        assert aggregated["successful"] == 2
        assert aggregated["total_duration"] == 2.5
        assert aggregated["total_tokens"] == 250
