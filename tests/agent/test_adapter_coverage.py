"""Targeted coverage tests for agent.adapter."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.adapter import (
    AgentSession,
    AgentTurnResult,
    _collect_modified_paths,
    _message_from_session_data,
    _message_to_session_data,
    run_agent_turn,
)
from src.agent.types import Message


class TestCollectModifiedPaths:
    def test_write_with_move(self):
        tool_calls = [SimpleNamespace(name="write", arguments={"operation": "move", "path": "/src", "destination": "/dst"})]
        result = _collect_modified_paths(tool_calls)
        assert "/src" in result
        assert "/dst" in result

    def test_write_with_copy(self):
        tool_calls = [SimpleNamespace(name="write", arguments={"operation": "copy", "path": "/a", "destination": "/b"})]
        result = _collect_modified_paths(tool_calls)
        assert "/a" in result
        assert "/b" in result

    def test_non_write_tool_ignored(self):
        tool_calls = [SimpleNamespace(name="read", arguments={"path": "/x"})]
        result = _collect_modified_paths(tool_calls)
        assert result == []

    def test_no_destination_in_move(self):
        tool_calls = [SimpleNamespace(name="write", arguments={"operation": "move", "path": "/x"})]
        result = _collect_modified_paths(tool_calls)
        assert result == ["/x"]


class TestMessageSessionDataRoundTrip:
    def test_roundtrip_full(self):
        msg = Message(role="assistant", content="hi", tool_calls=[{"id": "1"}], tool_call_id="1", name="fn", thought="think")
        data = _message_to_session_data(msg)
        restored = _message_from_session_data(data)
        assert restored.role == "assistant"
        assert restored.content == "hi"
        assert restored.tool_calls == [{"id": "1"}]
        assert restored.tool_call_id == "1"
        assert restored.name == "fn"
        assert restored.thought == "think"

    def test_roundtrip_minimal(self):
        msg = Message(role="user", content="hello")
        data = _message_to_session_data(msg)
        assert "tool_calls" not in data
        restored = _message_from_session_data(data)
        assert restored.content == "hello"
        assert restored.tool_calls is None


class TestAgentSessionProperties:
    def test_session_id_generates_if_none(self):
        session = AgentSession(model_client=None, registry=None)
        sid = session.session_id
        assert sid is not None
        assert len(sid) > 0

    def test_session_id_preserves_if_set(self):
        session = AgentSession(model_client=None, registry=None, session_id="fixed")
        assert session.session_id == "fixed"

    def test_get_state_none(self):
        session = AgentSession(model_client=None, registry=None)
        assert session.get_state() is None

    def test_get_trace_none(self):
        session = AgentSession(model_client=None, registry=None)
        assert session.get_trace() is None

    def test_get_orchestration_state_no_record(self):
        session = AgentSession(model_client=None, registry=None)
        assert session.get_orchestration_state() == {}

    def test_get_orchestration_runs_no_record(self):
        session = AgentSession(model_client=None, registry=None)
        assert session.get_orchestration_runs() == []

    def test_get_active_orchestration_run_id_no_record(self):
        session = AgentSession(model_client=None, registry=None)
        assert session.get_active_orchestration_run_id() is None

    def test_get_task_manager_returns_attribute(self):
        tm = MagicMock()
        session = AgentSession(model_client=None, registry=None, task_manager=tm)
        assert session.get_task_manager() is tm


class TestAgentSessionOrchestration:
    @pytest.mark.asyncio
    async def test_orchestrate_unknown_resume_run_raises(self):
        session = AgentSession(model_client=None, registry=None)
        session._agent = MagicMock()
        with pytest.raises(ValueError, match="unknown orchestration run"):
            await session.orchestrate("", resume_run_id="nonexistent")


class TestAgentSessionSpawnWorker:
    @pytest.mark.asyncio
    async def test_spawn_requires_agent(self):
        session = AgentSession(model_client=None, registry=None)
        with patch.object(session, "initialize", new_callable=AsyncMock):
            with patch.object(session, "_prepare_session_record"):
                with patch.object(session, "_initialize_state"):
                    with patch.object(session, "_ensure_runtime", new_callable=AsyncMock):
                        with patch.object(session, "_ensure_session_record"):
                            with patch.object(session, "_initialize_trace"):
                                with patch.object(session, "_rebuild_agent"):
                                    session._agent = None
                                    worker = await session.spawn_worker_session(worker_role="inspect")
                                    assert worker is not None
                                    assert worker.worker_role == "inspect"
                                    assert worker.persist_session is False


class TestAgentSessionClose:
    @pytest.mark.asyncio
    async def test_close_with_no_runtime(self):
        session = AgentSession(model_client=None, registry=None)
        result = await session.aclose()
        assert result is None

    @pytest.mark.asyncio
    async def test_close_with_runtime(self):
        session = AgentSession(model_client=None, registry=None)
        mock_runtime = MagicMock()
        mock_runtime.owns_model_client = True
        mock_runtime.owns_registry = True
        mock_runtime.aclose = AsyncMock()
        session._runtime = mock_runtime
        session._trace = None
        result = await session.aclose()
        mock_runtime.aclose.assert_called_once()
        assert session.model_client is None
        assert session.registry is None


class TestAgentSessionReset:
    def test_reset_clears_state(self):
        session = AgentSession(model_client=None, registry=None)
        session._agent = MagicMock()
        session.reset()
        assert session._agent is None
        assert session._state is None

    def test_reset_with_session_record(self):
        session = AgentSession(model_client=None, registry=None)
        session._agent = MagicMock()
        record = MagicMock()
        record.messages = [1, 2]
        record.orchestration_state = {"k": "v"}
        record.orchestration_runs = [{"r": 1}]
        record.active_orchestration_run_id = "run-1"
        record.metadata = MagicMock()
        record.metadata.message_count = 2
        record.metadata.total_tokens = 100
        record.metadata.name = "test"
        sm = MagicMock()
        session._session_manager = sm
        session._session_record = record
        session.reset()
        assert record.messages == []
        sm.save_session.assert_called_once_with(record)


class TestAgentSessionSetMode:
    @pytest.mark.asyncio
    async def test_set_mode_same_noop(self):
        session = AgentSession(model_client=None, registry=None, mode="build")
        await session.set_mode("build")
        assert session.mode == "build"

    @pytest.mark.asyncio
    async def test_set_mode_different_updates(self):
        session = AgentSession(model_client=None, registry=None, mode="build")
        session._runtime = None
        session._state = MagicMock()
        session._state.mode = "build"
        session._session_record = None
        session._session_manager = None
        await session.set_mode("plan")
        assert session.mode == "plan"
        assert session._state.mode == "plan"


class TestDeriveSessionName:
    def test_with_messages(self):
        session = AgentSession(model_client=None, registry=None)
        session._state = MagicMock()
        msg = SimpleNamespace(role="user", content="  hello   world  ")
        session._state.messages = [msg]
        result = session._derive_session_name()
        assert "hello world" in result

    def test_no_messages(self):
        session = AgentSession(model_client=None, registry=None)
        session._state = MagicMock()
        session._state.messages = []
        result = session._derive_session_name()
        assert result is None

    def test_long_message_truncated(self):
        session = AgentSession(model_client=None, registry=None)
        session._state = MagicMock()
        msg = SimpleNamespace(role="user", content="x" * 100)
        session._state.messages = [msg]
        result = session._derive_session_name()
        assert len(result) <= 63


class TestRunAgentTurn:
    @pytest.mark.asyncio
    async def test_run_agent_turn_timeout(self):
        with patch("src.agent.adapter.create_doraemon_agent") as mock_create:
            agent = MagicMock()
            agent.run = AsyncMock(side_effect=asyncio.TimeoutError())
            agent.state = MagicMock()
            mock_create.return_value = agent
            result = await run_agent_turn("hi", MagicMock(), MagicMock(), MagicMock(), display_output=False)
            assert result.success is False
            assert "timed out" in result.error

    @pytest.mark.asyncio
    async def test_run_agent_turn_generic_error(self):
        with patch("src.agent.adapter.create_doraemon_agent") as mock_create:
            agent = MagicMock()
            agent.run = AsyncMock(side_effect=RuntimeError("boom"))
            agent.state = MagicMock()
            mock_create.return_value = agent
            result = await run_agent_turn("hi", MagicMock(), MagicMock(), MagicMock(), display_output=False)
            assert result.success is False
            assert "boom" in result.error
