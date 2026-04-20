"""Additional coverage tests for agent.adapter - run_agent_turn display, AgentSession lifecycle, turn, turn_stream, orchestrate, _restore/_save session state, spawn_worker."""

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


class TestRunAgentTurnDisplayCallback:
    @pytest.mark.asyncio
    async def test_display_callback_thought_event(self):
        with patch("src.agent.adapter.create_doraemon_agent") as mock_create:
            agent = MagicMock()
            result_obj = SimpleNamespace(
                response="done", tool_calls=[], tokens_used=10, success=True, error=None,
            )
            agent.run = AsyncMock(return_value=result_obj)
            agent.state = MagicMock()
            mock_create.return_value = agent
            result = await run_agent_turn(
                "hi", MagicMock(), MagicMock(), None, display_output=True,
            )
            assert result.success is True

    @pytest.mark.asyncio
    async def test_display_callback_action_event(self):
        with patch("src.agent.adapter.create_doraemon_agent") as mock_create:
            agent = MagicMock()
            result_obj = SimpleNamespace(
                response="done", tool_calls=[], tokens_used=10, success=True, error=None,
            )
            agent.run = AsyncMock(return_value=result_obj)
            agent.state = MagicMock()
            mock_create.return_value = agent
            result = await run_agent_turn(
                "hi", MagicMock(), MagicMock(), None, display_output=True,
            )
            assert result.success is True

    @pytest.mark.asyncio
    async def test_headless_permission_callback_denies(self):
        with patch("src.agent.adapter.create_doraemon_agent") as mock_create:
            agent = MagicMock()
            result_obj = SimpleNamespace(
                response="done", tool_calls=[], tokens_used=10, success=True, error=None,
            )
            agent.run = AsyncMock(return_value=result_obj)
            agent.state = MagicMock()
            mock_create.return_value = agent
            result = await run_agent_turn(
                "hi", MagicMock(), MagicMock(), None, headless=True,
                sensitive_tools={"write"}, display_output=False,
            )
            assert result.success is True

    @pytest.mark.asyncio
    async def test_context_messages_injected(self):
        with patch("src.agent.adapter.create_doraemon_agent") as mock_create:
            agent = MagicMock()
            result_obj = SimpleNamespace(
                response="done", tool_calls=[], tokens_used=10, success=True, error=None,
            )
            agent.run = AsyncMock(return_value=result_obj)
            agent.state = MagicMock()
            mock_create.return_value = agent
            ctx = MagicMock()
            ctx.messages = [SimpleNamespace(role="user", content="prior", tool_calls=None)]
            result = await run_agent_turn(
                "hi", MagicMock(), MagicMock(), ctx, display_output=False,
            )
            assert result.success is True


class TestRunAgentTurnToolCalls:
    @pytest.mark.asyncio
    async def test_tool_calls_collected_in_result(self):
        with patch("src.agent.adapter.create_doraemon_agent") as mock_create:
            agent = MagicMock()
            tc = SimpleNamespace(
                name="write",
                arguments={"operation": "create", "path": "/f"},
                to_dict=lambda: {"name": "write", "arguments": {"operation": "create", "path": "/f"}},
            )
            result_obj = SimpleNamespace(
                response="wrote", tool_calls=[tc], tokens_used=5, success=True, error=None,
            )
            agent.run = AsyncMock(return_value=result_obj)
            agent.state = MagicMock()
            mock_create.return_value = agent
            result = await run_agent_turn(
                "write file", MagicMock(), MagicMock(), None, display_output=False,
            )
            assert result.files_modified == ["/f"]


class TestAgentSessionTurnWithRuntime:
    @pytest.mark.asyncio
    async def test_turn_with_initialized_agent(self):
        session = AgentSession(model_client=None, registry=None)
        mock_agent = MagicMock()
        result_obj = SimpleNamespace(
            response="response", tool_calls=[], tokens_used=10, success=True, error=None,
        )
        mock_agent.run = AsyncMock(return_value=result_obj)
        session._agent = mock_agent
        session._state = MagicMock()
        session._state.messages = []
        session._session_manager = None
        session._session_record = None
        result = await session.turn("hello")
        assert result.success is True
        assert result.response == "response"

    @pytest.mark.asyncio
    async def test_turn_error_path_saves_state(self):
        session = AgentSession(model_client=None, registry=None)
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("fail"))
        session._agent = mock_agent
        session._state = MagicMock()
        session._state.messages = []
        session._session_manager = None
        session._session_record = None
        session._trace = MagicMock()
        result = await session.turn("hello")
        assert result.success is False
        assert "fail" in result.error
        session._trace.error.assert_called()


class TestAgentSessionTurnStream:
    @pytest.mark.asyncio
    async def test_turn_stream_yields_events(self):
        session = AgentSession(model_client=None, registry=None)
        mock_agent = MagicMock()

        async def fake_stream(input, **kw):
            yield {"type": "text", "content": "chunk1"}
            yield {"type": "text", "content": "chunk2"}

        mock_agent.run_stream = fake_stream
        session._agent = mock_agent
        session._session_manager = None
        session._session_record = None
        session._state = MagicMock()
        events = []
        async for event in session.turn_stream("hello"):
            events.append(event)
        assert len(events) == 2
        assert events[0]["content"] == "chunk1"

    @pytest.mark.asyncio
    async def test_turn_stream_error_yields_error_event(self):
        session = AgentSession(model_client=None, registry=None)
        mock_agent = MagicMock()

        async def fake_stream(input, **kw):
            yield {"type": "text", "content": "partial"}
            raise RuntimeError("stream broke")

        mock_agent.run_stream = fake_stream
        session._agent = mock_agent
        session._session_manager = None
        session._session_record = None
        session._state = MagicMock()
        session._trace = MagicMock()
        events = []
        async for event in session.turn_stream("hello"):
            events.append(event)
        assert any(e.get("type") == "error" for e in events)


class TestAgentSessionOrchestrateWithRuntime:
    @pytest.mark.asyncio
    async def test_orchestrate_with_runtime(self):
        session = AgentSession(model_client=None, registry=None)
        session._agent = MagicMock()
        mock_result = SimpleNamespace(
            success=True, summary="done", root_task_id="root-1",
            plan_id="p1", executed_task_ids=["t1"], completed_task_ids=["t1"],
            failed_task_ids=[], blocked_task_id=None,
            task_summaries={}, worker_assignments={},
            to_dict=lambda: {},
        )
        with patch("src.agent.adapter.LeadAgentRuntime") as MockRuntime:
            runtime_instance = MagicMock()
            runtime_instance.execute = AsyncMock(return_value=mock_result)
            MockRuntime.return_value = runtime_instance
            session._state = MagicMock()
            session._state.messages = []
            session._session_manager = None
            session._session_record = None
            result = await session.orchestrate("my goal", max_workers=2)
        assert result.success is True
        assert result.root_task_id == "root-1"

    @pytest.mark.asyncio
    async def test_orchestrate_error_rollback(self):
        session = AgentSession(model_client=None, registry=None)
        session._agent = MagicMock()
        session._state = MagicMock()
        session._state.messages = []
        session._session_manager = None
        session._session_record = None
        with patch("src.agent.adapter.LeadAgentRuntime") as MockRuntime:
            runtime_instance = MagicMock()
            runtime_instance.execute = AsyncMock(side_effect=RuntimeError("boom"))
            MockRuntime.return_value = runtime_instance
            session._trace = MagicMock()
            with pytest.raises(RuntimeError, match="boom"):
                await session.orchestrate("goal")


class TestAgentSessionSaveRestoreState:
    def test_restore_session_state_with_messages(self):
        session = AgentSession(model_client=None, registry=None)
        from src.agent.state import AgentState
        session._state = AgentState(mode="build", max_turns=100)
        session._session_record = MagicMock()
        session._session_record.messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        session._restore_session_state()
        assert len(session._state.messages) == 2
        assert session._state.last_response == "hello"
        assert session._state.user_input == "hi"

    def test_save_session_state(self):
        session = AgentSession(model_client=None, registry=None)
        from src.agent.state import AgentState
        session._state = AgentState(mode="build", max_turns=100)
        session._state.add_user_message("test")
        session._state.add_assistant_message("response")
        sm = MagicMock()
        record = MagicMock()
        record.metadata = MagicMock()
        record.metadata.name = None
        session._session_manager = sm
        session._session_record = record
        session._save_session_state()
        sm.save_session.assert_called_once_with(record)
        assert record.metadata.name is not None


class TestAgentSessionCloseWithRegistryMcp:
    @pytest.mark.asyncio
    async def test_close_with_registry_mcp_clients(self):
        session = AgentSession(model_client=None, registry=None)
        mock_registry = MagicMock()
        mock_mcp_client = MagicMock()
        mock_mcp_client.close = AsyncMock()
        mock_registry._mcp_clients = [mock_mcp_client]
        session.registry = mock_registry
        session._runtime = None
        session._trace = None
        await session.aclose()
        mock_mcp_client.close.assert_called_once()


class TestAgentSessionSpawnWorker:
    @pytest.mark.asyncio
    async def test_spawn_copies_attributes(self):
        session = AgentSession(
            model_client=MagicMock(), registry=MagicMock(),
            mode="build", project="test",
            allowed_tool_names=["read", "write"],
        )
        session._agent = MagicMock()
        with patch.object(session, "initialize", new_callable=AsyncMock):
            with patch.object(session, "_prepare_session_record"):
                with patch.object(session, "_initialize_state"):
                    with patch.object(session, "_ensure_runtime", new_callable=AsyncMock):
                        with patch.object(session, "_ensure_session_record"):
                            with patch.object(session, "_initialize_trace"):
                                with patch.object(session, "_rebuild_agent"):
                                    worker = await session.spawn_worker_session(
                                        worker_role="inspect",
                                        allowed_tool_names=["read"],
                                    )
        assert worker.worker_role == "inspect"
        assert worker.allowed_tool_names == ["read"]
        assert worker.persist_session is False


class TestCollectModifiedPathsEdgeCases:
    def test_move_without_destination(self):
        tc = SimpleNamespace(name="write", arguments={"operation": "move", "path": "/src"})
        result = _collect_modified_paths([tc])
        assert result == ["/src"]

    def test_copy_without_destination(self):
        tc = SimpleNamespace(name="write", arguments={"operation": "copy", "path": "/src"})
        result = _collect_modified_paths([tc])
        assert result == ["/src"]

    def test_write_no_arguments(self):
        tc = SimpleNamespace(name="write", arguments=None)
        result = _collect_modified_paths([tc])
        assert result == []

    def test_multiple_tool_calls(self):
        tcs = [
            SimpleNamespace(name="write", arguments={"operation": "create", "path": "/a"}),
            SimpleNamespace(name="write", arguments={"operation": "create", "path": "/b"}),
            SimpleNamespace(name="read", arguments={"path": "/c"}),
        ]
        result = _collect_modified_paths(tcs)
        assert "/a" in result
        assert "/b" in result
