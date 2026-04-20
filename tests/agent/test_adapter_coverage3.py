"""Targeted coverage for agent/adapter.py uncovered lines: 161,165-178,184-192,338,354,473,498,510,551,578-584,650,787,803,840,866-868,872,877."""

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
)
from src.agent.types import Message


class TestAgentSessionOrchestrateResume:
    @pytest.mark.asyncio
    async def test_orchestrate_resume_with_prior_run(self):
        session = AgentSession(model_client=None, registry=None)
        session._agent = MagicMock()
        prior_run = {"goal": "resume goal", "summary": "was blocked", "root_task_id": "root-1"}
        with patch.object(session, "_get_orchestration_run", return_value=prior_run):
            with patch("src.agent.adapter.LeadAgentRuntime") as MockRuntime:
                runtime_instance = MagicMock()
                mock_result = SimpleNamespace(
                    success=True, summary="resumed", root_task_id="root-1",
                    plan_id=None, executed_task_ids=[], completed_task_ids=[],
                    failed_task_ids=[], blocked_task_id=None,
                    task_summaries={}, worker_assignments={},
                    to_dict=lambda: {},
                )
                runtime_instance.resume = AsyncMock(return_value=mock_result)
                MockRuntime.return_value = runtime_instance
                session._state = MagicMock()
                session._state.messages = []
                session._session_manager = None
                session._session_record = None
                result = await session.orchestrate("", resume_run_id="run-1", max_workers=2)
        assert result.success is True


class TestAgentSessionOrchestrateErrorPaths:
    @pytest.mark.asyncio
    async def test_orchestrate_generic_exception_reraises(self):
        session = AgentSession(model_client=None, registry=None)
        session._agent = MagicMock()
        session._state = MagicMock()
        session._state.messages = []
        session._session_manager = None
        session._session_record = None
        with patch("src.agent.adapter.LeadAgentRuntime") as MockRuntime:
            runtime_instance = MagicMock()
            runtime_instance.execute = AsyncMock(side_effect=RuntimeError("fatal"))
            MockRuntime.return_value = runtime_instance
            session._trace = MagicMock()
            session._trace.error = MagicMock()
            session._trace.end_turn = MagicMock()
            with pytest.raises(RuntimeError, match="fatal"):
                await session.orchestrate("goal")


class TestAgentSessionTurnStreamError:
    @pytest.mark.asyncio
    async def test_turn_stream_non_categorized_error(self):
        session = AgentSession(model_client=None, registry=None)
        mock_agent = MagicMock()

        async def fake_stream(input, **kw):
            yield {"type": "text", "content": "partial"}
            raise TypeError("bad type")

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


class TestAgentSessionCloseSync:
    def test_close_with_running_loop(self):
        session = AgentSession(model_client=None, registry=None)
        session._runtime = None
        session._trace = None
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(session.aclose())
        finally:
            loop.close()


class TestAgentSessionPrepareSessionNoPersist:
    def test_prepare_session_no_persist(self):
        session = AgentSession(model_client=None, registry=None, persist_session=False)
        session._prepare_session_record()
        assert session._session_manager is None
        assert session._session_record is None
        assert session._session_id is not None


class TestAgentSessionEnsureSessionRecord:
    def test_ensure_session_record_creates_when_needed(self):
        session = AgentSession(model_client=None, registry=None, persist_session=True)
        sm = MagicMock()
        record = MagicMock()
        record.metadata = MagicMock()
        record.metadata.id = "new-id"
        sm.create_session.return_value = record
        session._session_manager = sm
        session._session_record = None
        session._ensure_session_record()
        sm.create_session.assert_called_once()
        assert session._session_id == "new-id"

    def test_ensure_session_record_skips_when_no_persist(self):
        session = AgentSession(model_client=None, registry=None, persist_session=False)
        session._ensure_session_record()
        assert session._session_record is None

    def test_ensure_session_record_skips_when_record_exists(self):
        session = AgentSession(model_client=None, registry=None, persist_session=True)
        session._session_record = MagicMock()
        session._ensure_session_record()
        assert session._session_record is not None


class TestAgentSessionRollbackOrchestration:
    def test_rollback_orchestration_messages(self):
        session = AgentSession(model_client=None, registry=None)
        from src.agent.state import AgentState
        session._state = AgentState(mode="build", max_turns=100)
        session._state.add_user_message("before")
        session._state.add_assistant_message("response")
        session._state.add_user_message("orchestration input")
        session._state.add_assistant_message("orchestration response")
        initial_count = len(session._state.messages)
        session._rollback_orchestration_messages(2)
        assert len(session._state.messages) == 2


class TestAgentSessionGetOrchestrationRun:
    def test_get_run_from_state(self):
        session = AgentSession(model_client=None, registry=None)
        record = MagicMock()
        record.orchestration_runs = [{"run_id": "r1"}, {"run_id": "r2"}]
        record.orchestration_state = {"run_id": "r3"}
        session._session_record = record
        assert session._get_orchestration_run("r2") is not None
        assert session._get_orchestration_run("r3") is not None
        assert session._get_orchestration_run("r99") is None

    def test_get_run_no_record(self):
        session = AgentSession(model_client=None, registry=None)
        assert session._get_orchestration_run("r1") is None


class TestAgentSessionDeriveSessionName:
    def test_no_user_messages(self):
        session = AgentSession(model_client=None, registry=None)
        session._state = MagicMock()
        session._state.messages = [Message(role="assistant", content="hi")]
        result = session._derive_session_name()
        assert result is None

    def test_empty_user_content_skipped(self):
        session = AgentSession(model_client=None, registry=None)
        session._state = MagicMock()
        session._state.messages = [
            Message(role="user", content="   "),
            Message(role="user", content="valid input"),
        ]
        result = session._derive_session_name()
        assert "valid input" in result
