"""Additional coverage tests for agent.doraemon - execute_tool, run with trace, _begin/_finish runtime task, permission checks, state management."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.doraemon import DoraemonAgent, create_doraemon_agent


def _make_agent(**kw):
    mock_client = MagicMock()
    mock_registry = MagicMock()
    mock_registry._tool_schemas = {}
    mock_registry._tools = {}
    mock_registry.get_tool_names.return_value = []
    defaults = {"llm_client": mock_client, "tool_registry": mock_registry, "mode": "build"}
    defaults.update(kw)
    return create_doraemon_agent(**defaults)


class TestExecuteToolPermissionDenied:
    @pytest.mark.asyncio
    async def test_worker_tool_scope_denied(self):
        agent = _make_agent(allowed_tool_names=["read"])
        agent.allowed_tool_names = {"read"}
        agent.worker_role = "inspect"
        result, error = await agent.execute_tool("write", {"path": "/f"})
        assert "Permission Error" in error

    @pytest.mark.asyncio
    async def test_check_tool_execution_denied(self):
        agent = _make_agent()
        registry = agent.tool_registry
        registry.check_tool_execution = MagicMock(return_value=(False, "not allowed", {}))
        result, error = await agent.execute_tool("write", {"path": "/f"})
        assert "Permission Error" in error

    @pytest.mark.asyncio
    async def test_check_tool_execution_tuple_allowed(self):
        agent = _make_agent()
        registry = agent.tool_registry
        registry.call_tool = AsyncMock(return_value="ok")
        registry.check_tool_execution = MagicMock(return_value=(True, None, {}))
        registry._tools = {}
        result, error = await agent.execute_tool("read", {"path": "/f"})
        assert result == "ok"
        assert error is None

    @pytest.mark.asyncio
    async def test_check_tool_execution_non_tuple_allowed(self):
        agent = _make_agent()
        registry = agent.tool_registry
        registry.call_tool = AsyncMock(return_value="ok")
        registry.check_tool_execution = MagicMock(return_value="not_a_tuple")
        registry.get_tool_policy = MagicMock(return_value={"visible": True, "requires_approval": False})
        registry._tools = {}
        result, error = await agent.execute_tool("read", {"path": "/f"})
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_check_tool_execution_non_tuple_invisible(self):
        agent = _make_agent()
        registry = agent.tool_registry
        registry.call_tool = MagicMock()
        registry.check_tool_execution = MagicMock(return_value="not_a_tuple")
        registry.get_tool_policy = MagicMock(return_value={"visible": False, "requires_approval": True})
        registry._tools = {}
        result, error = await agent.execute_tool("write", {"path": "/f"})
        assert "Permission Error" in error


class TestExecuteToolHooks:
    @pytest.mark.asyncio
    async def test_hook_deny(self):
        agent = _make_agent()
        hooks = MagicMock()
        hook_result = MagicMock()
        hook_result.decision = MagicMock(value="deny")
        hook_result.reason = "blocked"
        hooks.trigger = AsyncMock(return_value=hook_result)
        agent.hooks = hooks
        result, error = await agent.execute_tool("read", {"path": "/f"})
        assert "blocked" in error

    @pytest.mark.asyncio
    async def test_hook_modifies_input(self):
        agent = _make_agent()
        registry = agent.tool_registry
        registry.call_tool = AsyncMock(return_value="modified ok")
        registry._tools = {}
        hooks = MagicMock()
        hook_result = MagicMock()
        hook_result.decision = MagicMock(value="allow")
        hook_result.modified_input = {"path": "/modified"}
        hooks.trigger = AsyncMock(return_value=hook_result)
        agent.hooks = hooks
        result, error = await agent.execute_tool("read", {"path": "/f"})
        assert result == "modified ok"
        registry.call_tool.assert_called_with("read", {"path": "/modified"})


class TestExecuteToolTrace:
    @pytest.mark.asyncio
    async def test_trace_records_tool_call(self):
        agent = _make_agent(enable_trace=True)
        registry = agent.tool_registry
        registry.call_tool = AsyncMock(return_value="ok")
        registry._tools = {"read": SimpleNamespace(source="built_in", metadata=None)}
        agent._trace = MagicMock()
        result, error = await agent.execute_tool("read", {"path": "/f"})
        agent._trace.tool_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_trace_records_with_run_id(self):
        agent = _make_agent(enable_trace=True)
        registry = agent.tool_registry
        registry.call_tool = AsyncMock(return_value="ok")
        registry._tools = {"read": SimpleNamespace(source="built_in", metadata=None)}
        agent._trace = MagicMock()
        agent._active_trace_run_id = "run-1"
        result, error = await agent.execute_tool("read", {"path": "/f"})
        call_kwargs = agent._trace.tool_call.call_args
        assert call_kwargs[1].get("metadata", {}).get("run_id") == "run-1"


class TestExecuteToolRecordExecution:
    @pytest.mark.asyncio
    async def test_record_tool_execution_on_registry(self):
        agent = _make_agent()
        registry = agent.tool_registry
        registry.call_tool = AsyncMock(return_value="ok")
        registry._tools = {}
        registry.record_tool_execution = MagicMock()
        result, error = await agent.execute_tool("read", {"path": "/f"})
        registry.record_tool_execution.assert_called_once()


class TestRuntimeTaskManagement:
    def test_begin_runtime_task_with_task_manager(self):
        agent = _make_agent()
        tm = MagicMock()
        tm.create_task.return_value = SimpleNamespace(id="task-1")
        tm.claim_task = MagicMock()
        agent.task_manager = tm
        task_id = agent._begin_runtime_task("do something")
        assert task_id == "task-1"
        tm.claim_task.assert_called_once()

    def test_begin_runtime_task_claim_fallback(self):
        agent = _make_agent()
        tm = MagicMock()
        tm.create_task.return_value = SimpleNamespace(id="task-1")
        tm.claim_task.side_effect = Exception("already claimed")
        tm.update_task = MagicMock()
        agent.task_manager = tm
        task_id = agent._begin_runtime_task("do something")
        assert task_id == "task-1"
        tm.update_task.assert_called()

    def test_begin_runtime_task_no_task_manager(self):
        agent = _make_agent()
        agent.task_manager = None
        result = agent._begin_runtime_task("do something")
        assert result is None

    def test_begin_runtime_task_long_title(self):
        agent = _make_agent()
        tm = MagicMock()
        tm.create_task.return_value = SimpleNamespace(id="task-1")
        tm.claim_task = MagicMock()
        agent.task_manager = tm
        long_input = "x" * 100
        task_id = agent._begin_runtime_task(long_input)
        create_call = tm.create_task.call_args
        assert len(create_call[1]["title"]) <= 80

    def test_finish_runtime_task_success(self):
        agent = _make_agent()
        tm = MagicMock()
        agent.task_manager = tm
        agent._finish_runtime_task("task-1", success=True)
        tm.update_task.assert_called_once()
        call_args = tm.update_task.call_args
        assert call_args[0][0] == "task-1"

    def test_finish_runtime_task_no_task_manager(self):
        agent = _make_agent()
        agent.task_manager = None
        agent._finish_runtime_task("task-1", success=True)

    def test_finish_runtime_task_none_id(self):
        agent = _make_agent()
        tm = MagicMock()
        agent.task_manager = tm
        agent._finish_runtime_task(None, success=True)
        tm.update_task.assert_not_called()


class TestRunWithTrace:
    @pytest.mark.asyncio
    async def test_run_creates_trace_turn(self):
        agent = _make_agent(enable_trace=True)
        agent._trace = MagicMock()
        with patch("src.agent.react.ReActAgent.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = SimpleNamespace(success=True, error=None, response="ok", tool_calls=[], tokens_used=0)
            result = await agent.run("test input")
        agent._trace.start_turn.assert_called_once()
        agent._trace.end_turn.assert_called_once()


class TestObserveWithSkills:
    @pytest.mark.asyncio
    async def test_observe_adds_skills_context(self):
        agent = _make_agent()
        skills = MagicMock()
        skills.get_skills_for_context.return_value = "skill info"
        skills.get_active_skills.return_value = ["skill1"]
        agent.skills = skills
        agent._trace = MagicMock()
        agent.enable_trace = True
        with patch("src.agent.react.ReActAgent.observe", new_callable=AsyncMock) as mock_observe:
            from src.agent.types import Observation
            mock_observe.return_value = Observation(user_input="test", context={})
            obs = await agent.observe()
        assert obs.context.get("skills") == "skill info"
        agent._trace.event.assert_called()


class TestBuildMessagesWithSkills:
    def test_build_messages_injects_skills(self):
        agent = _make_agent()
        from src.agent.types import Observation
        obs = Observation(user_input="test", context={"skills": "skill info"})
        with patch("src.agent.react.ReActAgent._build_messages", return_value=[{"role": "system", "content": "base"}]):
            messages = agent._build_messages(obs)
        assert "skill info" in messages[0]["content"]


class TestCreateCheckpoint:
    @pytest.mark.asyncio
    async def test_create_checkpoint_success(self):
        agent = _make_agent()
        checkpoints = MagicMock()
        checkpoints.snapshot = MagicMock()
        agent.checkpoints = checkpoints
        await agent._create_checkpoint("write", {"path": "/f"})
        checkpoints.snapshot.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_checkpoint_failure_logged(self):
        agent = _make_agent()
        checkpoints = MagicMock()
        checkpoints.snapshot.side_effect = Exception("snap fail")
        agent.checkpoints = checkpoints
        await agent._create_checkpoint("write", {"path": "/f"})

    @pytest.mark.asyncio
    async def test_create_checkpoint_no_checkpoints(self):
        agent = _make_agent()
        agent.checkpoints = None
        await agent._create_checkpoint("write", {"path": "/f"})


class TestGetRuntimeToolPolicy:
    def test_with_get_tool_policy(self):
        agent = _make_agent()
        registry = agent.tool_registry
        registry.get_tool_policy = MagicMock(return_value={"visible": True, "requires_approval": False})
        policy = agent._get_runtime_tool_policy("read")
        assert policy["visible"] is True

    def test_no_get_tool_policy(self):
        agent = _make_agent()
        registry = agent.tool_registry
        delattr(registry, "get_tool_policy")
        policy = agent._get_runtime_tool_policy("read")
        assert policy is None

    def test_get_tool_policy_non_dict(self):
        agent = _make_agent()
        registry = agent.tool_registry
        registry.get_tool_policy = MagicMock(return_value="not a dict")
        policy = agent._get_runtime_tool_policy("read")
        assert policy is None
