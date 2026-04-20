"""Targeted coverage for agent/doraemon.py uncovered lines: 134,138-140,149,210,231,249,345-347,391,407-418,520-528,566,575,726."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.doraemon import DoraemonAgent, create_doraemon_agent, create_doraemon_agent_with_tools


def _make_agent(**kw):
    mock_client = MagicMock()
    mock_registry = MagicMock()
    mock_registry._tool_schemas = {}
    mock_registry._tools = {}
    mock_registry.get_tool_names.return_value = []
    defaults = {"llm_client": mock_client, "tool_registry": mock_registry, "mode": "build"}
    defaults.update(kw)
    return create_doraemon_agent(**defaults)


class TestDoraemonAgentTraceSave:
    def test_save_trace_returns_path(self):
        agent = _make_agent(enable_trace=True)
        agent._trace = MagicMock()
        agent._trace.save.return_value = "/tmp/trace.json"
        result = agent.save_trace()
        assert result == "/tmp/trace.json"

    def test_save_trace_no_trace(self):
        agent = _make_agent(enable_trace=False)
        agent._trace = None
        result = agent.save_trace()
        assert result is None


class TestDoraemonAgentRunWithException:
    @pytest.mark.asyncio
    async def test_run_exception_cleans_up_trace(self):
        agent = _make_agent(enable_trace=True)
        agent._trace = MagicMock()
        agent._trace.start_turn = MagicMock()
        agent._trace.error = MagicMock()
        agent._trace.end_turn = MagicMock()
        with patch("src.agent.react.ReActAgent.run", new_callable=AsyncMock, side_effect=RuntimeError("crash")):
            with pytest.raises(RuntimeError, match="crash"):
                await agent.run("test input")
        agent._trace.error.assert_called()
        agent._trace.end_turn.assert_called()


class TestDoraemonAgentRunWithTaskManager:
    @pytest.mark.asyncio
    async def test_run_sets_and_resets_task_manager(self):
        agent = _make_agent(enable_trace=False)
        agent._trace = None
        agent.task_manager = MagicMock()
        with patch("src.agent.react.ReActAgent.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = SimpleNamespace(success=True, error=None, response="ok", tool_calls=[], tokens_used=0)
            with patch("src.servers.task.set_task_manager", return_value="token") as mock_set:
                with patch("src.servers.task.reset_task_manager") as mock_reset:
                    result = await agent.run("test")
                    mock_set.assert_called_once()
                    mock_reset.assert_called_once_with("token")


class TestDoraemonAgentThinkWithDisplay:
    @pytest.mark.asyncio
    async def test_think_calls_display_callback(self):
        agent = _make_agent()
        agent.display_callback = AsyncMock()
        with patch("src.agent.react.ReActAgent.think", new_callable=AsyncMock) as mock_think:
            from src.agent.types import Thought, Observation
            mock_think.return_value = Thought(reasoning="my reasoning", response="done")
            result = await agent.think(Observation(user_input="test"))
        assert agent.display_callback.call_count >= 1


class TestDoraemonAgentActWithDisplay:
    @pytest.mark.asyncio
    async def test_act_with_display_callback(self):
        agent = _make_agent()
        agent.display_callback = AsyncMock()
        with patch("src.agent.react.ReActAgent.act", new_callable=AsyncMock) as mock_act:
            from src.agent.types import Action, ActionType
            mock_act.return_value = Action(type=ActionType.RESPOND, response="ok")
            mock_act.return_value.to_dict = lambda: {"type": "respond", "response": "ok"}
            result = await agent.act(MagicMock())
        agent.display_callback.assert_any_await("action", {"action": {"type": "respond", "response": "ok"}})


class TestDoraemonAgentExecuteToolWithPostHook:
    @pytest.mark.asyncio
    async def test_execute_tool_post_hook_triggered(self):
        agent = _make_agent()
        registry = agent.tool_registry
        registry.call_tool = AsyncMock(return_value="ok")
        registry._tools = {}
        hooks = MagicMock()
        hook_result = MagicMock()
        hook_result.decision = MagicMock(value="allow")
        hook_result.modified_input = None
        hooks.trigger = AsyncMock(return_value=hook_result)
        agent.hooks = hooks
        result, error = await agent.execute_tool("read", {"path": "/f"})
        assert hooks.trigger.call_count == 2


class TestDoraemonAgentExecuteToolCheckpoint:
    @pytest.mark.asyncio
    async def test_write_triggers_checkpoint(self):
        agent = _make_agent()
        registry = agent.tool_registry
        registry.call_tool = AsyncMock(return_value="ok")
        registry._tools = {}
        checkpoints = MagicMock()
        checkpoints.snapshot = MagicMock()
        agent.checkpoints = checkpoints
        result, error = await agent.execute_tool("write", {"path": "/f"})
        checkpoints.snapshot.assert_called_once()


class TestDoraemonAgentConvertRegistryInvisibleTool:
    def test_invisible_tool_excluded(self):
        mock_client = MagicMock()
        mock_registry = MagicMock()
        mock_registry._tool_schemas = {}
        mock_registry._tools = {}
        mock_registry.get_tool_names.return_value = ["hidden"]
        mock_registry._tools = {
            "hidden": SimpleNamespace(
                name="hidden", description="hidden tool", parameters={}, sensitive=False, function=None,
            ),
        }
        mock_registry.get_tool_policy = MagicMock(return_value={
            "visible": False, "requires_approval": True,
        })
        agent = create_doraemon_agent(llm_client=mock_client, tool_registry=mock_registry, mode="build")
        tool_names = [t.name for t in agent.tools]
        assert "hidden" not in tool_names


class TestCreateDoraemonAgentWithToolsCompat:
    @pytest.mark.asyncio
    async def test_alias_calls_create_with_tools(self):
        with patch("src.host.mcp_registry.create_tool_registry", new_callable=AsyncMock) as mock_create:
            mock_registry = MagicMock()
            mock_registry._active_mcp_extensions = []
            mock_registry._tool_schemas = {}
            mock_registry._tools = {}
            mock_registry.get_tool_names.return_value = []
            mock_create.return_value = mock_registry
            result = await create_doraemon_agent_with_tools(
                llm_client=MagicMock(), mode="build",
            )
            assert result is not None


class TestCreateDoraemonAgentWithMcpCompat:
    @pytest.mark.asyncio
    async def test_mcp_alias(self):
        with patch("src.host.mcp_registry.create_tool_registry", new_callable=AsyncMock) as mock_create:
            mock_registry = MagicMock()
            mock_registry._active_mcp_extensions = []
            mock_registry._tool_schemas = {}
            mock_registry._tools = {}
            mock_registry.get_tool_names.return_value = []
            mock_create.return_value = mock_registry
            from src.agent.doraemon import create_doraemon_agent_with_mcp
            result = await create_doraemon_agent_with_mcp(
                llm_client=MagicMock(), mode="plan",
            )
            assert result is not None
