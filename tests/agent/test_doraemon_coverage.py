"""Targeted coverage tests for agent.doraemon."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.doraemon import DoraemonAgent, create_doraemon_agent


class TestCreateDoraemonAgent:
    def test_creates_agent_with_all_params(self):
        mock_client = MagicMock()
        mock_registry = MagicMock()
        mock_registry._tool_schemas = {}
        mock_registry._tools = {}
        mock_registry.get_tool_names.return_value = []

        agent = create_doraemon_agent(
            llm_client=mock_client,
            tool_registry=mock_registry,
            mode="build",
            max_turns=50,
        )
        assert agent is not None
        assert agent.state.mode == "build"

    def test_creates_agent_plan_mode(self):
        mock_client = MagicMock()
        mock_registry = MagicMock()
        mock_registry._tool_schemas = {}
        mock_registry._tools = {}
        mock_registry.get_tool_names.return_value = []

        agent = create_doraemon_agent(
            llm_client=mock_client,
            tool_registry=mock_registry,
            mode="plan",
        )
        assert agent.state.mode == "plan"


class TestDoraemonAgentSystemPrompt:
    def test_build_mode_prompt(self):
        mock_client = MagicMock()
        mock_registry = MagicMock()
        mock_registry._tool_schemas = {}
        mock_registry._tools = {}
        mock_registry.get_tool_names.return_value = []

        agent = create_doraemon_agent(
            llm_client=mock_client,
            tool_registry=mock_registry,
            mode="build",
        )
        prompt = agent._get_system_prompt()
        assert "coding agent" in prompt.lower()

    def test_plan_mode_prompt(self):
        mock_client = MagicMock()
        mock_registry = MagicMock()
        mock_registry._tool_schemas = {}
        mock_registry._tools = {}
        mock_registry.get_tool_names.return_value = []

        agent = create_doraemon_agent(
            llm_client=mock_client,
            tool_registry=mock_registry,
            mode="plan",
        )
        prompt = agent._get_system_prompt()
        assert "planning agent" in prompt.lower()

    def test_build_mode_worker_role(self):
        mock_client = MagicMock()
        mock_registry = MagicMock()
        mock_registry._tool_schemas = {}
        mock_registry._tools = {}
        mock_registry.get_tool_names.return_value = []

        agent = create_doraemon_agent(
            llm_client=mock_client,
            tool_registry=mock_registry,
            mode="build",
            worker_role="inspect",
        )
        prompt = agent._get_system_prompt()
        assert "inspect" in prompt

    def test_plan_mode_worker_role(self):
        mock_client = MagicMock()
        mock_registry = MagicMock()
        mock_registry._tool_schemas = {}
        mock_registry._tools = {}
        mock_registry.get_tool_names.return_value = []

        agent = create_doraemon_agent(
            llm_client=mock_client,
            tool_registry=mock_registry,
            mode="plan",
            worker_role="validate",
        )
        prompt = agent._get_system_prompt()
        assert "validate" in prompt


class TestDoraemonAgentSessionId:
    def test_session_id_format(self):
        sid = DoraemonAgent._generate_session_id()
        assert len(sid) > 10
        assert "_" in sid


class TestConvertRegistryToTools:
    def test_skips_lazy_tool_with_load_error(self):
        mock_client = MagicMock()
        mock_registry = MagicMock()
        mock_registry._tool_schemas = {}
        mock_registry._tools = {}
        mock_registry.get_tool_names.return_value = []

        lazy_fn = MagicMock()
        lazy_fn._load_error = "import failed"

        tool_def = SimpleNamespace(
            name="broken_tool",
            description="broken",
            parameters={},
            function=lazy_fn,
            sensitive=False,
        )
        mock_registry._tools = {"broken_tool": tool_def}

        agent = create_doraemon_agent(
            llm_client=mock_client,
            tool_registry=mock_registry,
            mode="build",
        )
        tool_names = [t.name for t in agent.tools]
        assert "broken_tool" not in tool_names


class TestDoraemonAgentCheckpointPaths:
    def test_get_checkpoint_paths_move(self):
        paths = DoraemonAgent._get_checkpoint_paths({"path": "/src", "operation": "move", "destination": "/dst"})
        assert "/src" in paths
        assert "/dst" in paths

    def test_get_checkpoint_paths_copy(self):
        paths = DoraemonAgent._get_checkpoint_paths({"path": "/a", "operation": "copy", "destination": "/b"})
        assert "/a" in paths
        assert "/b" in paths

    def test_get_checkpoint_paths_create(self):
        paths = DoraemonAgent._get_checkpoint_paths({"path": "/f"})
        assert paths == ["/f"]

    def test_get_checkpoint_paths_no_path(self):
        paths = DoraemonAgent._get_checkpoint_paths({})
        assert paths == []


class TestIsModifyingTool:
    def test_write_is_modifying(self):
        mock_client = MagicMock()
        mock_registry = MagicMock()
        mock_registry._tool_schemas = {}
        mock_registry._tools = {}
        mock_registry.get_tool_names.return_value = []
        agent = create_doraemon_agent(llm_client=mock_client, tool_registry=mock_registry)
        assert agent._is_modifying_tool("write") is True
        assert agent._is_modifying_tool("read") is False
