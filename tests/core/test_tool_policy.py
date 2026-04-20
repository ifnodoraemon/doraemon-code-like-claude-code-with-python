"""Tests for src.core.tool_policy."""

from unittest.mock import patch

from src.core.tool_policy import ToolPolicy, ToolPolicyEngine, get_default_tool_policy_engine


class TestToolPolicy:
    def test_to_dict(self):
        p = ToolPolicy(
            tool_name="read",
            visible=True,
            visible_modes=["plan", "build"],
            requires_approval=False,
            sandbox="workspace_read",
            audit_level="basic",
            background_safe=True,
            capability_group="read",
            source="built_in",
        )
        d = p.to_dict()
        assert d["tool_name"] == "read"
        assert d["visible"] is True
        assert d["sandbox"] == "workspace_read"


class TestToolPolicyEngine:
    def test_describe_built_in_tool(self):
        engine = ToolPolicyEngine()
        with (
            patch(
                "src.core.tool_policy.get_tools_for_mode",
                return_value=["read"],
            ),
            patch(
                "src.core.tool_policy.get_capability_group_for_tool",
                return_value="read",
            ),
            patch(
                "src.core.tool_policy.get_visible_modes_for_tool",
                return_value=["plan", "build"],
            ),
            patch(
                "src.core.tool_policy.get_capability_groups_for_mode",
                return_value=["read"],
            ),
        ):
            policy = engine.describe_tool("read", mode="build")
            assert policy.tool_name == "read"
            assert policy.sandbox == "workspace_read"

    def test_describe_write_tool(self):
        engine = ToolPolicyEngine()
        with (
            patch(
                "src.core.tool_policy.get_tools_for_mode",
                return_value=["write"],
            ),
            patch(
                "src.core.tool_policy.get_capability_group_for_tool",
                return_value="edit",
            ),
            patch(
                "src.core.tool_policy.get_visible_modes_for_tool",
                return_value=["build"],
            ),
            patch(
                "src.core.tool_policy.get_capability_groups_for_mode",
                return_value=["edit"],
            ),
        ):
            policy = engine.describe_tool("write", mode="build")
            assert policy.sandbox == "workspace_write"
            assert policy.requires_approval is True

    def test_describe_run_tool(self):
        engine = ToolPolicyEngine()
        with (
            patch(
                "src.core.tool_policy.get_tools_for_mode",
                return_value=["run"],
            ),
            patch(
                "src.core.tool_policy.get_capability_group_for_tool",
                return_value=None,
            ),
            patch(
                "src.core.tool_policy.get_visible_modes_for_tool",
                return_value=["build"],
            ),
            patch(
                "src.core.tool_policy.get_capability_groups_for_mode",
                return_value=[],
            ),
        ):
            policy = engine.describe_tool("run", mode="build")
            assert policy.sandbox == "workspace_exec"
            assert policy.requires_approval is True

    def test_describe_mcp_extension(self):
        engine = ToolPolicyEngine()
        with (
            patch(
                "src.core.tool_policy.get_tools_for_mode",
                return_value=[],
            ),
            patch(
                "src.core.tool_policy.get_capability_group_for_tool",
                return_value=None,
            ),
            patch(
                "src.core.tool_policy.get_visible_modes_for_tool",
                return_value=[],
            ),
            patch(
                "src.core.tool_policy.get_capability_groups_for_mode",
                return_value=[],
            ),
        ):
            policy = engine.describe_tool(
                "ext_tool",
                mode="build",
                source="mcp_extension",
                metadata={"extension_group": "browser"},
                active_mcp_extensions=["browser"],
            )
            assert policy.visible is True
            assert policy.sandbox == "extension"
            assert policy.audit_level == "full"

    def test_describe_mcp_remote(self):
        engine = ToolPolicyEngine()
        with (
            patch(
                "src.core.tool_policy.get_tools_for_mode",
                return_value=[],
            ),
            patch(
                "src.core.tool_policy.get_capability_group_for_tool",
                return_value=None,
            ),
            patch(
                "src.core.tool_policy.get_visible_modes_for_tool",
                return_value=[],
            ),
            patch(
                "src.core.tool_policy.get_capability_groups_for_mode",
                return_value=[],
            ),
        ):
            policy = engine.describe_tool(
                "remote_tool",
                mode="build",
                source="mcp_remote",
                metadata={"mcp_server": "my_server"},
                active_mcp_extensions=["my_server"],
            )
            assert policy.visible is True
            assert policy.sandbox == "mcp_remote"

    def test_mode_none_always_visible(self):
        engine = ToolPolicyEngine()
        with (
            patch(
                "src.core.tool_policy.get_tools_for_mode",
                return_value=[],
            ),
            patch(
                "src.core.tool_policy.get_capability_group_for_tool",
                return_value=None,
            ),
        ):
            policy = engine.describe_tool("read", mode=None)
            assert policy.visible is True

    def test_interactive_not_background_safe(self):
        engine = ToolPolicyEngine()
        with (
            patch(
                "src.core.tool_policy.get_tools_for_mode",
                return_value=["ask_user"],
            ),
            patch(
                "src.core.tool_policy.get_capability_group_for_tool",
                return_value=None,
            ),
            patch(
                "src.core.tool_policy.get_visible_modes_for_tool",
                return_value=["plan", "build"],
            ),
            patch(
                "src.core.tool_policy.get_capability_groups_for_mode",
                return_value=[],
            ),
        ):
            policy = engine.describe_tool("ask_user", mode="build")
            assert policy.background_safe is False

    def test_sensitive_requires_approval(self):
        engine = ToolPolicyEngine()
        with (
            patch(
                "src.core.tool_policy.get_tools_for_mode",
                return_value=["read"],
            ),
            patch(
                "src.core.tool_policy.get_capability_group_for_tool",
                return_value="read",
            ),
            patch(
                "src.core.tool_policy.get_visible_modes_for_tool",
                return_value=["plan", "build"],
            ),
            patch(
                "src.core.tool_policy.get_capability_groups_for_mode",
                return_value=["read"],
            ),
        ):
            policy = engine.describe_tool("read", mode="build", sensitive=True)
            assert policy.requires_approval is True

    def test_visible_modes_for_mcp(self):
        engine = ToolPolicyEngine()
        modes = engine._visible_modes("x", "mcp_extension", None)
        assert modes == ["plan", "build"]


class TestGetDefaultToolPolicyEngine:
    def test_singleton(self):
        e1 = get_default_tool_policy_engine()
        e2 = get_default_tool_policy_engine()
        assert e1 is e2
