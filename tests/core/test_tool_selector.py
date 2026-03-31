"""Tests for tool_selector.py"""

from src.core.tool_selector import ToolSelector


class TestToolSelector:
    def test_initialization(self):
        selector = ToolSelector()
        assert selector is not None

    def test_get_tools_for_plan_mode(self):
        selector = ToolSelector()
        tools = selector.get_tools_for_mode("plan")
        assert isinstance(tools, list)

    def test_get_tools_for_build_mode(self):
        selector = ToolSelector()
        tools = selector.get_tools_for_mode("build")
        assert isinstance(tools, list)

    def test_plan_mode_has_fewer_tools(self):
        selector = ToolSelector()
        plan_tools = selector.get_tools_for_mode("plan")
        build_tools = selector.get_tools_for_mode("build")
        assert len(plan_tools) <= len(build_tools)

    def test_get_tools_for_mode_preserves_default_build_order(self):
        selector = ToolSelector()
        tools = selector.get_tools_for_mode("build")
        assert tools.index("read") < tools.index("write")
        assert tools.index("search") < tools.index("run")

    def test_task_tools_are_not_exposed_by_default(self):
        selector = ToolSelector()
        plan_tools = selector.get_tools_for_mode("plan")
        build_tools = selector.get_tools_for_mode("build")

        assert "task" not in plan_tools
        assert "task" not in build_tools
