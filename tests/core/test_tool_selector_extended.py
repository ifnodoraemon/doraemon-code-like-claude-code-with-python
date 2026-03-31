"""Additional tests for the simplified tool selector."""

from src.core.tool_selector import (
    AUX_TOOLS,
    READ_TOOLS,
    WRITE_TOOLS,
    ToolSelector,
    get_default_selector,
    get_tools_for_mode,
)


class TestToolConstants:
    def test_read_tools_defined(self):
        assert READ_TOOLS == ["read", "search", "ask_user"]

    def test_write_tools_defined(self):
        assert WRITE_TOOLS == ["write", "run"]

    def test_aux_tools_defined(self):
        assert "task" in AUX_TOOLS
        assert "web_search" in AUX_TOOLS


class TestToolSelectorExtended:
    def test_plan_mode_excludes_write_tools(self):
        selector = ToolSelector()
        plan_tools = selector.get_tools_for_mode("plan")
        for tool_name in WRITE_TOOLS:
            assert tool_name not in plan_tools

    def test_build_mode_includes_write_tools(self):
        selector = ToolSelector()
        build_tools = selector.get_tools_for_mode("build")
        for tool_name in WRITE_TOOLS:
            assert tool_name in build_tools

    def test_both_modes_include_read_tools(self):
        selector = ToolSelector()
        plan_tools = selector.get_tools_for_mode("plan")
        build_tools = selector.get_tools_for_mode("build")
        for tool_name in READ_TOOLS:
            assert tool_name in plan_tools
            assert tool_name in build_tools

    def test_unknown_mode_defaults_to_build(self):
        selector = ToolSelector()
        assert selector.get_tools_for_mode("unknown") == selector.get_tools_for_mode("build")

    def test_get_tools_returns_copies(self):
        selector = ToolSelector()
        first = selector.get_tools_for_mode("plan")
        second = selector.get_tools_for_mode("plan")
        first.append("fake_tool")
        assert "fake_tool" not in second

    def test_build_mode_keeps_stable_default_order(self):
        selector = ToolSelector()
        tools = selector.get_tools_for_mode("build")
        assert tools.index("read") < tools.index("write")
        assert tools.index("search") < tools.index("run")

    def test_aux_tools_are_not_in_default_modes(self):
        selector = ToolSelector()
        plan_tools = selector.get_tools_for_mode("plan")
        build_tools = selector.get_tools_for_mode("build")
        for tool_name in AUX_TOOLS:
            assert tool_name not in plan_tools
            assert tool_name not in build_tools


class TestGlobalFunctions:
    def test_get_default_selector(self):
        assert isinstance(get_default_selector(), ToolSelector)

    def test_get_default_selector_is_singleton(self):
        assert get_default_selector() is get_default_selector()

    def test_get_tools_for_mode_convenience(self):
        tools = get_tools_for_mode("plan")
        assert isinstance(tools, list)
        assert tools == get_default_selector().get_tools_for_mode("plan")
