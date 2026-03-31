"""Additional tests for the tool selector."""

from src.core.tool_selector import (
    CAPABILITY_GROUPS,
    MODE_CAPABILITY_GROUPS,
    ToolSelector,
    get_capability_groups_for_mode,
    get_default_selector,
    get_tools_for_mode,
)


class TestToolConstants:
    def test_capability_groups_defined(self):
        assert CAPABILITY_GROUPS["read"] == ["read", "search", "ask_user"]
        assert CAPABILITY_GROUPS["edit"] == ["write", "run"]
        assert CAPABILITY_GROUPS["memory"] == [
            "memory_get",
            "memory_put",
            "memory_search",
            "memory_list",
        ]
        assert CAPABILITY_GROUPS["research"] == ["web_search", "web_fetch"]
        assert CAPABILITY_GROUPS["task"] == ["task"]

    def test_mode_capability_groups_defined(self):
        assert MODE_CAPABILITY_GROUPS["plan"] == ["read", "memory", "research", "task"]
        assert MODE_CAPABILITY_GROUPS["build"] == ["read", "edit", "memory", "research", "task"]


class TestToolSelectorExtended:
    def test_plan_mode_excludes_write_tools(self):
        selector = ToolSelector()
        plan_tools = selector.get_tools_for_mode("plan")
        for tool_name in CAPABILITY_GROUPS["edit"]:
            assert tool_name not in plan_tools

    def test_build_mode_includes_write_tools(self):
        selector = ToolSelector()
        build_tools = selector.get_tools_for_mode("build")
        for tool_name in CAPABILITY_GROUPS["edit"]:
            assert tool_name in build_tools

    def test_both_modes_include_read_tools(self):
        selector = ToolSelector()
        plan_tools = selector.get_tools_for_mode("plan")
        build_tools = selector.get_tools_for_mode("build")
        for tool_name in CAPABILITY_GROUPS["read"]:
            assert tool_name in plan_tools
            assert tool_name in build_tools

    def test_both_modes_include_shared_non_edit_capabilities(self):
        selector = ToolSelector()
        plan_tools = selector.get_tools_for_mode("plan")
        build_tools = selector.get_tools_for_mode("build")
        for group_name in ("memory", "research", "task"):
            for tool_name in CAPABILITY_GROUPS[group_name]:
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


class TestGlobalFunctions:
    def test_get_default_selector(self):
        assert isinstance(get_default_selector(), ToolSelector)

    def test_get_default_selector_is_singleton(self):
        assert get_default_selector() is get_default_selector()

    def test_get_tools_for_mode_convenience(self):
        tools = get_tools_for_mode("plan")
        assert isinstance(tools, list)
        assert tools == get_default_selector().get_tools_for_mode("plan")

    def test_get_capability_groups_for_mode_convenience(self):
        groups = get_capability_groups_for_mode("plan")
        assert groups == ["read", "memory", "research", "task"]
