"""Additional comprehensive tests for tool_selector.py"""

from src.core.tool_selector import (
    ADVANCED_TOOLS,
    AUX_TOOLS,
    READ_TOOLS,
    WRITE_TOOLS,
    ToolConfig,
    ToolSelector,
    get_default_selector,
    get_tools_for_mode,
)


class TestToolConstants:
    """Tests for tool constant lists."""

    def test_read_tools_defined(self):
        """Test that READ_TOOLS is defined."""
        assert isinstance(READ_TOOLS, list)
        assert len(READ_TOOLS) > 0
        assert "read" in READ_TOOLS

    def test_write_tools_defined(self):
        """Test that WRITE_TOOLS is defined."""
        assert isinstance(WRITE_TOOLS, list)
        assert len(WRITE_TOOLS) > 0
        assert "write" in WRITE_TOOLS
        assert "run" in WRITE_TOOLS

    def test_aux_tools_defined(self):
        """Test that AUX_TOOLS is defined."""
        assert isinstance(AUX_TOOLS, list)
        assert len(AUX_TOOLS) > 0

    def test_advanced_tools_defined(self):
        """Test that ADVANCED_TOOLS is defined."""
        assert isinstance(ADVANCED_TOOLS, list)
        assert len(ADVANCED_TOOLS) > 0


class TestToolConfig:
    """Tests for ToolConfig dataclass."""

    def test_creation_with_defaults(self):
        """Test creating ToolConfig with defaults."""
        config = ToolConfig()
        assert config.plan_tools == []
        assert config.build_tools == []
        assert config.mcp_tools == []

    def test_creation_with_values(self):
        """Test creating ToolConfig with values."""
        config = ToolConfig(
            plan_tools=["read_file"],
            build_tools=["read_file", "write_file"],
            mcp_tools=["custom_tool"]
        )
        assert len(config.plan_tools) == 1
        assert len(config.build_tools) == 2
        assert len(config.mcp_tools) == 1


class TestToolSelectorExtended:
    """Extended tests for ToolSelector."""

    def test_plan_mode_excludes_write_tools(self):
        """Test that plan mode doesn't include write tools."""
        selector = ToolSelector()
        plan_tools = selector.get_tools_for_mode("plan")
        for write_tool in WRITE_TOOLS:
            assert write_tool not in plan_tools

    def test_build_mode_includes_write_tools(self):
        """Test that build mode includes write tools."""
        selector = ToolSelector()
        build_tools = selector.get_tools_for_mode("build")
        for write_tool in WRITE_TOOLS:
            assert write_tool in build_tools

    def test_both_modes_include_read_tools(self):
        """Test that both modes include read tools."""
        selector = ToolSelector()
        plan_tools = selector.get_tools_for_mode("plan")
        build_tools = selector.get_tools_for_mode("build")
        for read_tool in READ_TOOLS:
            assert read_tool in plan_tools
            assert read_tool in build_tools

    def test_register_mcp_tools_single(self):
        """Test registering a single MCP tool."""
        selector = ToolSelector()
        selector.register_mcp_tools(["custom_tool"])
        assert "custom_tool" in selector.mcp_tools

    def test_register_mcp_tools_multiple(self):
        """Test registering multiple MCP tools."""
        selector = ToolSelector()
        selector.register_mcp_tools(["tool1", "tool2", "tool3"])
        assert len(selector.mcp_tools) == 3
        assert "tool1" in selector.mcp_tools
        assert "tool2" in selector.mcp_tools

    def test_register_mcp_tools_no_duplicates(self):
        """Test that registering same tool twice doesn't duplicate."""
        selector = ToolSelector()
        selector.register_mcp_tools(["tool1"])
        selector.register_mcp_tools(["tool1"])
        assert selector.mcp_tools.count("tool1") == 1

    def test_unregister_mcp_tools_specific(self):
        """Test unregistering specific MCP tools."""
        selector = ToolSelector()
        selector.register_mcp_tools(["tool1", "tool2", "tool3"])
        selector.unregister_mcp_tools(["tool2"])
        assert "tool1" in selector.mcp_tools
        assert "tool2" not in selector.mcp_tools
        assert "tool3" in selector.mcp_tools

    def test_unregister_mcp_tools_all(self):
        """Test unregistering all MCP tools."""
        selector = ToolSelector()
        selector.register_mcp_tools(["tool1", "tool2"])
        selector.unregister_mcp_tools(None)
        assert len(selector.mcp_tools) == 0

    def test_unregister_nonexistent_tool(self):
        """Test unregistering tool that doesn't exist."""
        selector = ToolSelector()
        selector.register_mcp_tools(["tool1"])
        selector.unregister_mcp_tools(["nonexistent"])
        assert "tool1" in selector.mcp_tools

    def test_mcp_tools_included_in_both_modes(self):
        """Test that MCP tools are included in both modes."""
        selector = ToolSelector()
        selector.register_mcp_tools(["custom_tool"])
        plan_tools = selector.get_tools_for_mode("plan")
        build_tools = selector.get_tools_for_mode("build")
        assert "custom_tool" in plan_tools
        assert "custom_tool" in build_tools

    def test_get_all_builtin_tools(self):
        """Test getting all builtin tools."""
        selector = ToolSelector()
        all_tools = selector.get_all_builtin_tools()
        assert isinstance(all_tools, list)
        assert len(all_tools) > 0
        # Should include tools from all categories
        assert any(tool in all_tools for tool in READ_TOOLS)
        assert any(tool in all_tools for tool in WRITE_TOOLS)

    def test_get_all_builtin_tools_no_duplicates(self):
        """Test that builtin tools list has no duplicates."""
        selector = ToolSelector()
        all_tools = selector.get_all_builtin_tools()
        assert len(all_tools) == len(set(all_tools))

    def test_get_tool_categories(self):
        """Test getting tool categories."""
        selector = ToolSelector()
        categories = selector.get_tool_categories()
        assert isinstance(categories, dict)
        assert "read" in categories
        assert "write" in categories
        assert "aux" in categories
        assert "advanced" in categories
        assert "mcp" in categories

    def test_get_tool_categories_with_mcp(self):
        """Test tool categories includes registered MCP tools."""
        selector = ToolSelector()
        selector.register_mcp_tools(["custom1", "custom2"])
        categories = selector.get_tool_categories()
        assert len(categories["mcp"]) == 2
        assert "custom1" in categories["mcp"]

    def test_tools_are_copied_not_referenced(self):
        """Test that get_tools_for_mode returns copies."""
        selector = ToolSelector()
        tools1 = selector.get_tools_for_mode("plan")
        tools2 = selector.get_tools_for_mode("plan")
        # Modifying one shouldn't affect the other
        tools1.append("fake_tool")
        assert "fake_tool" not in tools2

    def test_unknown_mode_defaults_to_build(self):
        """Test that unknown mode defaults to build mode."""
        selector = ToolSelector()
        unknown_tools = selector.get_tools_for_mode("unknown")
        build_tools = selector.get_tools_for_mode("build")
        assert set(unknown_tools) == set(build_tools)

    def test_build_mode_keeps_stable_default_order(self):
        """Test that build mode uses the configured default order."""
        selector = ToolSelector()
        tools = selector.get_tools_for_mode("build")
        assert tools.index("read") < tools.index("write")
        assert tools.index("write") < tools.index("web_search")


class TestGlobalFunctions:
    """Tests for module-level functions."""

    def test_get_default_selector(self):
        """Test getting default selector."""
        selector = get_default_selector()
        assert isinstance(selector, ToolSelector)

    def test_get_default_selector_is_singleton(self):
        """Test that default selector is a singleton."""
        selector1 = get_default_selector()
        selector2 = get_default_selector()
        assert selector1 is selector2

    def test_get_tools_for_mode_convenience(self):
        """Test convenience function for getting tools."""
        tools = get_tools_for_mode("plan")
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_get_tools_for_mode_uses_default_selector(self):
        """Test that convenience function uses default selector."""
        # Register MCP tool on default selector
        default = get_default_selector()
        default.register_mcp_tools(["test_tool"])

        # Should be available through convenience function
        tools = get_tools_for_mode("plan")
        assert "test_tool" in tools


class TestToolSelectorIntegration:
    """Integration tests for ToolSelector."""

    def test_workflow_register_and_use(self):
        """Test complete workflow of registering and using tools."""
        selector = ToolSelector()

        # Register MCP tools
        selector.register_mcp_tools(["db_query", "api_call"])

        # Get tools for plan mode
        plan_tools = selector.get_tools_for_mode("plan")
        assert "db_query" in plan_tools
        assert "read" in plan_tools
        assert "write" not in plan_tools

        # Get tools for build mode
        build_tools = selector.get_tools_for_mode("build")
        assert "db_query" in build_tools
        assert "write" in build_tools

        # Unregister one tool
        selector.unregister_mcp_tools(["db_query"])
        plan_tools = selector.get_tools_for_mode("plan")
        assert "db_query" not in plan_tools
        assert "api_call" in plan_tools

    def test_multiple_selectors_independent(self):
        """Test that multiple selectors are independent."""
        selector1 = ToolSelector()
        selector2 = ToolSelector()

        selector1.register_mcp_tools(["tool1"])
        selector2.register_mcp_tools(["tool2"])

        assert "tool1" in selector1.mcp_tools
        assert "tool1" not in selector2.mcp_tools
        assert "tool2" in selector2.mcp_tools
        assert "tool2" not in selector1.mcp_tools
