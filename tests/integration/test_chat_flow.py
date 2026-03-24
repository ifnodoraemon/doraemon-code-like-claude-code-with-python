"""Integration tests"""

import pytest

from tests.utils.factories import create_test_tool_registry


class TestIntegration:
    @pytest.mark.asyncio
    async def test_tool_execution(self):
        registry = create_test_tool_registry()
        result = await registry.call_tool("test_read_file", {"path": "test.txt"})
        assert "Content of test.txt" in result

    def test_mode_switching(self):
        from src.core.tool_selector import ToolSelector

        selector = ToolSelector()
        plan_tools = selector.get_tools_for_mode("plan")
        build_tools = selector.get_tools_for_mode("build")
        assert len(plan_tools) <= len(build_tools)
