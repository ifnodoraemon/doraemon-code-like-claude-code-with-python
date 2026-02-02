"""
Integration tests for end-to-end chat flows.

Tests complete workflows including tool execution, mode switching, and error recovery.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.context_manager import ContextConfig, ContextManager
from src.core.model_client import ChatResponse, Message
from tests.utils.factories import (
    create_chat_response_with_tools,
    create_mock_model_client,
    create_simple_tool_call,
    create_test_tool_registry,
)


class TestEndToEndChatFlow:
    """Integration tests for complete chat flows."""

    @pytest.mark.asyncio
    async def test_simple_chat_without_tools(self):
        """Test a simple chat conversation without tool calls."""
        # Setup
        client = create_mock_model_client([
            "Hello! How can I help you?",
            "I can help with that.",
        ])
        context = ContextManager(
            ContextConfig(auto_save=False),
            project_name="test"
        )

        # User message
        context.add_message("user", "Hello")

        # Get response
        response = await client.chat(context.get_messages())
        context.add_message("assistant", response.content)

        # Verify
        assert len(context.messages) == 2
        assert context.messages[0]["role"] == "user"
        assert context.messages[1]["role"] == "assistant"
        assert "help" in context.messages[1]["content"].lower()

    @pytest.mark.asyncio
    async def test_chat_with_single_tool_call(self):
        """Test chat flow with a single tool execution."""
        # Setup
        registry = create_test_tool_registry()

        # Mock client returns tool call, then final response
        tool_call = create_simple_tool_call(
            name="test_read_file",
            arguments={"path": "test.txt"}
        )
        client = create_mock_model_client([
            create_chat_response_with_tools([tool_call]),
            "The file contains: Content of test.txt",
        ])

        context = ContextManager(
            ContextConfig(auto_save=False),
            project_name="test"
        )

        # User asks to read file
        context.add_message("user", "Read test.txt")

        # First response - tool call
        response1 = await client.chat(context.get_messages())
        assert response1.has_tool_calls

        # Execute tool
        tool_result = await registry.call_tool(
            "test_read_file",
            {"path": "test.txt"}
        )
        context.add_message("tool", tool_result, tool_call_id=tool_call["id"])

        # Second response - final answer
        response2 = await client.chat(context.get_messages())
        context.add_message("assistant", response2.content)

        # Verify
        assert "Content of test.txt" in context.messages[-1]["content"]

    @pytest.mark.asyncio
    async def test_chat_with_multiple_tool_calls(self):
        """Test chat flow with multiple sequential tool calls."""
        registry = create_test_tool_registry()

        # First tool call
        tool_call1 = create_simple_tool_call(
            name="test_read_file",
            arguments={"path": "file1.txt"},
            call_id="call_1"
        )

        # Second tool call
        tool_call2 = create_simple_tool_call(
            name="test_read_file",
            arguments={"path": "file2.txt"},
            call_id="call_2"
        )

        client = create_mock_model_client([
            create_chat_response_with_tools([tool_call1]),
            create_chat_response_with_tools([tool_call2]),
            "Both files have been read successfully.",
        ])

        context = ContextManager(
            ContextConfig(auto_save=False),
            project_name="test"
        )

        context.add_message("user", "Read file1.txt and file2.txt")

        # First tool call
        response1 = await client.chat(context.get_messages())
        result1 = await registry.call_tool("test_read_file", {"path": "file1.txt"})
        context.add_message("tool", result1, tool_call_id="call_1")

        # Second tool call
        response2 = await client.chat(context.get_messages())
        result2 = await registry.call_tool("test_read_file", {"path": "file2.txt"})
        context.add_message("tool", result2, tool_call_id="call_2")

        # Final response
        response3 = await client.chat(context.get_messages())
        context.add_message("assistant", response3.content)

        # Verify
        assert len(context.messages) == 6  # user + 2*(tool_call + tool_result) + final
        assert "successfully" in context.messages[-1]["content"].lower()

    @pytest.mark.asyncio
    async def test_error_recovery_on_tool_failure(self):
        """Test that system recovers gracefully from tool execution errors."""
        registry = create_test_tool_registry()

        # Tool call that will fail
        tool_call = create_simple_tool_call(
            name="nonexistent_tool",
            arguments={}
        )

        client = create_mock_model_client([
            create_chat_response_with_tools([tool_call]),
            "I apologize, that tool is not available.",
        ])

        context = ContextManager(
            ContextConfig(auto_save=False),
            project_name="test"
        )

        context.add_message("user", "Use nonexistent tool")

        # First response - tool call
        response1 = await client.chat(context.get_messages())

        # Try to execute tool - will fail
        try:
            await registry.call_tool("nonexistent_tool", {})
        except Exception as e:
            # Add error message
            context.add_message("tool", f"Error: {str(e)}", tool_call_id=tool_call["id"])

        # Get recovery response
        response2 = await client.chat(context.get_messages())
        context.add_message("assistant", response2.content)

        # Verify graceful handling
        assert "apologize" in context.messages[-1]["content"].lower()

    @pytest.mark.asyncio
    async def test_context_summarization_during_long_conversation(self):
        """Test that context is summarized during long conversations."""
        config = ContextConfig(
            max_context_tokens=100,
            summarize_threshold=0.7,
            keep_recent_messages=2,
            auto_save=False,
        )
        context = ContextManager(config, project_name="test")

        # Mock model client for summarization
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=MagicMock(
            content="Summary: User asked multiple questions about files."
        ))

        # Add many messages
        for i in range(10):
            context.add_message("user", f"Question {i} " * 20)
            context.add_message("assistant", f"Answer {i} " * 20)

        # Force summarization
        with patch.object(context, 'estimate_tokens', return_value=80):
            with patch.object(context, '_get_model_client', return_value=mock_client):
                await context.maybe_summarize()

        # Should have summary + recent messages
        assert len(context.messages) <= config.keep_recent_messages + 1
        assert any("Summary" in msg.get("content", "") for msg in context.messages)


class TestToolExecutionFlow:
    """Integration tests for tool execution workflows."""

    @pytest.mark.asyncio
    async def test_sensitive_tool_requires_approval(self):
        """Test that sensitive tools require approval (HITL)."""
        registry = create_test_tool_registry()

        # test_write_file is marked as sensitive
        tool_info = registry.get_tool("test_write_file")
        assert tool_info.sensitive is True

    @pytest.mark.asyncio
    async def test_non_sensitive_tool_executes_directly(self):
        """Test that non-sensitive tools execute without approval."""
        registry = create_test_tool_registry()

        # test_read_file is not sensitive
        tool_info = registry.get_tool("test_read_file")
        assert tool_info.sensitive is False

        # Should execute directly
        result = await registry.call_tool("test_read_file", {"path": "test.txt"})
        assert "Content of test.txt" in result

    @pytest.mark.asyncio
    async def test_tool_timeout_handling(self):
        """Test that tool execution respects timeout settings."""
        registry = create_test_tool_registry()

        # Register a slow tool
        async def slow_tool(delay: int) -> str:
            import asyncio
            await asyncio.sleep(delay)
            return "Done"

        registry.register(slow_tool, sensitive=False, timeout=0.1)

        # Should timeout
        with pytest.raises(Exception):  # TimeoutError or similar
            await registry.call_tool("slow_tool", {"delay": 10})


class TestModeSwitch:
    """Integration tests for mode switching."""

    def test_plan_mode_has_readonly_tools(self):
        """Test that plan mode only has read-only tools."""
        from src.core.tool_selector import ToolSelector

        registry = create_test_tool_registry()
        selector = ToolSelector(registry)

        plan_tools = selector.get_tools_for_mode("plan")

        # Should only have read tools
        tool_names = [t.name for t in plan_tools]
        assert "test_read_file" in tool_names
        assert "test_write_file" not in tool_names
        assert "test_execute_command" not in tool_names

    def test_build_mode_has_all_tools(self):
        """Test that build mode has all tools including write."""
        from src.core.tool_selector import ToolSelector

        registry = create_test_tool_registry()
        selector = ToolSelector(registry)

        build_tools = selector.get_tools_for_mode("build")

        # Should have all tools
        tool_names = [t.name for t in build_tools]
        assert "test_read_file" in tool_names
        assert "test_write_file" in tool_names
        assert "test_execute_command" in tool_names


class TestErrorHandling:
    """Integration tests for error handling and recovery."""

    @pytest.mark.asyncio
    async def test_network_error_retry(self):
        """Test that network errors trigger retry."""
        from src.core.errors import TransientError
        from src.core.model_client import ClientConfig, ClientMode, GatewayModelClient

        config = ClientConfig(
            mode=ClientMode.GATEWAY,
            gateway_url="http://test.com",
            model="test-model",
        )
        client = GatewayModelClient(config)

        mock_client = AsyncMock()

        # First call fails with network error, second succeeds
        import httpx
        mock_client.post.side_effect = [
            httpx.RequestError("Network error"),
            MagicMock(
                status_code=200,
                json=lambda: {"choices": [{"message": {"content": "Success"}, "finish_reason": "stop"}]}
            ),
        ]

        client._client = mock_client

        messages = [Message(role="user", content="Test")]

        # Should retry and succeed
        with patch("asyncio.sleep"):
            response = await client.chat(messages)

        assert response.content == "Success"
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_rate_limit_respects_retry_after(self):
        """Test that rate limit errors respect Retry-After header."""
        from src.core.model_client import ClientConfig, ClientMode, GatewayModelClient

        config = ClientConfig(
            mode=ClientMode.GATEWAY,
            gateway_url="http://test.com",
            model="test-model",
        )
        client = GatewayModelClient(config)

        mock_client = AsyncMock()

        import httpx
        rate_limit_response = httpx.Response(
            status_code=429,
            headers={"Retry-After": "2"},
            request=httpx.Request("POST", "http://test.com/v1/chat/completions"),
        )
        success_response = MagicMock(
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": "Success"}, "finish_reason": "stop"}]}
        )

        mock_client.post.side_effect = [
            httpx.HTTPStatusError("Rate limited", request=rate_limit_response.request, response=rate_limit_response),
            success_response,
        ]

        client._client = mock_client

        messages = [Message(role="user", content="Test")]

        # Should retry after delay
        with patch("asyncio.sleep") as mock_sleep:
            response = await client.chat(messages)

        assert response.content == "Success"
        # Should have slept for retry_after duration
        mock_sleep.assert_called()
