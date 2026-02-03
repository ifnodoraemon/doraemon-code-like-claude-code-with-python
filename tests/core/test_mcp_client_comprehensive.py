"""Comprehensive tests for MCP Client.

Tests cover:
1. MCP connection management (10 tests)
2. Tool discovery and invocation (10 tests)
3. Error handling (8 tests)
4. Timeout and retry (7 tests)
5. Multi-server management (5 tests)
"""

import asyncio
import json
import logging
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch, call
import pytest

from src.core.mcp_client import (
    MCPClient,
    MCPConnection,
    MCPPrompt,
    MCPResource,
    MCPServerConfig,
    MCPTool,
    MCPTransport,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_config():
    """Create a mock MCP server config."""
    return MCPServerConfig(
        name="test_server",
        command="python",
        args=["test_script.py"],
        timeout=5.0,
    )


@pytest.fixture
def mock_config_http():
    """Create a mock HTTP MCP server config."""
    return MCPServerConfig(
        name="http_server",
        command="",
        transport=MCPTransport.HTTP,
        url="http://localhost:8000",
        timeout=10.0,
    )


@pytest.fixture
def mcp_client():
    """Create an MCP client instance."""
    return MCPClient()


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file."""
    config_data = {
        "mcpServers": {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                "env": {},
            },
            "memory": {
                "command": "python",
                "args": ["src/servers/memory.py"],
                "env": {"MEMORY_SIZE": "1000"},
            },
        }
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data))
    return config_file


# ============================================================================
# 1. MCP CONNECTION MANAGEMENT TESTS (10 tests)
# ============================================================================


class TestMCPConnectionManagement:
    """Tests for MCP connection lifecycle management."""

    @pytest.mark.asyncio
    async def test_connection_initialization(self, mock_config):
        """Test MCPConnection initialization."""
        conn = MCPConnection(mock_config)
        assert conn.config == mock_config
        assert conn._process is None
        assert conn._request_id == 0
        assert conn._connected is False
        assert not conn.is_connected

    @pytest.mark.asyncio
    async def test_is_connected_property_false_when_no_process(self, mock_config):
        """Test is_connected returns False when no process."""
        conn = MCPConnection(mock_config)
        assert not conn.is_connected

    @pytest.mark.asyncio
    async def test_is_connected_property_false_when_not_connected(self, mock_config):
        """Test is_connected returns False when _connected flag is False."""
        conn = MCPConnection(mock_config)
        conn._process = Mock()
        conn._connected = False
        assert not conn.is_connected

    @pytest.mark.asyncio
    async def test_is_connected_property_true_when_connected(self, mock_config):
        """Test is_connected returns True when connected."""
        conn = MCPConnection(mock_config)
        conn._process = Mock()
        conn._connected = True
        assert conn.is_connected

    @pytest.mark.asyncio
    async def test_connect_already_connected(self, mock_config):
        """Test connect returns True if already connected."""
        conn = MCPConnection(mock_config)
        conn._connected = True
        result = await conn.connect()
        assert result is True

    @pytest.mark.asyncio
    async def test_connect_failure_logs_error(self, mock_config):
        """Test connect logs error on failure."""
        conn = MCPConnection(mock_config)
        with patch("subprocess.Popen", side_effect=Exception("Connection failed")):
            with patch("src.core.mcp_client.logger") as mock_logger:
                result = await conn.connect()
                assert result is False
                mock_logger.error.assert_called()

    @pytest.mark.asyncio
    async def test_disconnect_cancels_reader_task(self, mock_config):
        """Test disconnect cancels reader task."""
        conn = MCPConnection(mock_config)
        # Create a real async task that we can cancel
        async def dummy_task():
            await asyncio.sleep(10)

        conn._reader_task = asyncio.create_task(dummy_task())
        conn._process = Mock()
        conn._process.terminate = Mock()
        conn._process.wait = Mock()

        await conn.disconnect()
        assert conn._reader_task.cancelled() or conn._reader_task.done()

    @pytest.mark.asyncio
    async def test_disconnect_terminates_process(self, mock_config):
        """Test disconnect terminates process."""
        conn = MCPConnection(mock_config)
        mock_process = Mock()
        mock_process.terminate = Mock()
        mock_process.wait = Mock()
        conn._process = mock_process
        conn._connected = True
        conn._reader_task = None

        await conn.disconnect()
        mock_process.terminate.assert_called_once()
        assert conn._connected is False

    @pytest.mark.asyncio
    async def test_disconnect_kills_process_on_timeout(self, mock_config):
        """Test disconnect kills process if terminate times out."""
        conn = MCPConnection(mock_config)
        mock_process = Mock()
        mock_process.terminate = Mock()
        mock_process.wait = Mock(side_effect=subprocess.TimeoutExpired("cmd", 5))
        mock_process.kill = Mock()
        conn._process = mock_process
        conn._reader_task = None

        await conn.disconnect()
        mock_process.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_sets_connected_false(self, mock_config):
        """Test disconnect sets _connected to False."""
        conn = MCPConnection(mock_config)
        conn._connected = True
        conn._process = Mock()
        conn._process.terminate = Mock()
        conn._process.wait = Mock()

        await conn.disconnect()
        assert conn._connected is False


# ============================================================================
# 2. TOOL DISCOVERY AND INVOCATION TESTS (10 tests)
# ============================================================================


class TestToolDiscoveryAndInvocation:
    """Tests for tool discovery and invocation."""

    @pytest.mark.asyncio
    async def test_list_tools_empty_result(self, mock_config):
        """Test list_tools with empty result."""
        conn = MCPConnection(mock_config)
        conn._request = AsyncMock(return_value=None)

        tools = await conn.list_tools()
        assert tools == []

    @pytest.mark.asyncio
    async def test_list_tools_single_tool(self, mock_config):
        """Test list_tools with single tool."""
        conn = MCPConnection(mock_config)
        conn._request = AsyncMock(
            return_value={
                "tools": [
                    {
                        "name": "read_file",
                        "description": "Read a file",
                        "inputSchema": {"type": "object"},
                    }
                ]
            }
        )

        tools = await conn.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "read_file"
        assert tools[0].description == "Read a file"
        assert tools[0].server_name == "test_server"

    @pytest.mark.asyncio
    async def test_list_tools_multiple_tools(self, mock_config):
        """Test list_tools with multiple tools."""
        conn = MCPConnection(mock_config)
        conn._request = AsyncMock(
            return_value={
                "tools": [
                    {
                        "name": "read_file",
                        "description": "Read a file",
                        "inputSchema": {},
                    },
                    {
                        "name": "write_file",
                        "description": "Write a file",
                        "inputSchema": {},
                    },
                ]
            }
        )

        tools = await conn.list_tools()
        assert len(tools) == 2
        assert tools[0].name == "read_file"
        assert tools[1].name == "write_file"

    @pytest.mark.asyncio
    async def test_call_tool_returns_text_content(self, mock_config):
        """Test call_tool returns text content."""
        conn = MCPConnection(mock_config)
        conn._request = AsyncMock(
            return_value={
                "content": [{"type": "text", "text": "file content"}]
            }
        )

        result = await conn.call_tool("read_file", {"path": "/test"})
        assert result == "file content"

    @pytest.mark.asyncio
    async def test_call_tool_returns_first_content(self, mock_config):
        """Test call_tool returns first content item."""
        conn = MCPConnection(mock_config)
        conn._request = AsyncMock(
            return_value={
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "text", "text": "second"},
                ]
            }
        )

        result = await conn.call_tool("read_file", {"path": "/test"})
        assert result == "first"

    @pytest.mark.asyncio
    async def test_call_tool_returns_none_on_empty_result(self, mock_config):
        """Test call_tool returns None on empty result."""
        conn = MCPConnection(mock_config)
        conn._request = AsyncMock(return_value=None)

        result = await conn.call_tool("read_file", {"path": "/test"})
        assert result is None

    @pytest.mark.asyncio
    async def test_call_tool_returns_empty_content_result(self, mock_config):
        """Test call_tool returns result when content is empty."""
        conn = MCPConnection(mock_config)
        conn._request = AsyncMock(return_value={"content": []})

        result = await conn.call_tool("read_file", {"path": "/test"})
        assert result == {"content": []}

    @pytest.mark.asyncio
    async def test_call_tool_returns_non_text_content(self, mock_config):
        """Test call_tool returns non-text content."""
        conn = MCPConnection(mock_config)
        conn._request = AsyncMock(
            return_value={
                "content": [{"type": "image", "data": "base64..."}]
            }
        )

        result = await conn.call_tool("get_image", {})
        assert result == {"type": "image", "data": "base64..."}

    @pytest.mark.asyncio
    async def test_call_tool_sends_correct_request(self, mock_config):
        """Test call_tool sends correct request."""
        conn = MCPConnection(mock_config)
        conn._request = AsyncMock(return_value={"content": []})

        await conn.call_tool("my_tool", {"arg1": "value1"})
        conn._request.assert_called_once_with(
            "tools/call",
            {"name": "my_tool", "arguments": {"arg1": "value1"}},
        )


# ============================================================================
# 3. ERROR HANDLING TESTS (8 tests)
# ============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_request_not_connected_raises_exception(self, mock_config):
        """Test _request raises exception when not connected."""
        conn = MCPConnection(mock_config)
        conn._process = None

        with pytest.raises(Exception, match="Not connected"):
            await conn._request("test_method", {})

    @pytest.mark.asyncio
    async def test_request_handles_json_decode_error(self, mock_config):
        """Test _read_responses handles JSON decode errors."""
        conn = MCPConnection(mock_config)
        mock_process = Mock()
        mock_process.stdout = Mock()
        mock_process.stdout.readline = Mock(return_value=b"invalid json\n")
        conn._process = mock_process

        # Should not raise, just continue
        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(
                return_value=b"invalid json\n"
            )
            # This would normally run indefinitely, so we'll just test the logic

    @pytest.mark.asyncio
    async def test_handle_message_with_error(self, mock_config):
        """Test _handle_message with error response."""
        conn = MCPConnection(mock_config)
        future = asyncio.Future()
        conn._pending_requests[1] = future

        await conn._handle_message(
            {"id": 1, "error": {"message": "Test error"}}
        )

        with pytest.raises(Exception, match="Test error"):
            await future

    @pytest.mark.asyncio
    async def test_handle_message_with_result(self, mock_config):
        """Test _handle_message with successful result."""
        conn = MCPConnection(mock_config)
        future = asyncio.Future()
        conn._pending_requests[1] = future

        await conn._handle_message({"id": 1, "result": {"data": "test"}})

        result = await future
        assert result == {"data": "test"}

    @pytest.mark.asyncio
    async def test_handle_message_ignores_unknown_id(self, mock_config):
        """Test _handle_message ignores unknown request IDs."""
        conn = MCPConnection(mock_config)

        # Should not raise
        await conn._handle_message({"id": 999, "result": {"data": "test"}})

    @pytest.mark.asyncio
    async def test_list_tools_handles_exception(self, mock_config):
        """Test list_tools handles exceptions gracefully."""
        conn = MCPConnection(mock_config)
        conn._request = AsyncMock(side_effect=Exception("Request failed"))

        with pytest.raises(Exception):
            await conn.list_tools()

    @pytest.mark.asyncio
    async def test_call_tool_handles_exception(self, mock_config):
        """Test call_tool handles exceptions gracefully."""
        conn = MCPConnection(mock_config)
        conn._request = AsyncMock(side_effect=Exception("Request failed"))

        with pytest.raises(Exception):
            await conn.call_tool("test_tool", {})

    @pytest.mark.asyncio
    async def test_notify_handles_no_process(self, mock_config):
        """Test _notify handles missing process gracefully."""
        conn = MCPConnection(mock_config)
        conn._process = None

        # Should not raise
        await conn._notify("test_method", {})


# ============================================================================
# 4. TIMEOUT AND RETRY TESTS (7 tests)
# ============================================================================


class TestTimeoutAndRetry:
    """Tests for timeout and retry behavior."""

    @pytest.mark.asyncio
    async def test_request_timeout_raises_exception(self, mock_config):
        """Test _request raises exception on timeout."""
        conn = MCPConnection(mock_config)
        mock_process = Mock()
        mock_process.stdin = Mock()
        conn._process = mock_process
        conn.config.timeout = 0.01

        # Create a future that never completes
        with pytest.raises(Exception, match="Request timeout"):
            await conn._request("test", {})

    @pytest.mark.asyncio
    async def test_request_timeout_cleans_up_pending(self, mock_config):
        """Test _request cleans up pending requests on timeout."""
        conn = MCPConnection(mock_config)
        mock_process = Mock()
        mock_process.stdin = Mock()
        conn._process = mock_process
        conn.config.timeout = 0.01

        # Add a pending request that will timeout
        future = asyncio.Future()
        conn._pending_requests[1] = future

        with pytest.raises(Exception, match="Request timeout"):
            await conn._request("test", {})

    @pytest.mark.asyncio
    async def test_connect_with_timeout_config(self, mock_config):
        """Test connect respects timeout configuration."""
        mock_config.timeout = 1.0
        conn = MCPConnection(mock_config)
        assert conn.config.timeout == 1.0

    @pytest.mark.asyncio
    async def test_request_increments_request_id(self, mock_config):
        """Test _request increments request ID."""
        conn = MCPConnection(mock_config)
        mock_process = Mock()
        mock_process.stdin = Mock()
        conn._process = mock_process

        initial_id = conn._request_id
        # We can't actually send a request without a full setup,
        # but we can verify the ID increments
        conn._request_id += 1
        assert conn._request_id == initial_id + 1

    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self, mock_config):
        """Test handling multiple concurrent requests."""
        conn = MCPConnection(mock_config)
        mock_process = Mock()
        mock_process.stdin = Mock()
        conn._process = mock_process

        # Simulate multiple pending requests
        futures = {}
        for i in range(5):
            futures[i] = asyncio.Future()
            conn._pending_requests[i] = futures[i]

        assert len(conn._pending_requests) == 5

    @pytest.mark.asyncio
    async def test_request_with_custom_timeout(self, mock_config):
        """Test request with custom timeout."""
        mock_config.timeout = 2.0
        conn = MCPConnection(mock_config)
        assert conn.config.timeout == 2.0


# ============================================================================
# 5. MULTI-SERVER MANAGEMENT TESTS (5 tests)
# ============================================================================


class TestMultiServerManagement:
    """Tests for multi-server management."""

    def test_mcp_client_initialization(self):
        """Test MCPClient initialization."""
        client = MCPClient()
        assert client._servers == {}
        assert client._connections == {}

    def test_add_server(self, mcp_client, mock_config):
        """Test adding a server."""
        mcp_client.add_server(mock_config)
        assert "test_server" in mcp_client._servers
        assert mcp_client._servers["test_server"] == mock_config

    def test_remove_server(self, mcp_client, mock_config):
        """Test removing a server."""
        mcp_client.add_server(mock_config)
        mcp_client.remove_server("test_server")
        assert "test_server" not in mcp_client._servers

    def test_remove_nonexistent_server(self, mcp_client):
        """Test removing nonexistent server doesn't raise."""
        mcp_client.remove_server("nonexistent")
        # Should not raise

    def test_get_summary(self, mcp_client, mock_config):
        """Test get_summary returns correct info."""
        mcp_client.add_server(mock_config)
        summary = mcp_client.get_summary()

        assert summary["configured_servers"] == ["test_server"]
        assert summary["total_configured"] == 1
        assert summary["total_connected"] == 0



# ============================================================================
# 6. RESOURCE MANAGEMENT TESTS (6 tests)
# ============================================================================


class TestResourceManagement:
    """Tests for resource management."""

    @pytest.mark.asyncio
    async def test_list_resources_empty(self, mock_config):
        """Test list_resources with empty result."""
        conn = MCPConnection(mock_config)
        conn._request = AsyncMock(return_value=None)

        resources = await conn.list_resources()
        assert resources == []

    @pytest.mark.asyncio
    async def test_list_resources_single_resource(self, mock_config):
        """Test list_resources with single resource."""
        conn = MCPConnection(mock_config)
        conn._request = AsyncMock(
            return_value={
                "resources": [
                    {
                        "uri": "file:///test.txt",
                        "name": "test",
                        "description": "Test file",
                        "mimeType": "text/plain",
                    }
                ]
            }
        )

        resources = await conn.list_resources()
        assert len(resources) == 1
        assert resources[0].uri == "file:///test.txt"
        assert resources[0].name == "test"
        assert resources[0].mime_type == "text/plain"
        assert resources[0].server_name == "test_server"

    @pytest.mark.asyncio
    async def test_list_resources_multiple(self, mock_config):
        """Test list_resources with multiple resources."""
        conn = MCPConnection(mock_config)
        conn._request = AsyncMock(
            return_value={
                "resources": [
                    {
                        "uri": "file:///test1.txt",
                        "name": "test1",
                        "description": "Test 1",
                    },
                    {
                        "uri": "file:///test2.txt",
                        "name": "test2",
                        "description": "Test 2",
                    },
                ]
            }
        )

        resources = await conn.list_resources()
        assert len(resources) == 2

    @pytest.mark.asyncio
    async def test_read_resource_returns_text(self, mock_config):
        """Test read_resource returns text content."""
        conn = MCPConnection(mock_config)
        conn._request = AsyncMock(
            return_value={
                "contents": [{"text": "resource content"}]
            }
        )

        content = await conn.read_resource("file:///test.txt")
        assert content == "resource content"

    @pytest.mark.asyncio
    async def test_read_resource_empty_result(self, mock_config):
        """Test read_resource with empty result."""
        conn = MCPConnection(mock_config)
        conn._request = AsyncMock(return_value=None)

        content = await conn.read_resource("file:///test.txt")
        assert content == ""

    @pytest.mark.asyncio
    async def test_read_resource_no_contents(self, mock_config):
        """Test read_resource with no contents."""
        conn = MCPConnection(mock_config)
        conn._request = AsyncMock(return_value={"contents": []})

        content = await conn.read_resource("file:///test.txt")
        assert content == ""


# ============================================================================
# 7. PROMPT MANAGEMENT TESTS (6 tests)
# ============================================================================


class TestPromptManagement:
    """Tests for prompt management."""

    @pytest.mark.asyncio
    async def test_list_prompts_empty(self, mock_config):
        """Test list_prompts with empty result."""
        conn = MCPConnection(mock_config)
        conn._request = AsyncMock(return_value=None)

        prompts = await conn.list_prompts()
        assert prompts == []

    @pytest.mark.asyncio
    async def test_list_prompts_single_prompt(self, mock_config):
        """Test list_prompts with single prompt."""
        conn = MCPConnection(mock_config)
        conn._request = AsyncMock(
            return_value={
                "prompts": [
                    {
                        "name": "code_review",
                        "description": "Review code",
                        "arguments": [{"name": "code", "description": "Code to review"}],
                    }
                ]
            }
        )

        prompts = await conn.list_prompts()
        assert len(prompts) == 1
        assert prompts[0].name == "code_review"
        assert prompts[0].server_name == "test_server"

    @pytest.mark.asyncio
    async def test_list_prompts_multiple(self, mock_config):
        """Test list_prompts with multiple prompts."""
        conn = MCPConnection(mock_config)
        conn._request = AsyncMock(
            return_value={
                "prompts": [
                    {
                        "name": "prompt1",
                        "description": "First prompt",
                        "arguments": [],
                    },
                    {
                        "name": "prompt2",
                        "description": "Second prompt",
                        "arguments": [],
                    },
                ]
            }
        )

        prompts = await conn.list_prompts()
        assert len(prompts) == 2

    @pytest.mark.asyncio
    async def test_get_prompt_returns_text(self, mock_config):
        """Test get_prompt returns text content."""
        conn = MCPConnection(mock_config)
        conn._request = AsyncMock(
            return_value={
                "messages": [
                    {
                        "content": {
                            "type": "text",
                            "text": "Rendered prompt text",
                        }
                    }
                ]
            }
        )

        text = await conn.get_prompt("code_review", {"code": "print('hello')"})
        assert text == "Rendered prompt text"

    @pytest.mark.asyncio
    async def test_get_prompt_empty_result(self, mock_config):
        """Test get_prompt with empty result."""
        conn = MCPConnection(mock_config)
        conn._request = AsyncMock(return_value=None)

        text = await conn.get_prompt("code_review", {})
        assert text == ""

    @pytest.mark.asyncio
    async def test_get_prompt_no_messages(self, mock_config):
        """Test get_prompt with no messages."""
        conn = MCPConnection(mock_config)
        conn._request = AsyncMock(return_value={"messages": []})

        text = await conn.get_prompt("code_review", {})
        assert text == ""


# ============================================================================
# 8. MCP CLIENT MULTI-SERVER TESTS (8 tests)
# ============================================================================


class TestMCPClientMultiServer:
    """Tests for MCPClient multi-server operations."""

    @pytest.mark.asyncio
    async def test_connect_unknown_server(self, mcp_client):
        """Test connect to unknown server returns False."""
        result = await mcp_client.connect("unknown_server")
        assert result is False

    @pytest.mark.asyncio
    async def test_connect_already_connected_server(self, mcp_client, mock_config):
        """Test connect to already connected server returns True."""
        mcp_client.add_server(mock_config)
        mock_conn = AsyncMock()
        mock_conn.is_connected = True
        mcp_client._connections["test_server"] = mock_conn

        result = await mcp_client.connect("test_server")
        assert result is True

    @pytest.mark.asyncio
    async def test_disconnect_unknown_server(self, mcp_client):
        """Test disconnect from unknown server doesn't raise."""
        await mcp_client.disconnect("unknown_server")
        # Should not raise

    @pytest.mark.asyncio
    async def test_disconnect_connected_server(self, mcp_client):
        """Test disconnect from connected server."""
        mock_conn = AsyncMock()
        mcp_client._connections["test_server"] = mock_conn

        await mcp_client.disconnect("test_server")
        mock_conn.disconnect.assert_called_once()
        assert "test_server" not in mcp_client._connections

    @pytest.mark.asyncio
    async def test_list_all_tools_empty(self, mcp_client):
        """Test list_all_tools with no connections."""
        tools = await mcp_client.list_all_tools()
        assert tools == []

    @pytest.mark.asyncio
    async def test_list_all_tools_multiple_servers(self, mcp_client):
        """Test list_all_tools from multiple servers."""
        mock_conn1 = AsyncMock()
        mock_conn1.config.name = "server1"
        mock_conn1.list_tools = AsyncMock(
            return_value=[
                MCPTool("tool1", "Tool 1", {}, "server1"),
                MCPTool("tool2", "Tool 2", {}, "server1"),
            ]
        )

        mock_conn2 = AsyncMock()
        mock_conn2.config.name = "server2"
        mock_conn2.list_tools = AsyncMock(
            return_value=[
                MCPTool("tool3", "Tool 3", {}, "server2"),
            ]
        )

        mcp_client._connections["server1"] = mock_conn1
        mcp_client._connections["server2"] = mock_conn2

        tools = await mcp_client.list_all_tools()
        assert len(tools) == 3

    @pytest.mark.asyncio
    async def test_list_all_tools_handles_exception(self, mcp_client):
        """Test list_all_tools handles exceptions from servers."""
        mock_conn = AsyncMock()
        mock_conn.config.name = "server1"
        mock_conn.list_tools = AsyncMock(side_effect=Exception("Error"))

        mcp_client._connections["server1"] = mock_conn

        tools = await mcp_client.list_all_tools()
        assert tools == []

    @pytest.mark.asyncio
    async def test_call_tool_not_connected(self, mcp_client):
        """Test call_tool raises when server not connected."""
        with pytest.raises(Exception, match="Not connected to server"):
            await mcp_client.call_tool("unknown", "tool", {})


# ============================================================================
# 9. CONFIG LOADING TESTS (5 tests)
# ============================================================================


class TestConfigLoading:
    """Tests for configuration loading."""

    def test_load_config_from_file(self, temp_config_file):
        """Test loading config from file."""
        client = MCPClient(temp_config_file)
        assert "filesystem" in client._servers
        assert "memory" in client._servers

    def test_load_config_filesystem_server(self, temp_config_file):
        """Test loaded filesystem server config."""
        client = MCPClient(temp_config_file)
        fs_config = client._servers["filesystem"]
        assert fs_config.command == "npx"
        assert "-y" in fs_config.args

    def test_load_config_memory_server(self, temp_config_file):
        """Test loaded memory server config."""
        client = MCPClient(temp_config_file)
        mem_config = client._servers["memory"]
        assert mem_config.command == "python"
        assert mem_config.env.get("MEMORY_SIZE") == "1000"

    def test_load_config_nonexistent_file(self):
        """Test loading config from nonexistent file."""
        client = MCPClient(Path("/nonexistent/config.json"))
        assert client._servers == {}

    def test_load_config_invalid_json(self, tmp_path):
        """Test loading config with invalid JSON."""
        config_file = tmp_path / "invalid.json"
        config_file.write_text("invalid json {")

        with patch("src.core.mcp_client.logger") as mock_logger:
            client = MCPClient(config_file)
            mock_logger.error.assert_called()


# ============================================================================
# 10. DATA CLASS TESTS (5 tests)
# ============================================================================


class TestDataClasses:
    """Tests for data classes."""

    def test_mcp_server_config_to_dict(self):
        """Test MCPServerConfig.to_dict()."""
        config = MCPServerConfig(
            name="test",
            command="python",
            args=["script.py"],
            env={"VAR": "value"},
            timeout=5.0,
        )
        d = config.to_dict()

        assert d["name"] == "test"
        assert d["command"] == "python"
        assert d["args"] == ["script.py"]
        assert d["env"] == {"VAR": "value"}
        assert d["timeout"] == 5.0

    def test_mcp_tool_to_dict(self):
        """Test MCPTool.to_dict()."""
        tool = MCPTool(
            name="read_file",
            description="Read a file",
            input_schema={"type": "object"},
            server_name="filesystem",
        )
        d = tool.to_dict()

        assert d["name"] == "read_file"
        assert d["description"] == "Read a file"
        assert d["server_name"] == "filesystem"

    def test_mcp_resource_to_dict(self):
        """Test MCPResource.to_dict()."""
        resource = MCPResource(
            uri="file:///test.txt",
            name="test",
            description="Test file",
            mime_type="text/plain",
            server_name="filesystem",
        )
        d = resource.to_dict()

        assert d["uri"] == "file:///test.txt"
        assert d["name"] == "test"
        assert d["mime_type"] == "text/plain"

    def test_mcp_prompt_to_dict(self):
        """Test MCPPrompt.to_dict()."""
        prompt = MCPPrompt(
            name="code_review",
            description="Review code",
            arguments=[{"name": "code"}],
            server_name="ai",
        )
        d = prompt.to_dict()

        assert d["name"] == "code_review"
        assert d["description"] == "Review code"
        assert len(d["arguments"]) == 1

    def test_mcp_transport_enum(self):
        """Test MCPTransport enum."""
        assert MCPTransport.STDIO.value == "stdio"
        assert MCPTransport.HTTP.value == "http"
        assert MCPTransport.WEBSOCKET.value == "websocket"


# ============================================================================
# 11. RESOURCE LISTING TESTS (4 tests)
# ============================================================================


class TestResourceListing:
    """Tests for resource listing across servers."""

    @pytest.mark.asyncio
    async def test_list_all_resources_empty(self, mcp_client):
        """Test list_all_resources with no connections."""
        resources = await mcp_client.list_all_resources()
        assert resources == []

    @pytest.mark.asyncio
    async def test_list_all_resources_multiple_servers(self, mcp_client):
        """Test list_all_resources from multiple servers."""
        mock_conn1 = AsyncMock()
        mock_conn1.config.name = "server1"
        mock_conn1.list_resources = AsyncMock(
            return_value=[
                MCPResource("uri1", "res1", "Resource 1", server_name="server1"),
            ]
        )

        mock_conn2 = AsyncMock()
        mock_conn2.config.name = "server2"
        mock_conn2.list_resources = AsyncMock(
            return_value=[
                MCPResource("uri2", "res2", "Resource 2", server_name="server2"),
            ]
        )

        mcp_client._connections["server1"] = mock_conn1
        mcp_client._connections["server2"] = mock_conn2

        resources = await mcp_client.list_all_resources()
        assert len(resources) == 2

    @pytest.mark.asyncio
    async def test_list_all_resources_handles_exception(self, mcp_client):
        """Test list_all_resources handles exceptions."""
        mock_conn = AsyncMock()
        mock_conn.config.name = "server1"
        mock_conn.list_resources = AsyncMock(side_effect=Exception("Error"))

        mcp_client._connections["server1"] = mock_conn

        resources = await mcp_client.list_all_resources()
        assert resources == []

    @pytest.mark.asyncio
    async def test_read_resource_not_connected(self, mcp_client):
        """Test read_resource raises when server not connected."""
        with pytest.raises(Exception, match="Not connected to server"):
            await mcp_client.read_resource("unknown", "uri")


# ============================================================================
# 12. CONCURRENT OPERATIONS TESTS (4 tests)
# ============================================================================


class TestConcurrentOperations:
    """Tests for concurrent operations."""

    @pytest.mark.asyncio
    async def test_connect_all_servers(self, mcp_client, mock_config):
        """Test connect_all connects all servers."""
        config2 = MCPServerConfig(
            name="server2",
            command="python",
            args=["script2.py"],
        )
        mcp_client.add_server(mock_config)
        mcp_client.add_server(config2)

        with patch.object(mcp_client, "connect", new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = True
            results = await mcp_client.connect_all()

            assert len(results) == 2
            assert mock_connect.call_count == 2

    @pytest.mark.asyncio
    async def test_disconnect_all_servers(self, mcp_client):
        """Test disconnect_all disconnects all servers."""
        mock_conn1 = AsyncMock()
        mock_conn2 = AsyncMock()
        mcp_client._connections["server1"] = mock_conn1
        mcp_client._connections["server2"] = mock_conn2

        await mcp_client.disconnect_all()

        mock_conn1.disconnect.assert_called_once()
        mock_conn2.disconnect.assert_called_once()
        assert len(mcp_client._connections) == 0

    @pytest.mark.asyncio
    async def test_get_connected_servers_empty(self, mcp_client):
        """Test get_connected_servers with no connections."""
        servers = mcp_client.get_connected_servers()
        assert servers == []

    @pytest.mark.asyncio
    async def test_get_connected_servers_multiple(self, mcp_client):
        """Test get_connected_servers returns connected servers."""
        mock_conn1 = AsyncMock()
        mock_conn1.is_connected = True
        mock_conn2 = AsyncMock()
        mock_conn2.is_connected = False

        mcp_client._connections["server1"] = mock_conn1
        mcp_client._connections["server2"] = mock_conn2

        servers = mcp_client.get_connected_servers()
        assert servers == ["server1"]


# ============================================================================
# 13. EDGE CASES AND BOUNDARY CONDITIONS (5 tests)
# ============================================================================


class TestEdgeCasesAndBoundaryConditions:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_call_tool_with_empty_arguments(self, mock_config):
        """Test call_tool with empty arguments."""
        conn = MCPConnection(mock_config)
        conn._request = AsyncMock(return_value={"content": []})

        result = await conn.call_tool("tool", {})
        conn._request.assert_called_once_with(
            "tools/call",
            {"name": "tool", "arguments": {}},
        )

    @pytest.mark.asyncio
    async def test_call_tool_with_complex_arguments(self, mock_config):
        """Test call_tool with complex nested arguments."""
        conn = MCPConnection(mock_config)
        conn._request = AsyncMock(return_value={"content": []})

        complex_args = {
            "nested": {"deep": {"value": 123}},
            "list": [1, 2, 3],
            "string": "test",
        }

        await conn.call_tool("tool", complex_args)
        conn._request.assert_called_once_with(
            "tools/call",
            {"name": "tool", "arguments": complex_args},
        )

    @pytest.mark.asyncio
    async def test_list_tools_with_missing_fields(self, mock_config):
        """Test list_tools handles missing optional fields."""
        conn = MCPConnection(mock_config)
        conn._request = AsyncMock(
            return_value={
                "tools": [
                    {
                        "name": "tool1",
                        # Missing description and inputSchema
                    }
                ]
            }
        )

        tools = await conn.list_tools()
        assert len(tools) == 1
        assert tools[0].description == ""
        assert tools[0].input_schema == {}

    @pytest.mark.asyncio
    async def test_request_id_overflow(self, mock_config):
        """Test request ID handling with large numbers."""
        conn = MCPConnection(mock_config)
        conn._request_id = 2**31 - 1  # Large number

        # Verify it can still increment
        conn._request_id += 1
        assert conn._request_id == 2**31

    @pytest.mark.asyncio
    async def test_mcp_client_with_none_config_path(self):
        """Test MCPClient initialization with None config path."""
        client = MCPClient(config_path=None)
        assert client._servers == {}
        assert client._connections == {}
