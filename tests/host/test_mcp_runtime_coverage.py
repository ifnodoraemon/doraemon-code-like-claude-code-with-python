"""Targeted coverage for host/mcp_runtime.py uncovered lines: 60,107,114,116,118,168-169,259,263,270,285,287,336,339,433,440,446,452-454,476-477,505,525,539-540."""

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.host.mcp_runtime import (
    MCP_SERVER_FAILURE_COOLDOWN_SECONDS,
    _MCP_SERVER_FAILURE_CACHE,
    _clear_server_failure,
    _get_cached_server_failure,
    _remember_server_failure,
    _server_cache_key,
    RemoteMCPServerConfig,
    StdioMCPClient,
    StreamableHttpMCPClient,
    build_remote_mcp_registry,
)


class TestStreamableHttpMcpValidation:
    def test_invalid_transport(self):
        server = RemoteMCPServerConfig(name="x", transport="grpc", url="http://x")
        with pytest.raises(ValueError, match="Unsupported MCP transport"):
            StreamableHttpMCPClient(server)


class TestStreamableHttpParseSse:
    @pytest.mark.asyncio
    async def test_sse_without_message_event(self):
        server = RemoteMCPServerConfig(name="s", transport="streamable_http", url="http://x")

        def handler(request: httpx.Request) -> httpx.Response:
            body = "\n".join([
                "event: update",
                'data: {"jsonrpc":"2.0","id":1,"result":{}}',
                "",
            ])
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

        client = StreamableHttpMCPClient(server, transport=httpx.MockTransport(handler))
        with pytest.raises(RuntimeError, match="SSE without"):
            await client._parse_sse_response(
                httpx.Response(200, headers={"content-type": "text/event-stream"}, content=b"event: update\ndata: {}\n\n")
            )


class TestStreamableHttpCallToolIsError:
    @pytest.mark.asyncio
    async def test_call_tool_error_result(self):
        server = RemoteMCPServerConfig(name="err", transport="streamable_http", url="http://x")

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content.decode("utf-8"))
            method = payload.get("method")
            if method == "initialize":
                return httpx.Response(200, json={
                    "jsonrpc": "2.0", "id": payload["id"],
                    "result": {"protocolVersion": "2025-11-25", "capabilities": {}, "serverInfo": {"name": "m", "version": "1.0"}},
                })
            if method == "notifications/initialized":
                return httpx.Response(200, json={})
            if method == "tools/call":
                return httpx.Response(200, json={
                    "jsonrpc": "2.0", "id": payload["id"],
                    "result": {"isError": True, "content": [{"type": "text", "text": "tool failed"}]},
                })
            raise AssertionError(f"unexpected: {method}")

        client = StreamableHttpMCPClient(server, transport=httpx.MockTransport(handler))
        result = await client.call_tool("mytool", {"x": 1})
        assert "Error" in result


class TestStreamableHttpCallToolEmptyContent:
    @pytest.mark.asyncio
    async def test_call_tool_empty_content(self):
        server = RemoteMCPServerConfig(name="empty", transport="streamable_http", url="http://x")

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content.decode("utf-8"))
            method = payload.get("method")
            if method == "initialize":
                return httpx.Response(200, json={
                    "jsonrpc": "2.0", "id": payload["id"],
                    "result": {"protocolVersion": "2025-11-25", "capabilities": {}, "serverInfo": {"name": "m", "version": "1.0"}},
                })
            if method == "notifications/initialized":
                return httpx.Response(200, json={})
            if method == "tools/call":
                return httpx.Response(200, json={
                    "jsonrpc": "2.0", "id": payload["id"],
                    "result": {},
                })
            raise AssertionError(f"unexpected: {method}")

        client = StreamableHttpMCPClient(server, transport=httpx.MockTransport(handler))
        result = await client.call_tool("mytool", {"x": 1})
        assert result == "{}" or "json" in result.lower()


class TestStreamableHttpCallToolNonTextContent:
    @pytest.mark.asyncio
    async def test_call_tool_non_text_content(self):
        server = RemoteMCPServerConfig(name="nt", transport="streamable_http", url="http://x")

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content.decode("utf-8"))
            method = payload.get("method")
            if method == "initialize":
                return httpx.Response(200, json={
                    "jsonrpc": "2.0", "id": payload["id"],
                    "result": {"protocolVersion": "2025-11-25", "capabilities": {}, "serverInfo": {"name": "m", "version": "1.0"}},
                })
            if method == "notifications/initialized":
                return httpx.Response(200, json={})
            if method == "tools/call":
                return httpx.Response(200, json={
                    "jsonrpc": "2.0", "id": payload["id"],
                    "result": {"content": [{"type": "image", "data": "base64..."}]},
                })
            raise AssertionError(f"unexpected: {method}")

        client = StreamableHttpMCPClient(server, transport=httpx.MockTransport(handler))
        result = await client.call_tool("mytool", {"x": 1})
        assert "image" in result


class TestStreamableHttpClose:
    @pytest.mark.asyncio
    async def test_close_with_no_client(self):
        server = RemoteMCPServerConfig(name="x", transport="streamable_http", url="http://x")
        client = StreamableHttpMCPClient(server)
        await client.close()
        assert client._client is None


class TestStdioMcpValidation:
    def test_invalid_transport(self):
        server = RemoteMCPServerConfig(name="x", transport="streamable_http", url="http://x", command=None)
        with pytest.raises(ValueError, match="Unsupported MCP transport"):
            StdioMCPClient(server)

    def test_missing_command(self):
        server = RemoteMCPServerConfig(name="x", transport="stdio", command=None)
        with pytest.raises(ValueError, match="command"):
            StdioMCPClient(server)


class TestStdioMcpCallToolIsError:
    @pytest.mark.asyncio
    async def test_call_tool_is_error(self, tmp_path):
        script = tmp_path / "mock_stdio_err.py"
        script.write_text(
            """
import json
import sys

for line in sys.stdin:
    payload = json.loads(line)
    method = payload.get("method")
    if method == "initialize":
        print(json.dumps({
            "jsonrpc": "2.0",
            "id": payload["id"],
            "result": {"protocolVersion": "2025-11-25", "capabilities": {}, "serverInfo": {"name": "m", "version": "1.0"}},
        }), flush=True)
    elif method == "notifications/initialized":
        continue
    elif method == "tools/call":
        print(json.dumps({
            "jsonrpc": "2.0",
            "id": payload["id"],
            "result": {"isError": True, "content": [{"type": "text", "text": "failed"}]},
        }), flush=True)
""",
            encoding="utf-8",
        )
        server = RemoteMCPServerConfig(
            name="err-stdio",
            transport="stdio",
            command="python3",
            args=["-u", str(script)],
        )
        client = StdioMCPClient(server)
        try:
            result = await client.call_tool("tool", {"x": 1})
            assert "Error" in result
        finally:
            await client.close()


class TestStdioMcpCloseNoProcess:
    @pytest.mark.asyncio
    async def test_close_no_process(self):
        server = RemoteMCPServerConfig(name="x", transport="stdio", command="echo")
        client = StdioMCPClient(server)
        client._process = None
        await client.close()


class TestStdioMcpCloseKillOnTimeout:
    @pytest.mark.asyncio
    async def test_close_kills_on_timeout(self):
        import sys

        server = RemoteMCPServerConfig(name="x", transport="stdio", command="sleep", args=["300"])
        client = StdioMCPClient(server)
        client._process = None
        proc = await client._ensure_process()
        client._process = proc
        with patch("asyncio.wait_for", side_effect=TimeoutError()):
            await client.close()
        assert client._process is None


class TestFailureCacheFunctions:
    def test_remember_and_get(self):
        server = RemoteMCPServerConfig(name="test", transport="streamable_http", url="http://x")
        _clear_server_failure(server)
        _remember_server_failure(server, "conn refused")
        result = _get_cached_server_failure(server)
        assert result == "conn refused"

    def test_get_expired_returns_none(self):
        server = RemoteMCPServerConfig(name="test2", transport="streamable_http", url="http://x")
        _MCP_SERVER_FAILURE_CACHE[_server_cache_key(server)] = (0, "old error")
        result = _get_cached_server_failure(server)
        assert result is None

    def test_get_no_cache(self):
        server = RemoteMCPServerConfig(name="test3", transport="streamable_http", url="http://x")
        _MCP_SERVER_FAILURE_CACHE.pop(_server_cache_key(server), None)
        result = _get_cached_server_failure(server)
        assert result is None

    def test_clear(self):
        server = RemoteMCPServerConfig(name="test4", transport="streamable_http", url="http://x")
        _remember_server_failure(server, "err")
        _clear_server_failure(server)
        assert _get_cached_server_failure(server) is None


class TestBuildRegistryDisabledServer:
    @pytest.mark.asyncio
    async def test_disabled_server_skipped(self):
        server = RemoteMCPServerConfig(
            name="disabled",
            transport="streamable_http",
            url="http://x",
            enabled=False,
        )
        registry, active, clients = await build_remote_mcp_registry([server])
        assert active == []
        assert len(clients) == 0


class TestBuildRegistryUnsupportedTransport:
    @pytest.mark.asyncio
    async def test_unsupported_transport_raises(self):
        server = RemoteMCPServerConfig(
            name="bad_transport",
            transport="websocket",
            url="http://x",
        )
        with pytest.raises(ValueError, match="Unsupported MCP transport"):
            await build_remote_mcp_registry([server])


class TestBuildRegistryCloseOnDiscoveryFailure:
    @pytest.mark.asyncio
    async def test_closes_client_after_discovery_failure(self):
        server = RemoteMCPServerConfig(
            name="fail-close",
            transport="streamable_http",
            url="http://x",
        )

        def bad_transport(s):
            def handler(req):
                raise httpx.ConnectTimeout("timeout")
            return httpx.MockTransport(handler)

        _MCP_SERVER_FAILURE_CACHE.clear()
        registry, active, clients = await build_remote_mcp_registry(
            [server], transport_factory=bad_transport
        )
        assert active == []
