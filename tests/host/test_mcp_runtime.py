import json
import sys

import httpx
import pytest

from src.host.mcp_runtime import (
    MCP_PROTOCOL_VERSION,
    RemoteMCPServerConfig,
    StdioMCPClient,
    StreamableHttpMCPClient,
    _MCP_SERVER_FAILURE_CACHE,
    build_remote_mcp_registry,
)


def _mock_mcp_transport():
    initialized = {"done": False}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        method = payload.get("method")

        if method == "initialize":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "protocolVersion": MCP_PROTOCOL_VERSION,
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "mock", "version": "1.0.0"},
                    },
                },
            )

        if method == "notifications/initialized":
            initialized["done"] = True
            return httpx.Response(200, json={})

        if method == "tools/list":
            assert initialized["done"] is True
            assert request.headers["MCP-Protocol-Version"] == MCP_PROTOCOL_VERSION
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "tools": [
                            {
                                "name": "remote_echo",
                                "description": "Echo input",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "text": {"type": "string"},
                                    },
                                    "required": ["text"],
                                },
                            }
                        ]
                    },
                },
            )

        if method == "tools/call":
            assert payload["params"]["name"] == "remote_echo"
            assert payload["params"]["arguments"] == {"text": "pong"}
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": "pong",
                            }
                        ]
                    },
                },
            )

        raise AssertionError(f"Unexpected MCP method: {method}")

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_streamable_http_mcp_client_lists_and_calls_tools():
    server = RemoteMCPServerConfig(
        name="mock",
        transport="streamable_http",
        url="https://mcp.example.test",
    )
    client = StreamableHttpMCPClient(server, transport=_mock_mcp_transport())

    tools = await client.list_tools()
    assert len(tools) == 1
    assert tools[0].name == "remote_echo"
    assert tools[0].input_schema["required"] == ["text"]

    result = await client.call_tool("remote_echo", {"text": "pong"})
    assert result == "pong"


@pytest.mark.asyncio
async def test_streamable_http_mcp_client_parses_sse_payload():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        method = payload.get("method")

        if method == "initialize":
            body = "\n".join(
                [
                    "event: message",
                    'data: {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-11-25","capabilities":{"tools":{}},"serverInfo":{"name":"mock","version":"1.0.0"}}}',
                    "",
                ]
            )
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=body.encode("utf-8"),
            )

        if method == "notifications/initialized":
            return httpx.Response(200, json={})

        if method == "tools/list":
            body = "\n".join(
                [
                    "event: message",
                    'data: {"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"sse_tool","description":"SSE tool","inputSchema":{"type":"object","properties":{},"required":[]}}]}}',
                    "",
                ]
            )
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=body.encode("utf-8"),
            )

        raise AssertionError(f"Unexpected MCP method: {method}")

    server = RemoteMCPServerConfig(
        name="mock-sse",
        transport="streamable_http",
        url="https://mcp.example.test",
    )
    client = StreamableHttpMCPClient(server, transport=httpx.MockTransport(handler))

    tools = await client.list_tools()
    assert [tool.name for tool in tools] == ["sse_tool"]


@pytest.mark.asyncio
async def test_streamable_http_mcp_client_reuses_mcp_session_id():
    observed_session_headers: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        method = payload.get("method")
        observed_session_headers.append(request.headers.get("mcp-session-id"))

        if method == "initialize":
            return httpx.Response(
                200,
                headers={"mcp-session-id": "session-123"},
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "protocolVersion": MCP_PROTOCOL_VERSION,
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "mock", "version": "1.0.0"},
                    },
                },
            )

        if method == "notifications/initialized":
            assert request.headers.get("mcp-session-id") == "session-123"
            return httpx.Response(200, json={})

        if method == "tools/list":
            assert request.headers.get("mcp-session-id") == "session-123"
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {"tools": []},
                },
            )

        raise AssertionError(f"Unexpected MCP method: {method}")

    server = RemoteMCPServerConfig(
        name="mock-session",
        transport="streamable_http",
        url="https://mcp.example.test",
    )
    client = StreamableHttpMCPClient(server, transport=httpx.MockTransport(handler))
    try:
        await client.list_tools()
        assert observed_session_headers == [None, "session-123", "session-123"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_streamable_http_mcp_client_retries_transient_network_errors():
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        method = payload.get("method")
        attempts["count"] += 1

        if attempts["count"] == 1:
            raise httpx.ConnectTimeout("timed out")

        if method == "initialize":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "protocolVersion": MCP_PROTOCOL_VERSION,
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "mock", "version": "1.0.0"},
                    },
                },
            )

        if method == "notifications/initialized":
            return httpx.Response(200, json={})

        if method == "tools/list":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {"tools": []},
                },
            )

        raise AssertionError(f"Unexpected MCP method: {method}")

    server = RemoteMCPServerConfig(
        name="retrying",
        transport="streamable_http",
        url="https://mcp.example.test",
    )
    client = StreamableHttpMCPClient(server, transport=httpx.MockTransport(handler))
    try:
        tools = await client.list_tools()
        assert tools == []
        assert attempts["count"] >= 3
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_build_remote_mcp_registry_uses_prefix():
    server = RemoteMCPServerConfig(
        name="mock",
        transport="streamable_http",
        url="https://mcp.example.test",
        tool_prefix="docs",
    )

    registry, active, clients = await build_remote_mcp_registry(
        [server],
        transport_factory=lambda _server: _mock_mcp_transport(),
    )

    assert active == ["mock"]
    assert len(clients) == 1
    assert "docs_remote_echo" in registry.get_tool_names()
    result = await registry.call_tool("docs_remote_echo", {"text": "pong"})
    assert result == "pong"


@pytest.mark.asyncio
async def test_build_remote_mcp_registry_raises_on_name_collision():
    server_one = RemoteMCPServerConfig(
        name="one",
        transport="streamable_http",
        url="https://mcp.example.test/one",
    )
    server_two = RemoteMCPServerConfig(
        name="two",
        transport="streamable_http",
        url="https://mcp.example.test/two",
    )

    with pytest.raises(RuntimeError, match="tool name collision"):
        await build_remote_mcp_registry(
            [server_one, server_two],
            transport_factory=lambda _server: _mock_mcp_transport(),
        )


@pytest.mark.asyncio
async def test_build_remote_mcp_registry_skips_failed_server_and_keeps_healthy_one():
    healthy = RemoteMCPServerConfig(
        name="healthy",
        transport="streamable_http",
        url="https://mcp.example.test/healthy",
        tool_prefix="ok",
    )
    failing = RemoteMCPServerConfig(
        name="failing",
        transport="streamable_http",
        url="https://mcp.example.test/failing",
        tool_prefix="bad",
    )

    def transport_factory(server: RemoteMCPServerConfig):
        if server.name == "failing":
            def handler(_request: httpx.Request) -> httpx.Response:
                raise httpx.ConnectTimeout("timed out")
            return httpx.MockTransport(handler)
        return _mock_mcp_transport()

    registry, active, clients = await build_remote_mcp_registry(
        [failing, healthy],
        transport_factory=transport_factory,
    )

    try:
        assert active == ["healthy"]
        assert len(clients) == 1
        assert "ok_remote_echo" in registry.get_tool_names()
        assert "bad_remote_echo" not in registry.get_tool_names()
        assert getattr(registry, "_mcp_server_errors", {}) == {"failing": "timed out"}
    finally:
        for client in clients:
            await client.close()


@pytest.mark.asyncio
async def test_build_remote_mcp_registry_skips_recently_failed_server_during_cooldown():
    _MCP_SERVER_FAILURE_CACHE.clear()

    failing = RemoteMCPServerConfig(
        name="failing",
        transport="streamable_http",
        url="https://mcp.example.test/failing",
        tool_prefix="bad",
    )
    attempts = {"count": 0}

    def transport_factory(_server: RemoteMCPServerConfig):
        def handler(_request: httpx.Request) -> httpx.Response:
            attempts["count"] += 1
            raise httpx.ConnectTimeout("timed out")

        return httpx.MockTransport(handler)

    registry, active, clients = await build_remote_mcp_registry(
        [failing],
        transport_factory=transport_factory,
    )
    assert active == []
    assert len(clients) == 0
    assert attempts["count"] > 0
    assert getattr(registry, "_mcp_server_errors", {}) == {"failing": "timed out"}

    first_attempt_count = attempts["count"]
    registry, active, clients = await build_remote_mcp_registry(
        [failing],
        transport_factory=transport_factory,
    )
    assert active == []
    assert len(clients) == 0
    assert attempts["count"] == first_attempt_count
    assert getattr(registry, "_mcp_server_errors", {}) == {
        "failing": "cooldown: timed out"
    }

    _MCP_SERVER_FAILURE_CACHE.clear()


@pytest.mark.asyncio
async def test_stdio_mcp_client_lists_and_calls_tools(tmp_path):
    script = tmp_path / "mock_stdio_mcp.py"
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
            "result": {
                "protocolVersion": "2025-11-25",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "stdio-mock", "version": "1.0.0"},
            },
        }), flush=True)
    elif method == "notifications/initialized":
        continue
    elif method == "tools/list":
        print(json.dumps({
            "jsonrpc": "2.0",
            "id": payload["id"],
            "result": {
                "tools": [{
                    "name": "echo",
                    "description": "Echo input",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                }]
            },
        }), flush=True)
    elif method == "tools/call":
        print(json.dumps({
            "jsonrpc": "2.0",
            "id": payload["id"],
            "result": {
                "content": [{"type": "text", "text": payload["params"]["arguments"]["text"]}]
            },
        }), flush=True)
""",
        encoding="utf-8",
    )

    server = RemoteMCPServerConfig(
        name="stdio",
        transport="stdio",
        command=sys.executable,
        args=["-u", str(script)],
    )
    client = StdioMCPClient(server)

    try:
        tools = await client.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "echo"

        result = await client.call_tool("echo", {"text": "pong"})
        assert result == "pong"
    finally:
        await client.close()
