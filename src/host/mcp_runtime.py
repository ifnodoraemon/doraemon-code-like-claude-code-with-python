"""Remote MCP runtime over Streamable HTTP and stdio."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from src.host.tools import ToolDefinition, ToolRegistry

logger = logging.getLogger(__name__)

MCP_PROTOCOL_VERSION = "2025-11-25"
MCP_SERVER_FAILURE_COOLDOWN_SECONDS = 60.0
_MCP_SERVER_FAILURE_CACHE: dict[str, tuple[float, str]] = {}


@dataclass
class RemoteMCPServerConfig:
    """Runtime configuration for a remote MCP server."""

    name: str
    transport: str
    url: str | None = None
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 30.0
    tool_prefix: str | None = None
    enabled: bool = True


@dataclass
class RemoteMCPTool:
    """Remote MCP tool descriptor."""

    name: str
    description: str
    input_schema: dict[str, Any]


class StreamableHttpMCPClient:
    """Minimal Streamable HTTP MCP client."""

    def __init__(
        self,
        server: RemoteMCPServerConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        if server.transport != "streamable_http":
            raise ValueError(f"Unsupported MCP transport: {server.transport}")

        self.server = server
        self.transport = transport
        self._client: httpx.AsyncClient | None = None
        self._request_id = 0
        self._initialized = False
        self._protocol_version = MCP_PROTOCOL_VERSION
        self._session_id: str | None = None

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _headers(self, *, include_protocol: bool) -> dict[str, str]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            **self.server.headers,
        }
        if include_protocol:
            headers["MCP-Protocol-Version"] = self._protocol_version
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            timeout = httpx.Timeout(
                connect=min(5.0, self.server.timeout_seconds),
                read=self.server.timeout_seconds,
                write=self.server.timeout_seconds,
                pool=self.server.timeout_seconds,
            )
            self._client = httpx.AsyncClient(
                timeout=timeout,
                transport=self.transport,
                trust_env=False,
            )
        return self._client

    async def _parse_sse_response(self, response: httpx.Response) -> dict[str, Any]:
        """Parse the first JSON-RPC payload from an SSE response."""
        event_type = "message"
        async for line in response.aiter_lines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("event:"):
                event_type = stripped.split(":", 1)[1].strip()
                continue
            if stripped.startswith("data:"):
                payload = stripped.split(":", 1)[1].strip()
                if payload == "[DONE]":
                    continue
                if event_type != "message":
                    continue
                return json.loads(payload)
        raise RuntimeError(
            f"MCP server '{self.server.name}' returned SSE without a JSON-RPC message payload"
        )

    async def _post(
        self,
        payload: dict[str, Any],
        *,
        include_protocol: bool,
    ) -> dict[str, Any]:
        attempts = 3
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                client = await self._get_client()
                async with client.stream(
                    "POST",
                    self.server.url,
                    json=payload,
                    headers=self._headers(include_protocol=include_protocol),
                ) as response:
                    response.raise_for_status()
                    session_id = response.headers.get("mcp-session-id")
                    if session_id:
                        self._session_id = session_id
                    content_type = response.headers.get("content-type", "").lower()
                    if content_type.startswith("text/event-stream"):
                        return await self._parse_sse_response(response)
                    body = await response.aread()
                    if not body.strip():
                        return {}
                    return json.loads(body.decode("utf-8"))
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
                if self._client is not None:
                    await self._client.aclose()
                    self._client = None
                if attempt >= attempts:
                    raise
                backoff = 0.5 * attempt
                logger.warning(
                    "Retrying MCP server '%s' after %s on attempt %s/%s",
                    self.server.name,
                    exc.__class__.__name__,
                    attempt,
                    attempts,
                )
                await asyncio.sleep(backoff)

        assert last_error is not None
        raise last_error

    async def initialize(self) -> None:
        """Perform MCP initialization handshake."""
        if self._initialized:
            return

        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "doraemon-code",
                    "version": "0.1.0",
                },
            },
        }
        response = await self._post(payload, include_protocol=False)
        result = response.get("result") or {}
        protocol_version = result.get("protocolVersion") or MCP_PROTOCOL_VERSION
        self._protocol_version = protocol_version

        await self._post(
            {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            },
            include_protocol=True,
        )
        self._initialized = True

    async def list_tools(self) -> list[RemoteMCPTool]:
        """List tools exposed by the remote MCP server."""
        await self.initialize()

        tools: list[RemoteMCPTool] = []
        cursor: str | None = None

        while True:
            params = {"cursor": cursor} if cursor else {}
            response = await self._post(
                {
                    "jsonrpc": "2.0",
                    "id": self._next_id(),
                    "method": "tools/list",
                    "params": params,
                },
                include_protocol=True,
            )
            result = response.get("result") or {}
            for item in result.get("tools", []):
                tools.append(
                    RemoteMCPTool(
                        name=item["name"],
                        description=item.get("description", ""),
                        input_schema=item.get("inputSchema")
                        or {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        },
                    )
                )
            cursor = result.get("nextCursor")
            if not cursor:
                break

        return tools

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Call a remote MCP tool and normalize the result to text."""
        await self.initialize()
        response = await self._post(
            {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments,
                },
            },
            include_protocol=True,
        )
        result = response.get("result") or {}

        if result.get("isError"):
            return f"Error: {json.dumps(result, ensure_ascii=False)}"

        content = result.get("content") or []
        if not content:
            return json.dumps(result, ensure_ascii=False)

        text_parts: list[str] = []
        for item in content:
            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            else:
                text_parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(part for part in text_parts if part).strip()

    async def close(self) -> None:
        """Close the underlying HTTP client, if initialized."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class StdioMCPClient:
    """Minimal stdio MCP client."""

    def __init__(self, server: RemoteMCPServerConfig):
        if server.transport != "stdio":
            raise ValueError(f"Unsupported MCP transport: {server.transport}")
        if not server.command:
            raise ValueError("stdio MCP server requires command")

        self.server = server
        self._request_id = 0
        self._initialized = False
        self._protocol_version = MCP_PROTOCOL_VERSION
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    _BLOCKED_ENV_KEYS = frozenset({
        "LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH",
        "PYTHONHOME", "PYTHONSTARTUP", "PYTHONIOENCODING",
        "PATH", "HOME", "USER", "SHELL",
        "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH",
    })

    async def _ensure_process(self) -> asyncio.subprocess.Process:
        if self._process and self._process.returncode is None:
            return self._process

        safe_env_keys = {
            "PATH",
            "HOME",
            "USER",
            "LANG",
            "LC_ALL",
            "TERM",
            "SHELL",
            "TMPDIR",
            "TEMP",
            "TMP",
        }
        env = {k: v for k, v in os.environ.items() if k in safe_env_keys}
        for k, v in self.server.env.items():
            if k in self._BLOCKED_ENV_KEYS:
                logger.warning(
                    "Blocked dangerous env key '%s' for MCP server '%s'",
                    k, self.server.name,
                )
                continue
            env[k] = v
        self._process = await asyncio.create_subprocess_exec(
            self.server.command,
            *self.server.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.server.cwd or None,
            env=env,
        )
        return self._process

    async def _read_response(self, expected_id: int | None) -> dict[str, Any]:
        process = await self._ensure_process()
        assert process.stdout is not None

        while True:
            line = await process.stdout.readline()
            if not line:
                raise RuntimeError(f"MCP stdio server '{self.server.name}' closed stdout")
            payload = json.loads(line.decode("utf-8").strip())
            if expected_id is None:
                return payload
            if payload.get("id") == expected_id:
                return payload

    async def _send(
        self, payload: dict[str, Any], *, expect_response: bool
    ) -> dict[str, Any] | None:
        process = await self._ensure_process()
        assert process.stdin is not None

        async with self._lock:
            process.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
            await process.stdin.drain()
            if not expect_response:
                return None
            return await self._read_response(payload["id"])

    async def initialize(self) -> None:
        if self._initialized:
            return

        response = await self._send(
            {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {
                        "name": "doraemon-code",
                        "version": "0.1.0",
                    },
                },
            },
            expect_response=True,
        )
        assert response is not None
        result = response.get("result") or {}
        self._protocol_version = result.get("protocolVersion") or MCP_PROTOCOL_VERSION

        await self._send(
            {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            },
            expect_response=False,
        )
        self._initialized = True

    async def list_tools(self) -> list[RemoteMCPTool]:
        await self.initialize()
        response = await self._send(
            {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/list",
                "params": {},
            },
            expect_response=True,
        )
        assert response is not None
        result = response.get("result") or {}
        return [
            RemoteMCPTool(
                name=item["name"],
                description=item.get("description", ""),
                input_schema=item.get("inputSchema")
                or {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            )
            for item in result.get("tools", [])
        ]

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        await self.initialize()
        response = await self._send(
            {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments,
                },
            },
            expect_response=True,
        )
        assert response is not None
        result = response.get("result") or {}
        if result.get("isError"):
            return f"Error: {json.dumps(result, ensure_ascii=False)}"
        content = result.get("content") or []
        text_parts = []
        for item in content:
            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            else:
                text_parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(part for part in text_parts if part).strip()

    async def close(self) -> None:
        process = self._process
        if process is None:
            return

        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except TimeoutError:
                process.kill()
                await process.wait()
        self._process = None


def _resolved_tool_name(server: RemoteMCPServerConfig, tool_name: str) -> str:
    if server.tool_prefix:
        return f"{server.tool_prefix}_{tool_name}"
    return tool_name


def _server_cache_key(server: RemoteMCPServerConfig) -> str:
    return f"{server.transport}:{server.name}:{server.url or server.command or ''}"


def _get_cached_server_failure(server: RemoteMCPServerConfig) -> str | None:
    key = _server_cache_key(server)
    cached = _MCP_SERVER_FAILURE_CACHE.get(key)
    if not cached:
        return None

    expires_at, message = cached
    if time.time() >= expires_at:
        _MCP_SERVER_FAILURE_CACHE.pop(key, None)
        return None
    return message


def _remember_server_failure(server: RemoteMCPServerConfig, message: str) -> None:
    _MCP_SERVER_FAILURE_CACHE[_server_cache_key(server)] = (
        time.time() + MCP_SERVER_FAILURE_COOLDOWN_SECONDS,
        message,
    )


def _clear_server_failure(server: RemoteMCPServerConfig) -> None:
    _MCP_SERVER_FAILURE_CACHE.pop(_server_cache_key(server), None)


async def build_remote_mcp_registry(
    servers: list[RemoteMCPServerConfig],
    *,
    transport_factory: callable | None = None,
) -> tuple[ToolRegistry, list[str], list[Any]]:
    """Build a ToolRegistry from remote MCP servers."""
    registry = ToolRegistry()
    active_servers: list[str] = []
    clients: list[Any] = []
    server_errors: dict[str, str] = {}

    for server in servers:
        if not server.enabled:
            continue

        cached_failure = _get_cached_server_failure(server)
        if cached_failure:
            server_errors[server.name] = f"cooldown: {cached_failure}"
            logger.warning(
                "Skipping MCP server '%s' during cooldown after recent failure: %s",
                server.name,
                cached_failure,
            )
            continue

        if server.transport == "streamable_http":
            client = StreamableHttpMCPClient(
                server,
                transport=transport_factory(server) if transport_factory else None,
            )
        elif server.transport == "stdio":
            client = StdioMCPClient(server)
        else:
            raise ValueError(f"Unsupported MCP transport: {server.transport}")
        try:
            remote_tools = await client.list_tools()
        except Exception as exc:
            message = str(exc) or exc.__class__.__name__
            server_errors[server.name] = message
            _remember_server_failure(server, message)
            logger.warning(
                "Skipping MCP server '%s' after discovery failure: %s",
                server.name,
                message,
            )
            try:
                await client.close()
            except Exception:
                logger.debug(
                    "Failed to close MCP client for server '%s' after discovery failure",
                    server.name,
                    exc_info=True,
                )
            continue

        clients.append(client)
        _clear_server_failure(server)

        for remote_tool in remote_tools:
            resolved_name = _resolved_tool_name(server, remote_tool.name)
            if resolved_name in registry.get_tool_names():
                raise RuntimeError(
                    f"MCP tool name collision: {resolved_name}. "
                    f"Set tool_prefix for server '{server.name}'."
                )

            async def _call_remote_tool(
                _arguments: dict[str, Any] | None = None,
                *,
                _client: StreamableHttpMCPClient = client,
                _tool_name: str = remote_tool.name,
                **kwargs,
            ) -> str:
                arguments = _arguments.copy() if _arguments else {}
                arguments.update(kwargs)
                return await _client.call_tool(_tool_name, arguments)

            registry._tools[resolved_name] = ToolDefinition(
                name=resolved_name,
                description=remote_tool.description,
                function=_call_remote_tool,
                parameters=remote_tool.input_schema,
                sensitive=False,
                timeout=server.timeout_seconds,
                source="mcp_remote",
                metadata={
                    "mcp_server": server.name,
                    "mcp_transport": server.transport,
                    "remote_tool_name": remote_tool.name,
                },
            )

        active_servers.append(server.name)

    registry._mcp_server_errors = server_errors
    return registry, active_servers, clients
