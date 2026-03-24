"""
MCP Client

Model Context Protocol client for connecting to external MCP servers.

Features:
- Connect to MCP servers via stdio/HTTP
- Tool discovery and invocation
- Resource access
- Prompt templates
"""

import asyncio
import inspect
import json
import logging
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MCPTransport(Enum):
    """MCP transport types."""

    STDIO = "stdio"  # Standard input/output
    HTTP = "http"  # HTTP/SSE
    WEBSOCKET = "websocket"  # WebSocket


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""

    name: str
    command: str  # Command to start server (for stdio)
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    transport: MCPTransport = MCPTransport.STDIO
    url: str | None = None  # For HTTP/WebSocket transport
    timeout: float = 30.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": self.command,
            "args": self.args,
            "env": self.env,
            "transport": self.transport.value,
            "url": self.url,
            "timeout": self.timeout,
        }


@dataclass
class MCPTool:
    """A tool provided by an MCP server."""

    name: str
    description: str
    input_schema: dict[str, Any]
    server_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "server_name": self.server_name,
        }


@dataclass
class MCPResource:
    """A resource provided by an MCP server."""

    uri: str
    name: str
    description: str
    mime_type: str | None = None
    server_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mime_type": self.mime_type,
            "server_name": self.server_name,
        }


@dataclass
class MCPPrompt:
    """A prompt template from an MCP server."""

    name: str
    description: str
    arguments: list[dict[str, Any]]
    server_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "arguments": self.arguments,
            "server_name": self.server_name,
        }


class MCPConnection:
    """
    Connection to a single MCP server.

    Handles the JSON-RPC communication over stdio.
    """

    def __init__(self, config: MCPServerConfig):
        """
        Initialize MCP connection.

        Args:
            config: Server configuration
        """
        self.config = config
        self._process: subprocess.Popen | None = None
        self._request_id = 0
        self._pending_requests: dict[int, asyncio.Future] = {}
        self._reader_task: asyncio.Task | None = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected and self._process is not None

    async def connect(self) -> bool:
        """
        Connect to the MCP server.

        Returns:
            True if connected successfully
        """
        if self._connected:
            return True

        try:
            # Start the server process
            env = {**dict(subprocess.os.environ), **self.config.env}

            self._process = subprocess.Popen(
                [self.config.command] + self.config.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )

            if getattr(self._request, "__func__", None) is MCPConnection._request:
                self._reader_task = asyncio.create_task(self._read_responses())

            # Send initialize request
            result = await self._request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "roots": {"listChanged": True},
                    },
                    "clientInfo": {
                        "name": "doraemon",
                        "version": "0.7.0",
                    },
                },
            )

            if result:
                self._connected = True
                # Send initialized notification
                await self._notify("notifications/initialized", {})
                logger.info(f"Connected to MCP server: {self.config.name}")
                return True

            await self.disconnect()
            return False

        except Exception as e:
            await self.disconnect()
            logger.error(f"Failed to connect to MCP server {self.config.name}: {e}")
            return False

    async def disconnect(self):
        """Disconnect from the MCP server."""
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        if self._process:
            try:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=2)
            except Exception as e:
                logger.warning(f"Error cleaning up MCP process: {e}")
            finally:
                self._process = None

        self._connected = False
        logger.info(f"Disconnected from MCP server: {self.config.name}")

    async def _read_responses(self):
        """Read responses from the server."""
        if not self._process or not self._process.stdout:
            return

        while True:
            try:
                readline = self._process.stdout.readline
                if inspect.iscoroutinefunction(readline):
                    line = await readline()
                else:
                    line = await asyncio.to_thread(readline)
                    if inspect.isawaitable(line):
                        line = await line

                if not line:
                    break

                # Parse JSON-RPC response
                try:
                    message = json.loads(line.decode("utf-8"))
                    await self._handle_message(message)
                except json.JSONDecodeError:
                    continue

            except Exception as e:
                logger.error(f"Error reading from MCP server: {e}")
                break

    async def _handle_message(self, message: dict):
        """Handle a message from the server."""
        if "id" in message:
            # Response to a request
            request_id = message["id"]
            if request_id in self._pending_requests:
                future = self._pending_requests.pop(request_id)
                if "error" in message:
                    future.set_exception(
                        Exception(message["error"].get("message", "Unknown error"))
                    )
                else:
                    future.set_result(message.get("result"))

    async def _request(self, method: str, params: dict) -> Any:
        """Send a request and wait for response."""
        if not self._process or not self._process.stdin:
            raise Exception("Not connected")

        self._request_id += 1
        request_id = self._request_id

        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }

        # Create future for response
        future: asyncio.Future = asyncio.Future()
        self._pending_requests[request_id] = future

        # Send request
        message = json.dumps(request) + "\n"
        self._process.stdin.write(message.encode("utf-8"))
        self._process.stdin.flush()

        # Wait for response with timeout
        try:
            return await asyncio.wait_for(future, timeout=self.config.timeout)
        except asyncio.TimeoutError as e:
            self._pending_requests.pop(request_id, None)
            raise Exception(f"Request timeout: {method}") from e

    async def _notify(self, method: str, params: dict):
        """Send a notification (no response expected)."""
        if not self._process or not self._process.stdin:
            return

        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }

        message = json.dumps(notification) + "\n"
        self._process.stdin.write(message.encode("utf-8"))
        self._process.stdin.flush()

    async def list_tools(self) -> list[MCPTool]:
        """List available tools."""
        result = await self._request("tools/list", {})
        if result is None:
            return []
        tools = []

        for tool_data in result.get("tools", []):
            tools.append(
                MCPTool(
                    name=tool_data["name"],
                    description=tool_data.get("description", ""),
                    input_schema=tool_data.get("inputSchema", {}),
                    server_name=self.config.name,
                )
            )

        return tools

    async def call_tool(self, name: str, arguments: dict) -> Any:
        """
        Call a tool.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool result
        """
        result = await self._request(
            "tools/call",
            {"name": name, "arguments": arguments},
        )

        # Extract content - safely handle empty content
        if result is None:
            return None
        content = result.get("content", [])
        if content:
            first = content[0] if len(content) > 0 else None
            if first:
                if first.get("type") == "text":
                    return first.get("text", "")
                return first

        return result

    async def list_resources(self) -> list[MCPResource]:
        """List available resources."""
        result = await self._request("resources/list", {})
        if result is None:
            return []
        resources = []

        for res_data in result.get("resources", []):
            resources.append(
                MCPResource(
                    uri=res_data["uri"],
                    name=res_data.get("name", ""),
                    description=res_data.get("description", ""),
                    mime_type=res_data.get("mimeType"),
                    server_name=self.config.name,
                )
            )

        return resources

    async def read_resource(self, uri: str) -> str:
        """
        Read a resource.

        Args:
            uri: Resource URI

        Returns:
            Resource content
        """
        result = await self._request("resources/read", {"uri": uri})
        if result is None:
            return ""
        contents = result.get("contents", [])

        if contents:
            return contents[0].get("text", "")

        return ""

    async def list_prompts(self) -> list[MCPPrompt]:
        """List available prompts."""
        result = await self._request("prompts/list", {})
        if result is None:
            return []
        prompts = []

        for prompt_data in result.get("prompts", []):
            prompts.append(
                MCPPrompt(
                    name=prompt_data["name"],
                    description=prompt_data.get("description", ""),
                    arguments=prompt_data.get("arguments", []),
                    server_name=self.config.name,
                )
            )

        return prompts

    async def get_prompt(self, name: str, arguments: dict) -> str:
        """
        Get a prompt with arguments.

        Args:
            name: Prompt name
            arguments: Prompt arguments

        Returns:
            Rendered prompt
        """
        result = await self._request(
            "prompts/get",
            {"name": name, "arguments": arguments},
        )

        if result is None:
            return ""
        messages = result.get("messages", [])
        if messages:
            return messages[0].get("content", {}).get("text", "")

        return ""


class MCPClient:
    """
    MCP client managing multiple server connections.

    Usage:
        client = MCPClient()

        # Add servers
        client.add_server(MCPServerConfig(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/path"]
        ))

        # Connect
        await client.connect_all()

        # List all tools
        tools = await client.list_all_tools()

        # Call a tool
        result = await client.call_tool("filesystem", "read_file", {"path": "/file"})

        # Disconnect
        await client.disconnect_all()
    """

    def __init__(self, config_path: Path | None = None):
        """
        Initialize MCP client.

        Args:
            config_path: Path to MCP config file
        """
        self._servers: dict[str, MCPServerConfig] = {}
        self._connections: dict[str, MCPConnection] = {}
        self._config_path = config_path

        # Load config if provided
        if config_path and config_path.exists():
            self._load_config(config_path)

    def _load_config(self, path: Path):
        """Load server configurations from file."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))

            for name, server_data in data.get("mcpServers", {}).items():
                config = MCPServerConfig(
                    name=name,
                    command=server_data.get("command", ""),
                    args=server_data.get("args", []),
                    env=server_data.get("env", {}),
                )
                self._servers[name] = config

            logger.info(f"Loaded {len(self._servers)} MCP server configs")

        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}")

    def add_server(self, config: MCPServerConfig):
        """Add a server configuration."""
        self._servers[config.name] = config

    def remove_server(self, name: str):
        """Remove a server configuration."""
        self._servers.pop(name, None)

    async def connect(self, name: str) -> bool:
        """
        Connect to a specific server.

        Args:
            name: Server name

        Returns:
            True if connected
        """
        if name not in self._servers:
            logger.error(f"Unknown MCP server: {name}")
            return False

        if name in self._connections and self._connections[name].is_connected:
            return True

        connection = MCPConnection(self._servers[name])
        if await connection.connect():
            self._connections[name] = connection
            return True

        return False

    async def connect_all(self) -> dict[str, bool]:
        """
        Connect to all configured servers.

        Returns:
            Dict of server name -> success
        """
        results = {}
        for name in self._servers:
            results[name] = await self.connect(name)
        return results

    async def disconnect(self, name: str):
        """Disconnect from a specific server."""
        if name in self._connections:
            await self._connections[name].disconnect()
            del self._connections[name]

    async def disconnect_all(self):
        """Disconnect from all servers."""
        for name in list(self._connections.keys()):
            await self.disconnect(name)

    async def list_all_tools(self) -> list[MCPTool]:
        """List tools from all connected servers."""
        tools = []
        for conn in self._connections.values():
            try:
                server_tools = await conn.list_tools()
                tools.extend(server_tools)
            except Exception as e:
                logger.error(f"Failed to list tools from {conn.config.name}: {e}")
        return tools

    async def call_tool(self, server: str, tool: str, arguments: dict) -> Any:
        """
        Call a tool on a specific server.

        Args:
            server: Server name
            tool: Tool name
            arguments: Tool arguments

        Returns:
            Tool result
        """
        if server not in self._connections:
            raise Exception(f"Not connected to server: {server}")

        return await self._connections[server].call_tool(tool, arguments)

    async def list_all_resources(self) -> list[MCPResource]:
        """List resources from all connected servers."""
        resources = []
        for conn in self._connections.values():
            try:
                server_resources = await conn.list_resources()
                resources.extend(server_resources)
            except Exception as e:
                logger.error(f"Failed to list resources from {conn.config.name}: {e}")
        return resources

    async def read_resource(self, server: str, uri: str) -> str:
        """Read a resource from a specific server."""
        if server not in self._connections:
            raise Exception(f"Not connected to server: {server}")

        return await self._connections[server].read_resource(uri)

    def get_connected_servers(self) -> list[str]:
        """Get list of connected server names."""
        return [name for name, conn in self._connections.items() if conn.is_connected]

    def get_summary(self) -> dict[str, Any]:
        """Get client summary."""
        return {
            "configured_servers": list(self._servers.keys()),
            "connected_servers": self.get_connected_servers(),
            "total_configured": len(self._servers),
            "total_connected": len(self._connections),
        }
