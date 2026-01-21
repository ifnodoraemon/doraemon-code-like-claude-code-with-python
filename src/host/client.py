"""
Multi-Server MCP Client

Manages connections to multiple MCP servers and routes tool calls appropriately.
Handles server lifecycle, tool discovery, and execution tracing.
"""

import logging
import os
import time
from contextlib import AsyncExitStack
from typing import Any

from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.core.logger import TraceLogger

logger = logging.getLogger(__name__)


class MCPConnectionError(Exception):
    """Raised when MCP server connection fails."""

    def __init__(self, server_name: str, message: str):
        self.server_name = server_name
        super().__init__(f"Failed to connect to MCP server '{server_name}': {message}")


class ToolNotFoundError(Exception):
    """Raised when a requested tool is not found."""

    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' not found in any connected server")


class MultiServerMCPClient:
    """
    Client for managing multiple MCP server connections.

    Handles:
    - Server connection and lifecycle management
    - Tool discovery and routing
    - Execution tracing and logging
    """

    def __init__(self, tracer: TraceLogger | None = None):
        self.exit_stack = AsyncExitStack()
        self.sessions: dict[str, ClientSession] = {}
        self.tool_map: dict[str, str] = {}
        self.tracer = tracer or TraceLogger()
        self._connection_errors: list[MCPConnectionError] = []

    async def connect_to_config(self, config: dict[str, Any]):
        """
        Connect to all MCP servers defined in configuration.

        Args:
            config: Configuration dictionary with 'mcpServers' key

        Note:
            Connection errors are logged but don't stop other servers from connecting.
            Use get_connection_errors() to check for failures.
        """
        servers_conf = config.get("mcpServers", {})

        # 解析包根路径
        pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        for name, details in servers_conf.items():
            command = details.get("command")
            args = details.get("args", [])
            env = os.environ.copy()
            env.update(details.get("env", {}))

            # 智能路径解析
            resolved_args = []
            for arg in args:
                if arg.endswith(".py"):
                    # 1. 绝对路径 - 直接使用
                    if os.path.isabs(arg):
                        resolved_args.append(arg)
                        continue

                    # 2. 尝试相对于当前工作目录
                    cwd_path = os.path.abspath(os.path.join(os.getcwd(), arg))
                    if os.path.exists(cwd_path):
                        resolved_args.append(cwd_path)
                        continue

                    # 3. 尝试相对于项目根目录 (pkg_root)
                    # 假设 arg 像 "src/servers/fs_read.py" 或 "servers/fs_read.py"
                    # pkg_root 指向 polymath/src 的上一级，即 polymath/

                    # 如果 arg 包含 'src/', 我们假设它是从项目根开始的
                    # 如果不包含，可能是相对于 src/servers 的简写？(虽然 config 一般写全路径)

                    candidates = [
                        os.path.join(pkg_root, arg),  # polymath/src/servers/...
                        os.path.join(pkg_root, "src", arg) if not arg.startswith("src") else None,
                    ]

                    found = False
                    for p in candidates:
                        if p and os.path.exists(p):
                            resolved_args.append(p)
                            found = True
                            break

                    if not found:
                        # Fallback: keep original relative path and hope execution context is right
                        resolved_args.append(arg)
                else:
                    resolved_args.append(arg)

            params = StdioServerParameters(command=command, args=resolved_args, env=env)

            try:
                read, write = await self.exit_stack.enter_async_context(stdio_client(params))
                session = await self.exit_stack.enter_async_context(ClientSession(read, write))
                await session.initialize()

                self.sessions[name] = session

                tools_list = await session.list_tools()
                for tool in tools_list.tools:
                    self.tool_map[tool.name] = name

                logger.info(f"Connected to MCP server '{name}' with {len(tools_list.tools)} tools")

            except FileNotFoundError as e:
                error = MCPConnectionError(name, f"Server script not found: {e}")
                self._connection_errors.append(error)
                logger.error(str(error))
            except PermissionError as e:
                error = MCPConnectionError(name, f"Permission denied: {e}")
                self._connection_errors.append(error)
                logger.error(str(error))
            except ConnectionError as e:
                error = MCPConnectionError(name, f"Connection failed: {e}")
                self._connection_errors.append(error)
                logger.error(str(error))
            except Exception as e:
                error = MCPConnectionError(name, str(e))
                self._connection_errors.append(error)
                logger.error(str(error))

    async def get_genai_tools(self) -> list[Any]:
        """
        Convert MCP tool definitions to Google GenAI FunctionDeclarations.

        Returns:
            List of FunctionDeclaration objects for use with Gemini API
        """
        genai_tools = []

        for _server_name, session in self.sessions.items():
            result = await session.list_tools()
            for tool in result.tools:
                # 兼容性处理：OpenAI/Gemini 的 Schema 转换
                func_decl = types.FunctionDeclaration(
                    name=tool.name, description=tool.description or "", parameters=tool.inputSchema
                )
                genai_tools.append(func_decl)

        return genai_tools

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """
        Route tool call to appropriate server and record trace.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool execution result

        Raises:
            ToolNotFoundError: If tool is not registered
            Exception: If tool execution fails
        """
        start_time = time.time()

        try:
            server_name = self.tool_map.get(tool_name)
            if not server_name:
                raise ToolNotFoundError(tool_name)

            session = self.sessions[server_name]

            # Log Request
            self.tracer.log("tool_call", tool_name, arguments)

            result = await session.call_tool(tool_name, arguments)

            # Extract content
            content = "No content"
            if result.content and len(result.content) > 0:
                content = result.content[0].text

            duration = (time.time() - start_time) * 1000
            self.tracer.log("tool_result", tool_name, content, duration)

            return content

        except ToolNotFoundError:
            duration = (time.time() - start_time) * 1000
            self.tracer.log("tool_error", tool_name, "Tool not found", duration)
            raise
        except TimeoutError as e:
            duration = (time.time() - start_time) * 1000
            self.tracer.log("tool_error", tool_name, f"Timeout: {e}", duration)
            raise
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.tracer.log("tool_error", tool_name, str(e), duration)
            raise

    async def cleanup(self):
        await self.exit_stack.aclose()
