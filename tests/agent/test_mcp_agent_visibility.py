import json
import sys

import pytest

from src.agent.adapter import AgentSession
from src.agent.doraemon import create_doraemon_agent_with_tools
from src.host.mcp_registry import create_tool_registry


class DummyLLMClient:
    pass


@pytest.mark.asyncio
async def test_remote_mcp_tools_are_visible_to_agent(tmp_path):
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
                "serverInfo": {"name": "docs", "version": "1.0.0"},
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
                    "name": "lookup",
                    "description": "Lookup docs",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                }]
            },
        }), flush=True)
    elif method == "tools/call":
        print(json.dumps({
            "jsonrpc": "2.0",
            "id": payload["id"],
            "result": {
                "content": [{"type": "text", "text": "doc result"}]
            },
        }), flush=True)
""",
        encoding="utf-8",
    )

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model": "gpt-5.4",
                "mcp_servers": [
                    {
                        "name": "docs",
                        "transport": "stdio",
                        "command": sys.executable,
                        "args": ["-u", str(script)],
                        "tool_prefix": "docs",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    agent = await create_doraemon_agent_with_tools(
        llm_client=DummyLLMClient(),
        mode="build",
        config_path=config_path,
    )
    try:
        assert "docs_lookup" in [tool.name for tool in agent.tools]
        assert "write" in [tool.name for tool in agent.tools]
    finally:
        for client in getattr(agent.tool_registry, "_mcp_clients", []):
            await client.close()


@pytest.mark.asyncio
async def test_agent_session_uses_remote_mcp_extensions_from_passed_registry(tmp_path):
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
                "serverInfo": {"name": "docs", "version": "1.0.0"},
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
                    "name": "lookup",
                    "description": "Lookup docs",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                }]
            },
        }), flush=True)
""",
        encoding="utf-8",
    )

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model": "gpt-5.4",
                "mcp_servers": [
                    {
                        "name": "docs",
                        "transport": "stdio",
                        "command": sys.executable,
                        "args": ["-u", str(script)],
                        "tool_prefix": "docs",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class DummyModelClient:
        pass

    registry = await create_tool_registry(config_path=config_path, mode="build")
    session = AgentSession(
        model_client=DummyModelClient(),
        registry=registry,
        mode="build",
        config_path=config_path,
        project_dir=tmp_path,
        enable_trace=False,
    )
    try:
        await session.initialize()
        assert "docs" in session._mcp_extensions
        assert "docs_lookup" in [tool.name for tool in session._agent.tools]
    finally:
        await session.aclose()
