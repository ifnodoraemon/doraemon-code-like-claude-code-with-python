import json
import sys

import pytest

from src.host.mcp_registry import (
    EXTENSION_GROUPS,
    create_tool_registry,
    get_enabled_mcp_extensions,
    get_enabled_mcp_servers,
    resolve_extension_tools,
)


def test_resolve_extension_tools_expands_named_groups():
    tools = resolve_extension_tools(["browser", "database"])

    assert set(EXTENSION_GROUPS["browser"]).issubset(set(tools))
    assert set(EXTENSION_GROUPS["database"]).issubset(set(tools))


def test_resolve_extension_tools_unknown_passes_through():
    tools = resolve_extension_tools(["custom_tool"])
    assert "custom_tool" in tools


def test_resolve_extension_tools_none_returns_empty():
    tools = resolve_extension_tools(None)
    assert tools == []


def test_get_enabled_mcp_extensions_empty_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"model": "gpt-5.4"}), encoding="utf-8")
    assert get_enabled_mcp_extensions(config_path) == []


def test_get_enabled_mcp_extensions_non_list_value(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"model": "gpt-5.4", "mcp_extensions": "browser"}), encoding="utf-8")
    assert get_enabled_mcp_extensions(config_path) == []


def test_get_enabled_mcp_extensions_with_non_string_items(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"model": "gpt-5.4", "mcp_extensions": ["browser", 123, None, "", "  ", "browser"]}),
        encoding="utf-8",
    )
    assert get_enabled_mcp_extensions(config_path) == ["browser"]


def test_get_enabled_mcp_extensions_invalid_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text("not json", encoding="utf-8")
    assert get_enabled_mcp_extensions(config_path) == []


def test_get_enabled_mcp_servers_empty_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"model": "gpt-5.4"}), encoding="utf-8")
    assert get_enabled_mcp_servers(config_path) == []


def test_get_enabled_mcp_servers_non_list_value(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"model": "gpt-5.4", "mcp_servers": "not-a-list"}),
        encoding="utf-8",
    )
    assert get_enabled_mcp_servers(config_path) == []


def test_get_enabled_mcp_servers_non_dict_items(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"model": "gpt-5.4", "mcp_servers": ["not-a-dict", 42]}),
        encoding="utf-8",
    )
    assert get_enabled_mcp_servers(config_path) == []


def test_get_enabled_mcp_servers_disabled_server(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({
            "model": "gpt-5.4",
            "mcp_servers": [{"name": "off", "transport": "stdio", "command": "cat", "enabled": False}],
        }),
        encoding="utf-8",
    )
    assert get_enabled_mcp_servers(config_path) == []


def test_get_enabled_mcp_servers_invalid_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text("not json", encoding="utf-8")
    assert get_enabled_mcp_servers(config_path) == []


def test_resolve_extension_tools_dedup():
    tools = resolve_extension_tools(["browser", "browser"])
    browser_tools = EXTENSION_GROUPS["browser"]
    for t in browser_tools:
        assert tools.count(t) == 1


def test_get_enabled_mcp_extensions_from_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model": "gpt-5.4",
                "mcp_extensions": ["browser", "database", "browser"],
            }
        ),
        encoding="utf-8",
    )

    assert get_enabled_mcp_extensions(config_path) == ["browser", "database"]


def test_get_enabled_mcp_servers_from_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model": "gpt-5.4",
                "mcp_servers": [
                    {
                        "name": "docs",
                        "transport": "streamable_http",
                        "url": "https://mcp.example.test",
                        "tool_prefix": "docs",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    servers = get_enabled_mcp_servers(config_path)
    assert len(servers) == 1
    assert servers[0].name == "docs"
    assert servers[0].url == "https://mcp.example.test"
    assert servers[0].tool_prefix == "docs"


@pytest.mark.asyncio
async def test_create_tool_registry_uses_configured_extensions(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model": "gpt-5.4",
                "mcp_extensions": ["database"],
            }
        ),
        encoding="utf-8",
    )

    registry = await create_tool_registry(config_path=config_path, mode="build")
    tools = registry.get_tool_names()

    assert "read" in tools
    assert "write" in tools
    assert "db_read_query" in tools
    assert "db_list_tables" in tools
    assert "browser_click" not in tools
    assert registry.get_tool_policy(
        "db_read_query",
        mode="build",
        active_mcp_extensions=["database"],
    )["visible"] is True


@pytest.mark.asyncio
async def test_create_tool_registry_with_extensions_does_not_mutate_default_mainline(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model": "gpt-5.4",
                "mcp_extensions": ["browser"],
            }
        ),
        encoding="utf-8",
    )

    extended = await create_tool_registry(config_path=config_path, mode="build")
    plain = await create_tool_registry(mode="build")

    assert "browser_click" in extended.get_tool_names()
    assert "browser_click" not in plain.get_tool_names()


@pytest.mark.asyncio
async def test_create_tool_registry_with_stdio_mcp_server(tmp_path):
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

    registry = await create_tool_registry(config_path=config_path, mode="build")
    try:
        assert "docs_lookup" in registry.get_tool_names()
        result = await registry.call_tool("docs_lookup", {"query": "mcp"})
        assert result == "doc result"
        assert getattr(registry, "_active_mcp_extensions", []) == ["docs"]
    finally:
        for client in getattr(registry, "_mcp_clients", []):
            await client.close()


@pytest.mark.asyncio
async def test_create_unified_registry_alias(tmp_path):
    from src.host.mcp_registry import create_unified_registry
    registry = await create_unified_registry(mode="build")
    assert "read" in registry.get_tool_names()


@pytest.mark.asyncio
async def test_create_tool_registry_with_extension_tools_override(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"model": "gpt-5.4"}), encoding="utf-8")
    registry = await create_tool_registry(
        config_path=config_path,
        mode="build",
        extension_tools=["database"],
    )
    assert "db_read_query" in registry.get_tool_names()
