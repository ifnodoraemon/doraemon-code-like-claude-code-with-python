"""Comprehensive tests for mcp_client.py"""

from src.core.mcp_client import MCPPrompt, MCPResource, MCPServerConfig, MCPTool, MCPTransport


class TestMCPTransport:
    """Tests for MCPTransport enum."""

    def test_all_transports_defined(self):
        """Test that all transport types are defined."""
        assert MCPTransport.STDIO.value == "stdio"
        assert MCPTransport.HTTP.value == "http"
        assert MCPTransport.WEBSOCKET.value == "websocket"

    def test_transport_count(self):
        """Test expected number of transports."""
        assert len(MCPTransport) == 3


class TestMCPServerConfig:
    """Tests for MCPServerConfig."""

    def test_creation_minimal(self):
        """Test creating minimal server config."""
        config = MCPServerConfig(name="test_server", command="python")
        assert config.name == "test_server"
        assert config.command == "python"
        assert config.args == []
        assert config.env == {}
        assert config.transport == MCPTransport.STDIO
        assert config.timeout == 30.0

    def test_creation_with_args(self):
        """Test creating config with arguments."""
        config = MCPServerConfig(
            name="server_with_args", command="python", args=["-m", "server", "--port", "8000"]
        )
        assert len(config.args) == 4
        assert config.args[0] == "-m"
        assert config.args[-1] == "8000"

    def test_creation_with_env(self):
        """Test creating config with environment variables."""
        env = {"API_KEY": "secret", "DEBUG": "true"}
        config = MCPServerConfig(name="server_with_env", command="node", env=env)
        assert config.env["API_KEY"] == "secret"
        assert config.env["DEBUG"] == "true"

    def test_creation_http_transport(self):
        """Test creating config with HTTP transport."""
        config = MCPServerConfig(
            name="http_server", command="", transport=MCPTransport.HTTP, url="http://localhost:8000"
        )
        assert config.transport == MCPTransport.HTTP
        assert config.url == "http://localhost:8000"

    def test_creation_websocket_transport(self):
        """Test creating config with WebSocket transport."""
        config = MCPServerConfig(
            name="ws_server",
            command="",
            transport=MCPTransport.WEBSOCKET,
            url="ws://localhost:9000",
        )
        assert config.transport == MCPTransport.WEBSOCKET
        assert config.url == "ws://localhost:9000"

    def test_custom_timeout(self):
        """Test creating config with custom timeout."""
        config = MCPServerConfig(name="slow_server", command="python", timeout=120.0)
        assert config.timeout == 120.0

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = MCPServerConfig(
            name="test_server",
            command="python",
            args=["-m", "server"],
            env={"KEY": "value"},
            transport=MCPTransport.HTTP,
            url="http://localhost:8000",
            timeout=60.0,
        )
        data = config.to_dict()
        assert data["name"] == "test_server"
        assert data["command"] == "python"
        assert data["args"] == ["-m", "server"]
        assert data["env"] == {"KEY": "value"}
        assert data["transport"] == "http"
        assert data["url"] == "http://localhost:8000"
        assert data["timeout"] == 60.0

    def test_to_dict_stdio_default(self):
        """Test to_dict with default STDIO transport."""
        config = MCPServerConfig(name="stdio_server", command="python")
        data = config.to_dict()
        assert data["transport"] == "stdio"
        assert data["url"] is None


class TestMCPTool:
    """Tests for MCPTool."""

    def test_creation(self):
        """Test creating MCP tool."""
        schema = {
            "type": "object",
            "properties": {"file": {"type": "string"}, "content": {"type": "string"}},
            "required": ["file"],
        }
        tool = MCPTool(
            name="write_file",
            description="Write content to a file",
            input_schema=schema,
            server_name="filesystem",
        )
        assert tool.name == "write_file"
        assert tool.description == "Write content to a file"
        assert tool.input_schema == schema
        assert tool.server_name == "filesystem"

    def test_to_dict(self):
        """Test converting tool to dictionary."""
        schema = {"type": "object", "properties": {}}
        tool = MCPTool(
            name="test_tool",
            description="Test tool",
            input_schema=schema,
            server_name="test_server",
        )
        data = tool.to_dict()
        assert data["name"] == "test_tool"
        assert data["description"] == "Test tool"
        assert data["input_schema"] == schema
        assert data["server_name"] == "test_server"

    def test_complex_schema(self):
        """Test tool with complex input schema."""
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
                "filters": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["query"],
        }
        tool = MCPTool(
            name="search",
            description="Search with filters",
            input_schema=schema,
            server_name="search_server",
        )
        assert "filters" in tool.input_schema["properties"]
        assert tool.input_schema["required"] == ["query"]


class TestMCPResource:
    """Tests for MCPResource."""

    def test_creation_minimal(self):
        """Test creating minimal resource."""
        resource = MCPResource(uri="file:///test.txt", name="test.txt", description="Test file")
        assert resource.uri == "file:///test.txt"
        assert resource.name == "test.txt"
        assert resource.description == "Test file"
        assert resource.mime_type is None
        assert resource.server_name == ""

    def test_creation_with_mime_type(self):
        """Test creating resource with MIME type."""
        resource = MCPResource(
            uri="file:///document.pdf",
            name="document.pdf",
            description="PDF document",
            mime_type="application/pdf",
        )
        assert resource.mime_type == "application/pdf"

    def test_creation_with_server_name(self):
        """Test creating resource with server name."""
        resource = MCPResource(
            uri="db://users/123",
            name="User 123",
            description="User record",
            server_name="database_server",
        )
        assert resource.server_name == "database_server"

    def test_to_dict(self):
        """Test converting resource to dictionary."""
        resource = MCPResource(
            uri="http://example.com/api/data",
            name="API Data",
            description="Data from API",
            mime_type="application/json",
            server_name="api_server",
        )
        data = resource.to_dict()
        assert data["uri"] == "http://example.com/api/data"
        assert data["name"] == "API Data"
        assert data["description"] == "Data from API"
        assert data["mime_type"] == "application/json"
        assert data["server_name"] == "api_server"

    def test_various_uri_schemes(self):
        """Test resources with different URI schemes."""
        file_resource = MCPResource(
            uri="file:///path/to/file", name="file", description="File resource"
        )
        http_resource = MCPResource(
            uri="http://example.com/resource", name="http", description="HTTP resource"
        )
        custom_resource = MCPResource(
            uri="custom://resource/id", name="custom", description="Custom resource"
        )
        assert file_resource.uri.startswith("file://")
        assert http_resource.uri.startswith("http://")
        assert custom_resource.uri.startswith("custom://")


class TestMCPPrompt:
    """Tests for MCPPrompt."""

    def test_creation(self):
        """Test creating MCP prompt."""
        prompt = MCPPrompt(name="code_review", description="Review code for issues", arguments=[])
        assert prompt.name == "code_review"
        assert prompt.description == "Review code for issues"
        assert prompt.arguments == []

    def test_creation_with_arguments(self):
        """Test creating prompt with arguments."""
        args = [
            {"name": "code", "type": "string", "required": True},
            {"name": "language", "type": "string", "required": False},
        ]
        prompt = MCPPrompt(name="analyze", description="Analyze code", arguments=args)
        assert len(prompt.arguments) == 2
        assert prompt.arguments[0]["name"] == "code"

    def test_to_dict(self):
        """Test converting prompt to dictionary."""
        prompt = MCPPrompt(
            name="test_prompt",
            description="Test",
            arguments=[{"name": "arg1", "type": "string"}],
            server_name="test_server",
        )
        data = prompt.to_dict()
        assert data["name"] == "test_prompt"
        assert data["arguments"] == [{"name": "arg1", "type": "string"}]

    def test_prompt_names(self):
        """Test various prompt names."""
        prompts = [
            MCPPrompt(name="summarize", description="Summarize text", arguments=[]),
            MCPPrompt(name="translate", description="Translate text", arguments=[]),
            MCPPrompt(name="explain", description="Explain concept", arguments=[]),
        ]
        assert len(prompts) == 3
        assert prompts[0].name == "summarize"
        assert prompts[1].name == "translate"
        assert prompts[2].name == "explain"


class TestMCPIntegration:
    """Integration tests for MCP components."""

    def test_server_with_multiple_tools(self):
        """Test server configuration with multiple tools."""
        config = MCPServerConfig(
            name="filesystem_server", command="python", args=["-m", "mcp_filesystem"]
        )

        tools = [
            MCPTool(
                name="read_file",
                description="Read file",
                input_schema={"type": "object"},
                server_name=config.name,
            ),
            MCPTool(
                name="write_file",
                description="Write file",
                input_schema={"type": "object"},
                server_name=config.name,
            ),
        ]

        assert all(t.server_name == config.name for t in tools)
        assert len(tools) == 2

    def test_server_with_resources(self):
        """Test server with resources."""
        config = MCPServerConfig(name="docs_server", command="python", args=["-m", "mcp_docs"])

        resources = [
            MCPResource(
                uri="docs://api/reference",
                name="API Reference",
                description="API documentation",
                server_name=config.name,
            ),
            MCPResource(
                uri="docs://guides/quickstart",
                name="Quickstart",
                description="Getting started guide",
                server_name=config.name,
            ),
        ]

        assert all(r.server_name == config.name for r in resources)

    def test_complete_server_setup(self):
        """Test complete server setup with all components."""
        config = MCPServerConfig(
            name="complete_server",
            command="node",
            args=["server.js"],
            env={"PORT": "8000"},
            transport=MCPTransport.HTTP,
            url="http://localhost:8000",
            timeout=45.0,
        )

        tool = MCPTool(
            name="process_data",
            description="Process data",
            input_schema={"type": "object"},
            server_name=config.name,
        )

        resource = MCPResource(
            uri="data://processed/results",
            name="Results",
            description="Processed results",
            server_name=config.name,
        )

        prompt = MCPPrompt(name="analyze", description="Analyze data", arguments=[])

        # Verify all components are properly configured
        assert config.transport == MCPTransport.HTTP
        assert tool.server_name == config.name
        assert resource.server_name == config.name
        assert prompt.name == "analyze"
