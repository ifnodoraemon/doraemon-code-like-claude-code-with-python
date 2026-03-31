"""Configuration schema validation using Pydantic."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PersonaConfig(BaseModel):
    """Agent persona configuration."""

    name: str = Field(default="Agent", description="Agent name")
    role: str = Field(default="Generalist AI Assistant", description="Agent role")

    @field_validator("name", "role")
    @classmethod
    def validate_not_empty(cls, v):
        """Ensure fields are not empty."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


class MCPServerConfig(BaseModel):
    """Remote MCP server configuration."""

    name: str = Field(..., description="Unique MCP server name")
    transport: str = Field(
        default="streamable_http",
        description="MCP transport type: streamable_http or stdio",
    )
    url: str | None = Field(default=None, description="HTTP MCP server URL")
    command: str | None = Field(default=None, description="stdio MCP server command")
    args: list[str] = Field(default_factory=list, description="stdio MCP server arguments")
    env: dict[str, str] = Field(default_factory=dict, description="stdio MCP server environment")
    cwd: str | None = Field(default=None, description="stdio MCP server working directory")
    headers: dict[str, str] = Field(default_factory=dict, description="Optional HTTP headers")
    timeout_seconds: float = Field(default=30.0, gt=0, description="Per-request timeout")
    tool_prefix: str | None = Field(
        default=None,
        description="Optional prefix added to all tools from this MCP server",
    )
    enabled: bool = Field(default=True, description="Whether this server is enabled")

    @field_validator("name", "transport")
    @classmethod
    def validate_required_strings(cls, v: str) -> str:
        """Ensure required string fields are not empty."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()

    @field_validator("transport")
    @classmethod
    def validate_transport(cls, v: str) -> str:
        """Validate supported MCP transport types."""
        normalized = v.strip()
        if normalized not in {"streamable_http", "stdio"}:
            raise ValueError("transport must be 'streamable_http' or 'stdio'")
        return normalized

    @field_validator("url", "command", "cwd")
    @classmethod
    def validate_optional_strings(cls, v: str | None) -> str | None:
        """Normalize optional string fields."""
        if v is None:
            return None
        stripped = v.strip()
        return stripped or None

    @model_validator(mode="after")
    def validate_transport_fields(self) -> "MCPServerConfig":
        """Ensure transport-specific fields are configured."""
        if self.transport == "streamable_http" and not self.url:
            raise ValueError("url is required for streamable_http MCP servers")
        if self.transport == "stdio" and not self.command:
            raise ValueError("command is required for stdio MCP servers")
        return self


class AgentConfig(BaseModel):
    """Main agent configuration."""

    model: str = Field(..., description="Primary model identifier")
    gateway_url: str | None = Field(default=None, description="Gateway server URL")
    gateway_key: str | None = Field(default=None, description="Gateway API key")
    google_api_key: str | None = Field(default=None, description="Google API key")
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    openai_api_base: str | None = Field(
        default=None, description="OpenAI-compatible API base URL"
    )
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key")
    anthropic_api_base: str | None = Field(
        default=None, description="Anthropic-compatible API base URL"
    )
    temperature: float | None = Field(default=None, description="Model temperature override")
    daily_budget_usd: float | None = Field(default=None, description="Daily budget in USD")
    session_budget_usd: float | None = Field(default=None, description="Session budget in USD")
    log_level: str | None = Field(default=None, description="Global log level")
    log_file: str | None = Field(default=None, description="Global log file path")
    persona: PersonaConfig | None = Field(default=None, description="Agent persona configuration")
    sensitive_tools: list[str] = Field(
        default_factory=lambda: [
            "execute_python",
            "write_file",
            "memory_put",
            "move_file",
            "delete_file",
        ],
        description="List of tools that require user approval",
    )
    instructions: list[str] = Field(
        default_factory=list, description="Additional instruction files to load (supports globs)"
    )
    mcp_extensions: list[str] = Field(
        default_factory=list,
        description="Optional extension groups to attach at runtime, e.g. ['browser', 'database']",
    )
    mcp_servers: list[MCPServerConfig] = Field(
        default_factory=list,
        description="Optional remote MCP servers attached at runtime",
    )
    tool_timeouts: dict[str, float] = Field(
        default_factory=dict, description="Timeout overrides for specific tools in seconds"
    )

    model_config = ConfigDict(populate_by_name=True)  # Allow both snake_case and camelCase

    @field_validator("sensitive_tools")
    @classmethod
    def validate_sensitive_tools(cls, v):
        """Ensure sensitive tools list is not empty."""
        if not v:
            raise ValueError("Sensitive tools list cannot be empty")
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v):
        """Ensure the configured model is not empty."""
        if not v or not v.strip():
            raise ValueError("Model cannot be empty")
        return v.strip()


def validate_config_file(config_path: Path) -> AgentConfig:
    """
    Validate a configuration file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Validated configuration

    Raises:
        ValueError: If configuration is invalid
        FileNotFoundError: If configuration file doesn't exist
    """
    import json

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        config_data = json.load(f)

    try:
        config = AgentConfig.model_validate(config_data)
        return config
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}") from e


def get_default_config() -> dict:
    """Return the default configuration structure."""
    return {
        "persona": {"name": "Agent", "role": "Generalist AI Assistant & Coder"},
        "mcp_extensions": [],
        "sensitive_tools": [
            "execute_python",
            "write_file",
            "memory_put",
            "move_file",
            "delete_file",
        ],
    }
