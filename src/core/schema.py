"""Configuration schema validation using Pydantic."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ServerConfig(BaseModel):
    """MCP Server configuration."""

    command: str = Field(..., description="Command to run the server")
    args: list[str] = Field(default_factory=list, description="Command arguments")
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables")

    @field_validator("command")
    @classmethod
    def validate_command(cls, v):
        """Ensure command is not empty."""
        if not v or not v.strip():
            raise ValueError("Command cannot be empty")
        return v.strip()


class PersonaConfig(BaseModel):
    """Agent persona configuration."""

    name: str = Field(default="Doraemon", description="Agent name")
    role: str = Field(default="Generalist AI Assistant", description="Agent role")

    @field_validator("name", "role")
    @classmethod
    def validate_not_empty(cls, v):
        """Ensure fields are not empty."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


class DoraemonConfig(BaseModel):
    """Main Doraemon configuration."""

    mcpServers: dict[str, ServerConfig] = Field(
        ..., alias="mcpServers", description="MCP servers configuration"
    )
    persona: PersonaConfig | None = Field(default=None, description="Agent persona configuration")
    sensitive_tools: list[str] = Field(
        default_factory=lambda: [
            "execute_python",
            "write_file",
            "save_note",
            "move_file",
            "delete_file",
        ],
        description="List of tools that require user approval",
    )
    instructions: list[str] = Field(
        default_factory=list, description="Additional instruction files to load (supports globs)"
    )

    model_config = ConfigDict(populate_by_name=True)  # Allow both snake_case and camelCase

    @model_validator(mode="after")
    def validate_required_servers(self):
        """Ensure required servers are configured."""
        required_servers = {
            "memory": "Long-term memory server",
            "fs_read": "File reading server",
            "fs_write": "File writing server",
            "fs_edit": "File editing server",
            "fs_ops": "File operations server",
        }

        missing = []
        for server_name, description in required_servers.items():
            if server_name not in self.mcpServers:
                missing.append(f"{server_name} ({description})")

        if missing:
            raise ValueError(f"Missing required servers: {', '.join(missing)}")

        return self

    @field_validator("sensitive_tools")
    @classmethod
    def validate_sensitive_tools(cls, v):
        """Ensure sensitive tools list is not empty."""
        if not v:
            raise ValueError("Sensitive tools list cannot be empty")
        return v


def validate_config_file(config_path: Path) -> DoraemonConfig:
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
        config = DoraemonConfig.model_validate(config_data)
        return config
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}") from e


def get_default_config() -> dict:
    """Return the default configuration structure."""
    return {
        "mcpServers": {
            "memory": {"command": "python3", "args": ["src/servers/memory.py"], "env": {}},
            "fs_read": {
                "command": "python3",
                "args": ["src/servers/fs_read.py"],
                "env": {"VISION_PROVIDER": "google", "VISION_MODEL": "gemini-1.5-flash"},
            },
            "fs_write": {"command": "python3", "args": ["src/servers/fs_write.py"], "env": {}},
            "fs_edit": {"command": "python3", "args": ["src/servers/fs_edit.py"], "env": {}},
            "fs_ops": {"command": "python3", "args": ["src/servers/fs_ops.py"], "env": {}},
            "web": {"command": "python3", "args": ["src/servers/web.py"], "env": {}},
            "computer": {"command": "python3", "args": ["src/servers/computer.py"], "env": {}},
            "task": {"command": "python3", "args": ["src/servers/task.py"], "env": {}},
        },
        "persona": {"name": "Doraemon", "role": "Generalist AI Assistant & Coder"},
        "sensitive_tools": [
            "execute_python",
            "write_file",
            "save_note",
            "move_file",
            "delete_file",
        ],
    }
