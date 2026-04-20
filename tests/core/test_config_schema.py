import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.core.config.schema import (
    AgentConfig,
    MCPServerConfig,
    PersonaConfig,
    get_default_config,
    validate_config_file,
)


class TestPersonaConfig:
    def test_defaults(self):
        p = PersonaConfig()
        assert p.name == "Agent"
        assert p.role == "Generalist AI Assistant"

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            PersonaConfig(name="  ", role="test")

    def test_strips_whitespace(self):
        p = PersonaConfig(name="  Agent  ", role="  Role  ")
        assert p.name == "Agent"
        assert p.role == "Role"


class TestMCPServerConfig:
    def test_valid_streamable_http(self):
        cfg = MCPServerConfig(name="test", url="http://localhost:8080")
        assert cfg.transport == "streamable_http"

    def test_valid_stdio(self):
        cfg = MCPServerConfig(name="test", transport="stdio", command="npx")
        assert cfg.command == "npx"

    def test_missing_url_for_http(self):
        with pytest.raises(ValidationError, match="url is required"):
            MCPServerConfig(name="test", transport="streamable_http")

    def test_missing_command_for_stdio(self):
        with pytest.raises(ValidationError, match="command is required"):
            MCPServerConfig(name="test", transport="stdio")

    def test_invalid_transport(self):
        with pytest.raises(ValidationError):
            MCPServerConfig(name="test", transport="grpc")

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            MCPServerConfig(name="  ")

    def test_optional_strings_stripped(self):
        cfg = MCPServerConfig(name="test", url="  http://x  ")
        assert cfg.url == "http://x"

    def test_optional_strings_none(self):
        cfg = MCPServerConfig(name="test", url="http://x")
        assert cfg.command is None
        assert cfg.cwd is None

    def test_timeout_positive(self):
        cfg = MCPServerConfig(name="test", url="http://x", timeout_seconds=60)
        assert cfg.timeout_seconds == 60

    def test_timeout_zero_raises(self):
        with pytest.raises(ValidationError):
            MCPServerConfig(name="test", url="http://x", timeout_seconds=0)


class TestAgentConfig:
    def test_minimal_valid(self):
        cfg = AgentConfig(model="gemini-3-pro")
        assert cfg.model == "gemini-3-pro"

    def test_empty_model_raises(self):
        with pytest.raises(ValidationError):
            AgentConfig(model="  ")

    def test_sensitive_tools_default(self):
        cfg = AgentConfig(model="gemini-3-pro")
        assert "write" in cfg.sensitive_tools
        assert "run" in cfg.sensitive_tools

    def test_sensitive_tools_empty_raises(self):
        with pytest.raises(ValidationError):
            AgentConfig(model="gemini-3-pro", sensitive_tools=[])

    def test_mcp_servers(self):
        cfg = AgentConfig(
            model="gemini-3-pro",
            mcp_servers=[
                MCPServerConfig(name="s1", url="http://localhost:8080"),
            ],
        )
        assert len(cfg.mcp_servers) == 1

    def test_populate_by_name(self):
        cfg = AgentConfig(model="gemini-3-pro", persona=PersonaConfig())
        assert cfg.persona is not None


class TestValidateConfigFile:
    def test_valid_file(self, tmp_path):
        config = {"model": "gemini-3-pro"}
        f = tmp_path / "config.json"
        f.write_text(json.dumps(config))
        result = validate_config_file(f)
        assert result.model == "gemini-3-pro"

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            validate_config_file(tmp_path / "missing.json")

    def test_invalid_config(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text(json.dumps({"model": ""}))
        with pytest.raises(ValueError, match="Invalid configuration"):
            validate_config_file(f)


class TestGetDefaultConfig:
    def test_structure(self):
        cfg = get_default_config()
        assert "persona" in cfg
        assert "sensitive_tools" in cfg
        assert cfg["persona"]["name"] == "Agent"
