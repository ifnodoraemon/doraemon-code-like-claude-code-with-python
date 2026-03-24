"""
Unit tests for unified_config.py

Tests configuration loading, validation, and precedence.
"""

import json
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.core.unified_config import UnifiedConfig

BASE_CONFIG = {"model": "test-model"}


class TestUnifiedConfig:
    """Tests for UnifiedConfig class."""

    def test_default_values(self):
        """Test that required and default values are set correctly."""
        config = UnifiedConfig(**BASE_CONFIG)

        assert config.model == "test-model"
        assert config.temperature == 0.7
        assert config.max_context_tokens == 100_000
        assert config.max_tool_steps == 15
        assert config.checkpoint_enabled is True

    def test_model_is_required(self):
        """Test that model must be configured explicitly."""
        with pytest.raises(ValidationError):
            UnifiedConfig()

    def test_custom_values(self):
        """Test setting custom values."""
        config = UnifiedConfig(model="custom-model", temperature=0.5, max_tool_steps=20)

        assert config.model == "custom-model"
        assert config.temperature == 0.5
        assert config.max_tool_steps == 20

    def test_temperature_validation(self):
        """Test that temperature is validated."""
        # Valid temperatures
        UnifiedConfig(model="test-model", temperature=0.0)
        UnifiedConfig(model="test-model", temperature=1.0)
        UnifiedConfig(model="test-model", temperature=2.0)

        # Invalid temperatures
        with pytest.raises(ValidationError):
            UnifiedConfig(model="test-model", temperature=-0.1)

        with pytest.raises(ValidationError):
            UnifiedConfig(model="test-model", temperature=2.1)

    def test_positive_value_validation(self):
        """Test that positive values are validated."""
        # Valid
        UnifiedConfig(model="test-model", max_context_tokens=1000)
        UnifiedConfig(model="test-model", max_tool_steps=1)

        # Invalid
        with pytest.raises(ValidationError):
            UnifiedConfig(model="test-model", max_context_tokens=0)

        with pytest.raises(ValidationError):
            UnifiedConfig(model="test-model", max_tool_steps=0)

    def test_log_level_validation(self):
        """Test that log level is validated."""
        # Valid
        UnifiedConfig(model="test-model", log_level="DEBUG")
        UnifiedConfig(model="test-model", log_level="INFO")
        UnifiedConfig(model="test-model", log_level="WARNING")
        UnifiedConfig(model="test-model", log_level="ERROR")

        # Invalid
        with pytest.raises(ValidationError):
            UnifiedConfig(model="test-model", log_level="INVALID")

    def test_from_env_and_file_requires_model(self, tmp_path):
        """Test loading without a configured model fails."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValidationError):
                UnifiedConfig.from_env_and_file(tmp_path / ".agent" / "config.json")

    def test_from_env_and_file_with_file(self, tmp_path):
        """Test loading from config file."""
        config_file = tmp_path / ".agent" / "config.json"
        config_file.parent.mkdir(parents=True)

        config_data = {"model": "file-model", "max_tool_steps": 25}

        with open(config_file, "w") as f:
            json.dump(config_data, f)

        config = UnifiedConfig.from_env_and_file(config_file)

        assert config.model == "file-model"
        assert config.max_tool_steps == 25

    def test_from_env_and_file_does_not_override_model_from_env(self, tmp_path):
        """Test that model must come from the config file."""
        config_file = tmp_path / ".agent" / "config.json"
        config_file.parent.mkdir(parents=True)

        config_data = {"model": "file-model"}
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        with patch.dict("os.environ", {"AGENT_MODEL": "env-model"}):
            config = UnifiedConfig.from_env_and_file(config_file)
            assert config.model == "file-model"

    def test_to_file(self, tmp_path):
        """Test saving config to file."""
        config = UnifiedConfig(model="test-model", max_tool_steps=30)

        config_file = tmp_path / "test_config.json"
        config.to_file(config_file)

        assert config_file.exists()

        with open(config_file) as f:
            saved_data = json.load(f)

        assert saved_data["model"] == "test-model"
        assert saved_data["max_tool_steps"] == 30


class TestConfigHelpers:
    """Tests for config helper functions."""

    def test_parse_int_valid(self):
        """Test parsing valid integers."""
        from src.core.unified_config import _parse_int

        assert _parse_int("123") == 123
        assert _parse_int("0") == 0
        assert _parse_int("-5") == -5

    def test_parse_int_invalid(self):
        """Test parsing invalid integers."""
        from src.core.unified_config import _parse_int

        assert _parse_int(None) is None
        assert _parse_int("abc") is None
        assert _parse_int("12.5") is None

    def test_parse_float_valid(self):
        """Test parsing valid floats."""
        from src.core.unified_config import _parse_float

        assert _parse_float("1.5") == 1.5
        assert _parse_float("0.0") == 0.0
        assert _parse_float("-2.5") == -2.5

    def test_parse_float_invalid(self):
        """Test parsing invalid floats."""
        from src.core.unified_config import _parse_float

        assert _parse_float(None) is None
        assert _parse_float("abc") is None

    def test_parse_bool_valid(self):
        """Test parsing valid booleans."""
        from src.core.unified_config import _parse_bool

        assert _parse_bool("true") is True
        assert _parse_bool("True") is True
        assert _parse_bool("1") is True
        assert _parse_bool("yes") is True
        assert _parse_bool("on") is True

        assert _parse_bool("false") is False
        assert _parse_bool("False") is False
        assert _parse_bool("0") is False
        assert _parse_bool("no") is False
        assert _parse_bool("off") is False

    def test_parse_bool_invalid(self):
        """Test parsing invalid booleans."""
        from src.core.unified_config import _parse_bool

        assert _parse_bool(None) is None
        assert _parse_bool("maybe") is None
