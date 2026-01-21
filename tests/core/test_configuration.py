"""
Unit tests for the Configuration Management System.

Tests configuration building, merging, type-safe getters, and validation.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from src.core.configuration import (
    Configuration,
    ConfigurationBuilder,
    configure,
    get_configuration,
)


class TestConfiguration:
    """Tests for Configuration class"""

    def test_get_simple_value(self):
        """Test getting a simple top-level value"""
        config = Configuration({"name": "Polymath", "version": "0.4.0"})

        assert config.get("name") == "Polymath"
        assert config.get("version") == "0.4.0"

    def test_get_nested_value(self):
        """Test getting nested values using dot notation"""
        config = Configuration({
            "database": {
                "host": "localhost",
                "port": 5432,
                "credentials": {
                    "username": "admin"
                }
            }
        })

        assert config.get("database.host") == "localhost"
        assert config.get("database.port") == 5432
        assert config.get("database.credentials.username") == "admin"

    def test_get_with_default(self):
        """Test getting a value with default when key doesn't exist"""
        config = Configuration({"existing": "value"})

        assert config.get("missing") is None
        assert config.get("missing", "default") == "default"
        assert config.get("deep.missing.key", "fallback") == "fallback"

    def test_get_required(self):
        """Test get_required raises for missing values"""
        config = Configuration({"existing": "value"})

        assert config.get_required("existing") == "value"

        with pytest.raises(ValueError, match="Required configuration not found"):
            config.get_required("missing")

    def test_get_str(self):
        """Test type-safe string getter"""
        config = Configuration({
            "string": "hello",
            "number": 42,
            "boolean": True
        })

        assert config.get_str("string") == "hello"
        assert config.get_str("number") == "42"
        assert config.get_str("boolean") == "True"
        assert config.get_str("missing") == ""
        assert config.get_str("missing", "default") == "default"

    def test_get_int(self):
        """Test type-safe integer getter"""
        config = Configuration({
            "integer": 42,
            "string_int": "100",
            "float": 3.14,
            "invalid": "not a number"
        })

        assert config.get_int("integer") == 42
        assert config.get_int("string_int") == 100
        assert config.get_int("float") == 3
        assert config.get_int("invalid") == 0
        assert config.get_int("missing") == 0
        assert config.get_int("missing", 99) == 99

    def test_get_bool(self):
        """Test type-safe boolean getter"""
        config = Configuration({
            "bool_true": True,
            "bool_false": False,
            "string_true": "true",
            "string_yes": "yes",
            "string_1": "1",
            "string_false": "false",
            "number_1": 1,
            "number_0": 0
        })

        assert config.get_bool("bool_true") is True
        assert config.get_bool("bool_false") is False
        assert config.get_bool("string_true") is True
        assert config.get_bool("string_yes") is True
        assert config.get_bool("string_1") is True
        assert config.get_bool("string_false") is False
        assert config.get_bool("number_1") is True
        assert config.get_bool("number_0") is False
        assert config.get_bool("missing") is False
        assert config.get_bool("missing", True) is True

    def test_get_list(self):
        """Test type-safe list getter"""
        config = Configuration({
            "list": [1, 2, 3],
            "single": "item"
        })

        assert config.get_list("list") == [1, 2, 3]
        assert config.get_list("single") == ["item"]
        assert config.get_list("missing") == []
        assert config.get_list("missing", ["default"]) == ["default"]

    def test_set_value(self):
        """Test setting values at runtime"""
        config = Configuration({"existing": "old"})

        config.set("existing", "new")
        config.set("new_key", "value")
        config.set("nested.deep.key", "deep_value")

        assert config.get("existing") == "new"
        assert config.get("new_key") == "value"
        assert config.get("nested.deep.key") == "deep_value"

    def test_has(self):
        """Test checking if path exists"""
        config = Configuration({
            "existing": "value",
            "nested": {"key": "value"}
        })

        assert config.has("existing") is True
        assert config.has("nested.key") is True
        assert config.has("missing") is False
        assert config.has("nested.missing") is False

    def test_get_section(self):
        """Test getting a configuration section"""
        config = Configuration({
            "database": {
                "host": "localhost",
                "port": 5432
            }
        })

        db_config = config.get_section("database")

        assert isinstance(db_config, Configuration)
        assert db_config.get("host") == "localhost"
        assert db_config.get("port") == 5432

    def test_to_dict(self):
        """Test exporting configuration as dictionary"""
        data = {"key": "value", "nested": {"inner": "data"}}
        config = Configuration(data)

        exported = config.to_dict()

        assert exported == data
        # Should be a copy
        exported["key"] = "modified"
        assert config.get("key") == "value"

    def test_validate_required(self):
        """Test validation of required fields"""
        config = Configuration({"present": "value"})

        schema = {
            "present": {"required": True},
            "missing": {"required": True}
        }

        errors = config.validate(schema)

        assert len(errors) == 1
        assert "missing" in errors[0]

    def test_validate_type(self):
        """Test validation of field types"""
        config = Configuration({
            "string": "hello",
            "number": "not a number"
        })

        schema = {
            "string": {"type": str},
            "number": {"type": int}
        }

        errors = config.validate(schema)

        assert len(errors) == 1
        assert "number" in errors[0]
        assert "int" in errors[0]

    def test_validate_min_max(self):
        """Test validation of numeric ranges"""
        config = Configuration({
            "too_small": 0,
            "too_large": 100,
            "just_right": 50
        })

        schema = {
            "too_small": {"type": int, "min": 1},
            "too_large": {"type": int, "max": 99},
            "just_right": {"type": int, "min": 1, "max": 99}
        }

        errors = config.validate(schema)

        assert len(errors) == 2

    def test_validate_enum(self):
        """Test validation of enum values"""
        config = Configuration({
            "valid": "option1",
            "invalid": "option3"
        })

        schema = {
            "valid": {"enum": ["option1", "option2"]},
            "invalid": {"enum": ["option1", "option2"]}
        }

        errors = config.validate(schema)

        assert len(errors) == 1
        assert "invalid" in errors[0]


class TestConfigurationBuilder:
    """Tests for ConfigurationBuilder class"""

    def test_add_defaults(self):
        """Test adding default values"""
        config = (
            ConfigurationBuilder()
            .add_defaults({"debug": False, "log_level": "INFO"})
            .build()
        )

        assert config.get("debug") is False
        assert config.get("log_level") == "INFO"

    def test_add_json_file(self):
        """Test loading configuration from JSON file"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"name": "TestApp", "port": 8080}, f)
            temp_path = f.name

        try:
            config = (
                ConfigurationBuilder()
                .add_json_file(temp_path)
                .build()
            )

            assert config.get("name") == "TestApp"
            assert config.get("port") == 8080
        finally:
            os.unlink(temp_path)

    def test_add_json_file_optional_missing(self):
        """Test that optional missing file doesn't raise error"""
        config = (
            ConfigurationBuilder()
            .add_defaults({"default": "value"})
            .add_json_file("/nonexistent/path.json", optional=True)
            .build()
        )

        assert config.get("default") == "value"

    def test_add_json_file_required_missing(self):
        """Test that required missing file raises error"""
        builder = ConfigurationBuilder().add_json_file("/nonexistent/path.json")

        with pytest.raises(FileNotFoundError):
            builder.build()

    def test_add_environment_variables(self):
        """Test loading configuration from environment variables"""
        # Set test environment variables
        os.environ["TEST_APP_NAME"] = "EnvApp"
        os.environ["TEST_APP_PORT"] = "9000"
        os.environ["TEST_APP_DEBUG"] = "true"

        try:
            config = (
                ConfigurationBuilder()
                .add_environment_variables(prefix="TEST_APP_")
                .build()
            )

            assert config.get("name") == "EnvApp"
            assert config.get("port") == 9000
            assert config.get("debug") is True
        finally:
            del os.environ["TEST_APP_NAME"]
            del os.environ["TEST_APP_PORT"]
            del os.environ["TEST_APP_DEBUG"]

    def test_add_environment_variables_nested(self):
        """Test loading nested configuration from environment variables"""
        os.environ["TEST_DB__HOST"] = "localhost"
        os.environ["TEST_DB__PORT"] = "5432"

        try:
            config = (
                ConfigurationBuilder()
                .add_environment_variables(prefix="TEST_")
                .build()
            )

            assert config.get("db.host") == "localhost"
            assert config.get("db.port") == 5432
        finally:
            del os.environ["TEST_DB__HOST"]
            del os.environ["TEST_DB__PORT"]

    def test_source_priority(self):
        """Test that later sources override earlier ones"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"name": "FileApp", "port": 8000}, f)
            temp_path = f.name

        os.environ["PRIORITY_TEST_PORT"] = "9000"

        try:
            config = (
                ConfigurationBuilder()
                .add_defaults({"name": "DefaultApp", "port": 7000, "debug": False})
                .add_json_file(temp_path)
                .add_environment_variables(prefix="PRIORITY_TEST_")
                .build()
            )

            # Name from file (overrides default)
            assert config.get("name") == "FileApp"
            # Port from env (overrides file and default)
            assert config.get("port") == 9000
            # Debug from default (not overridden)
            assert config.get("debug") is False
        finally:
            os.unlink(temp_path)
            del os.environ["PRIORITY_TEST_PORT"]

    def test_deep_merge(self):
        """Test deep merging of nested configurations"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({
                "database": {
                    "port": 5432,
                    "ssl": True
                }
            }, f)
            temp_path = f.name

        try:
            config = (
                ConfigurationBuilder()
                .add_defaults({
                    "database": {
                        "host": "localhost",
                        "port": 3306,
                        "name": "mydb"
                    }
                })
                .add_json_file(temp_path)
                .build()
            )

            # From defaults
            assert config.get("database.host") == "localhost"
            assert config.get("database.name") == "mydb"
            # From file (overrides default)
            assert config.get("database.port") == 5432
            # From file (added)
            assert config.get("database.ssl") is True
        finally:
            os.unlink(temp_path)


class TestGlobalFunctions:
    """Tests for global configuration functions"""

    def test_configure_and_get_configuration(self):
        """Test configure function and get_configuration"""

        def setup(builder: ConfigurationBuilder):
            builder.add_defaults({"app": "Polymath", "version": "1.0"})

        config = configure(setup)

        assert config.get("app") == "Polymath"

        # Should be accessible globally
        global_config = get_configuration()
        assert global_config.get("app") == "Polymath"
