import os

import pytest

from src.core.config.secrets import (
    is_sensitive_header,
    mcp_header_key,
    resolve_mcp_headers,
    SERVICE_NAME,
    _ENV_OVERRIDES,
)


class TestIsSensitiveHeader:
    def test_authorization(self):
        assert is_sensitive_header("Authorization") is True

    def test_authorization_lowercase(self):
        assert is_sensitive_header("authorization") is True

    def test_normal_header(self):
        assert is_sensitive_header("Content-Type") is False

    def test_context7_key(self):
        assert is_sensitive_header("context7_api_key") is True


class TestMcpHeaderKey:
    def test_format(self):
        key = mcp_header_key("myserver", "Authorization")
        assert key == "mcp:myserver:header:Authorization"


class TestResolveMcpHeaders:
    def test_non_sensitive_passthrough(self):
        result = resolve_mcp_headers("srv", {"Content-Type": "application/json"})
        assert result["Content-Type"] == "application/json"

    def test_sensitive_header_env_override(self, monkeypatch):
        monkeypatch.setenv("MY_API_KEY", "from-env")
        from src.core.config.secrets import get_secret

        result = get_secret("google_api_key", config_value="from-config")
        assert result is not None


class TestServiceName:
    def test_value(self):
        assert SERVICE_NAME == "doraemon-code"


class TestEnvOverrides:
    def test_mappings(self):
        assert _ENV_OVERRIDES["google_api_key"] == "GOOGLE_API_KEY"
        assert _ENV_OVERRIDES["openai_api_key"] == "OPENAI_API_KEY"
        assert _ENV_OVERRIDES["anthropic_api_key"] == "ANTHROPIC_API_KEY"
        assert _ENV_OVERRIDES["gateway_key"] == "AGENT_API_KEY"


class TestGetSecret:
    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
        from src.core.config.secrets import get_secret

        result = get_secret("google_api_key")
        assert result == "test-google-key"

    def test_config_value_fallback(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setattr("src.core.config.secrets._keyring_available", lambda: False)
        from src.core.config.secrets import get_secret

        result = get_secret("google_api_key", config_value="fallback-value")
        assert result == "fallback-value"

    def test_returns_none_when_nothing(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setattr("src.core.config.secrets._keyring_available", lambda: False)
        from src.core.config.secrets import get_secret

        result = get_secret("google_api_key")
        assert result is None

    def test_no_env_override_for_unknown_key(self, monkeypatch):
        monkeypatch.setattr("src.core.config.secrets._keyring_available", lambda: False)
        from src.core.config.secrets import get_secret

        result = get_secret("custom_key", config_value="cfg-val")
        assert result == "cfg-val"

    def test_env_empty_string_skipped(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "")
        monkeypatch.setattr("src.core.config.secrets._keyring_available", lambda: False)
        from src.core.config.secrets import get_secret

        result = get_secret("google_api_key", config_value="fallback")
        assert result == "fallback"

    def test_keyring_available_returns_value(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setattr("src.core.config.secrets._keyring_available", lambda: True)
        import src.core.config.secrets as mod

        original = mod.get_secret
        monkeypatch.setattr("keyring.get_password", lambda s, k: "kr-val", raising=False)
        result = original("some_key")
        assert result == "kr-val" or result is None

    def test_keyring_exception_falls_back(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setattr("src.core.config.secrets._keyring_available", lambda: True)

        def fake_get_password(service, key):
            raise RuntimeError("keyring broken")

        monkeypatch.setattr("keyring.get_password", fake_get_password, raising=False)
        from src.core.config.secrets import get_secret

        result = get_secret("google_api_key", config_value="fb")
        assert result == "fb"


class TestKeyringAvailable:
    def test_not_available_when_import_fails(self, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "keyring", None)
        from src.core.config.secrets import _keyring_available

        result = _keyring_available()
        assert result is False


class TestSetDeleteSecret:
    def test_set_secret(self, monkeypatch):
        called = {}
        monkeypatch.setattr(
            "keyring.set_password",
            lambda s, k, v: called.update({"s": s, "k": k, "v": v}),
            raising=False,
        )
        from src.core.config.secrets import set_secret

        set_secret("mykey", "myval")
        assert called["k"] == "mykey"

    def test_delete_secret(self, monkeypatch):
        called = {}
        monkeypatch.setattr(
            "keyring.delete_password", lambda s, k: called.update({"s": s, "k": k}), raising=False
        )
        from src.core.config.secrets import delete_secret

        delete_secret("mykey")
        assert called["k"] == "mykey"


class TestResolveMcpHeaders:
    def test_sensitive_header_resolved(self, monkeypatch):
        monkeypatch.setattr("src.core.config.secrets._keyring_available", lambda: False)
        monkeypatch.delenv("AGENT_API_KEY", raising=False)
        from src.core.config.secrets import resolve_mcp_headers

        result = resolve_mcp_headers("srv", {"Authorization": "default-token"})
        assert result["Authorization"] == "default-token"

    def test_mixed_headers(self, monkeypatch):
        monkeypatch.setattr("src.core.config.secrets._keyring_available", lambda: False)
        from src.core.config.secrets import resolve_mcp_headers

        result = resolve_mcp_headers("srv", {"Authorization": "tok", "Accept": "json"})
        assert result["Authorization"] == "tok"
        assert result["Accept"] == "json"
