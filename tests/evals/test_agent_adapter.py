import os
from types import SimpleNamespace

import pytest

from tests.evals.agent_adapter import DoraemonAgentAdapter


class DummyModelClient:
    @classmethod
    async def create(cls, config):
        return cls()

    async def close(self):
        return None


class DummyTrace:
    events = []


class DummySession:
    def __init__(self, *args, **kwargs):
        pass

    async def initialize(self):
        return None

    async def turn(self, prompt):
        return SimpleNamespace(success=True, error=None, response="ok", tokens_used=0)

    def get_trace(self):
        return DummyTrace()

    def save_trace(self):
        return None

    async def aclose(self):
        return None

    def close(self):
        return None


@pytest.mark.asyncio
async def test_async_execute_uses_google_provider_for_gemini(monkeypatch):
    adapter = DoraemonAgentAdapter()
    adapter._initialized = True
    adapter._agent_session_cls = DummySession
    adapter._model_client_cls = DummyModelClient
    captured = {}

    class DummyClientConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    adapter._client_config_cls = DummyClientConfig

    monkeypatch.setenv("REAL_API_BASE", "https://unused.example/v1")
    monkeypatch.setenv("REAL_API_KEY", "google-key")
    monkeypatch.setenv("REAL_MODEL", "gemini-3.1-pro-preview")

    try:
        result = await adapter._async_execute("ping")
    finally:
        monkeypatch.delenv("REAL_API_BASE", raising=False)
        monkeypatch.delenv("REAL_API_KEY", raising=False)
        monkeypatch.delenv("REAL_MODEL", raising=False)

    assert result["success"] is True
    assert captured["model"] == "gemini-3.1-pro-preview"
    assert captured["google_api_key"] == "google-key"
    assert "openai_api_key" not in captured
    assert "anthropic_api_key" not in captured
