"""Real protocol integration tests against a live OpenAI/Anthropic-compatible endpoint."""

import importlib
import os

import anthropic
import openai
import pytest
from fastapi.testclient import TestClient

from src.core.llm.model_utils import normalize_anthropic_base_url


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        pytest.skip(f"{name} is not configured")
    return value


def _reload_gateway_server():
    """Reload the gateway module after environment changes."""
    import src.gateway.server as gateway_server

    return importlib.reload(gateway_server)


@pytest.mark.integration
def test_real_openai_protocol_upstream():
    """Hit the real upstream OpenAI-compatible endpoint without mocks."""
    base_url = _require_env("REAL_API_BASE").rstrip("/")
    api_key = _require_env("REAL_API_KEY")
    model = _require_env("REAL_MODEL")

    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    payload = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "reply with pong"}],
        max_tokens=64,
        temperature=0,
    )

    assert payload.model == model
    assert payload.choices
    assert payload.choices[0].message.content


@pytest.mark.integration
def test_real_anthropic_protocol_upstream():
    """Hit the real upstream Anthropic-compatible endpoint without mocks."""
    base_url = _require_env("REAL_API_BASE").rstrip("/")
    api_key = _require_env("REAL_API_KEY")
    model = _require_env("REAL_MODEL")

    client = anthropic.Anthropic(
        api_key=api_key,
        base_url=normalize_anthropic_base_url(base_url),
    )
    payload = client.messages.create(
        model=model,
        messages=[{"role": "user", "content": "reply with pong"}],
        max_tokens=64,
        temperature=0,
    )

    assert payload.model == model
    assert payload.type == "message"
    assert payload.content


@pytest.mark.integration
def test_real_gateway_openai_and_anthropic_routes():
    """Exercise the local gateway app against the live upstream in both protocols."""
    base_url = _require_env("REAL_API_BASE").rstrip("/")
    api_key = _require_env("REAL_API_KEY")
    model = _require_env("REAL_MODEL")

    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = base_url
    os.environ["ANTHROPIC_API_KEY"] = api_key
    os.environ["ANTHROPIC_API_BASE"] = base_url
    os.environ["AGENT_API_KEY"] = api_key

    gateway_server = _reload_gateway_server()

    with TestClient(gateway_server.app) as client:
        openai_resp = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "reply with pong"}],
                "max_tokens": 64,
                "temperature": 0,
            },
        )
        openai_resp.raise_for_status()
        openai_payload = openai_resp.json()

        anthropic_resp = client.post(
            "/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": "reply with pong"}],
                "max_tokens": 64,
                "temperature": 0,
            },
        )
        anthropic_resp.raise_for_status()
        anthropic_payload = anthropic_resp.json()

    assert openai_payload["model"] == model
    assert openai_payload["choices"]
    assert anthropic_payload["model"] == model
    assert anthropic_payload["type"] == "message"
