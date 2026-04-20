import base64
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.core.llm.model_utils import (
    ChatResponse,
    ClientConfig,
    ClientMode,
    Message,
    Provider,
    ProviderCapabilities,
    StreamChunk,
    ToolCall,
    ToolDefinition,
    get_content_text,
    is_image_path,
    make_image_part,
    make_text_part,
    normalize_anthropic_base_url,
)


class TestClientMode:
    def test_values(self):
        assert ClientMode.GATEWAY.value == "gateway"
        assert ClientMode.DIRECT.value == "direct"


class TestProvider:
    def test_values(self):
        assert Provider.GOOGLE.value == "google"
        assert Provider.OPENAI.value == "openai"
        assert Provider.ANTHROPIC.value == "anthropic"


class TestProviderCapabilities:
    def test_defaults(self):
        cap = ProviderCapabilities()
        assert cap.tools is True
        assert cap.streaming is True


class TestMessage:
    def test_to_dict_minimal(self):
        msg = Message(role="user", content="hello")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "hello"

    def test_to_dict_full(self):
        msg = Message(
            role="assistant",
            content="response",
            thought="thinking...",
            tool_calls=[{"id": "tc1", "name": "run", "arguments": {}}],
            tool_call_id="tc1",
            name="run",
        )
        d = msg.to_dict()
        assert d["thought"] == "thinking..."
        assert len(d["tool_calls"]) == 1
        assert d["tool_call_id"] == "tc1"

    def test_to_dict_none_content(self):
        msg = Message(role="assistant", content=None)
        d = msg.to_dict()
        assert "content" not in d

    def test_to_dict_list_content(self):
        msg = Message(role="user", content=[{"type": "text", "text": "hello"}])
        d = msg.to_dict()
        assert isinstance(d["content"], list)


class TestToolDefinition:
    def test_to_openai_format(self):
        tool = ToolDefinition(
            name="get_weather",
            description="Get weather",
            parameters={"type": "object", "properties": {"loc": {"type": "string"}}},
        )
        d = tool.to_openai_format()
        assert d["type"] == "function"
        assert d["function"]["name"] == "get_weather"


class TestChatResponse:
    def test_has_tool_calls_true(self):
        resp = ChatResponse(tool_calls=[{"name": "run"}])
        assert resp.has_tool_calls is True

    def test_has_tool_calls_false(self):
        resp = ChatResponse(content="hello")
        assert resp.has_tool_calls is False

    def test_has_tool_calls_empty_list(self):
        resp = ChatResponse(tool_calls=[])
        assert resp.has_tool_calls is False


class TestStreamChunk:
    def test_defaults(self):
        chunk = StreamChunk()
        assert chunk.content is None
        assert chunk.finish_reason is None


class TestToolCall:
    def test_to_dict(self):
        tc = ToolCall(id="tc1", name="run", arguments={"cmd": "ls"})
        d = tc.to_dict()
        assert d["id"] == "tc1"
        assert d["name"] == "run"

    def test_from_dict(self):
        data = {"id": "tc1", "name": "run", "arguments": {"cmd": "ls"}}
        tc = ToolCall.from_dict(data)
        assert tc.id == "tc1"
        assert tc.name == "run"

    def test_from_dict_missing_fields(self):
        tc = ToolCall.from_dict({})
        assert tc.id == ""
        assert tc.name == ""


class TestClientConfig:
    def test_defaults(self):
        cfg = ClientConfig()
        assert cfg.mode == ClientMode.DIRECT
        assert cfg.temperature == 0.7

    def test_custom(self):
        cfg = ClientConfig(
            mode=ClientMode.GATEWAY,
            model="test",
            temperature=0.5,
            gateway_url="http://gw",
        )
        assert cfg.mode == ClientMode.GATEWAY


class TestGetContentText:
    def test_string(self):
        assert get_content_text("hello") == "hello"

    def test_none(self):
        assert get_content_text(None) == ""

    def test_parts(self):
        parts = [
            {"type": "text", "text": "hello"},
            {"type": "image", "source": "data"},
            {"type": "text", "text": "world"},
        ]
        assert get_content_text(parts) == "hello world"

    def test_empty_list(self):
        assert get_content_text([]) == ""


class TestIsImagePath:
    def test_png(self):
        assert is_image_path("photo.png") is True

    def test_py(self):
        assert is_image_path("script.py") is False

    def test_jpg(self):
        assert is_image_path("photo.jpg") is True


class TestMakeTextPart:
    def test_format(self):
        part = make_text_part("hello")
        assert part == {"type": "text", "text": "hello"}


class TestMakeImagePart:
    def test_creates_base64(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        part = make_image_part(str(img))
        assert part["type"] == "image"
        assert part["source"]["type"] == "base64"
        assert part["source"]["media_type"] == "image/png"


class TestNormalizeAnthropicBaseUrl:
    def test_none(self):
        assert normalize_anthropic_base_url(None) is None

    def test_strips_trailing_slash(self):
        assert normalize_anthropic_base_url("http://host/") == "http://host"

    def test_strips_v1(self):
        assert normalize_anthropic_base_url("http://host/v1") == "http://host"

    def test_no_v1(self):
        assert normalize_anthropic_base_url("http://host") == "http://host"
