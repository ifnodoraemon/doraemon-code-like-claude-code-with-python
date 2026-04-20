"""Tests for gateway.schema — unified data model and serialization."""

import pytest

from src.gateway.schema import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Choice,
    ErrorResponse,
    FinishReason,
    ModelInfo,
    Role,
    StreamChunk,
    ToolCall,
    ToolDefinition,
    ToolResult,
    Usage,
)


# ── Role ──────────────────────────────────────────────────────────────

class TestRole:
    def test_values(self):
        assert Role.SYSTEM.value == "system"
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"
        assert Role.TOOL.value == "tool"

    def test_is_str_enum(self):
        assert isinstance(Role.SYSTEM, str)
        assert Role.SYSTEM == "system"


# ── FinishReason ─────────────────────────────────────────────────────

class TestFinishReason:
    def test_values(self):
        assert FinishReason.STOP.value == "stop"
        assert FinishReason.TOOL_CALLS.value == "tool_calls"
        assert FinishReason.LENGTH.value == "length"
        assert FinishReason.ERROR.value == "error"


# ── ToolCall ─────────────────────────────────────────────────────────

class TestToolCall:
    def test_to_dict(self):
        tc = ToolCall(id="call_1", name="get_weather", arguments={"city": "NYC"})
        d = tc.to_dict()
        assert d["id"] == "call_1"
        assert d["type"] == "function"
        assert d["function"]["name"] == "get_weather"
        assert d["function"]["arguments"] == {"city": "NYC"}

    def test_from_dict(self):
        data = {"id": "call_2", "function": {"name": "search", "arguments": {"q": "test"}}}
        tc = ToolCall.from_dict(data)
        assert tc.id == "call_2"
        assert tc.name == "search"
        assert tc.arguments == {"q": "test"}

    def test_from_dict_missing_fields(self):
        tc = ToolCall.from_dict({})
        assert tc.id == ""
        assert tc.name == ""
        assert tc.arguments == {}

    def test_roundtrip(self):
        tc = ToolCall(id="c1", name="fn", arguments={"a": 1})
        assert ToolCall.from_dict(tc.to_dict()).name == "fn"


# ── ToolResult ───────────────────────────────────────────────────────

class TestToolResult:
    def test_to_dict(self):
        tr = ToolResult(tool_call_id="tc1", content="42")
        d = tr.to_dict()
        assert d == {"tool_call_id": "tc1", "content": "42"}


# ── ChatMessage ──────────────────────────────────────────────────────

class TestChatMessage:
    def test_to_dict_user(self):
        msg = ChatMessage(role=Role.USER, content="hello")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "hello"
        assert "tool_calls" not in d

    def test_to_dict_with_tool_calls(self):
        tc = ToolCall(id="c1", name="fn", arguments={})
        msg = ChatMessage(role=Role.ASSISTANT, content=None, tool_calls=[tc])
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert "content" not in d
        assert len(d["tool_calls"]) == 1

    def test_to_dict_with_tool_call_id(self):
        msg = ChatMessage(role=Role.TOOL, content="result", tool_call_id="c1", name="fn")
        d = msg.to_dict()
        assert d["tool_call_id"] == "c1"
        assert d["name"] == "fn"

    def test_from_dict(self):
        data = {"role": "user", "content": "hi"}
        msg = ChatMessage.from_dict(data)
        assert msg.role == "user"
        assert msg.content == "hi"

    def test_from_dict_with_tool_calls(self):
        data = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "function": {"name": "fn", "arguments": {}}}],
            "tool_call_id": "c1",
        }
        msg = ChatMessage.from_dict(data)
        assert msg.role == "assistant"
        assert len(msg.tool_calls) == 1
        assert msg.tool_call_id == "c1"

    def test_from_dict_defaults(self):
        msg = ChatMessage.from_dict({})
        assert msg.role == "user"
        assert msg.content is None
        assert msg.tool_calls is None

    def test_string_role(self):
        msg = ChatMessage(role="user", content="hi")
        d = msg.to_dict()
        assert d["role"] == "user"

    def test_roundtrip(self):
        original = ChatMessage(role=Role.ASSISTANT, content="hey", tool_call_id="x", name="n")
        restored = ChatMessage.from_dict(original.to_dict())
        assert restored.role == "assistant"
        assert restored.content == "hey"
        assert restored.tool_call_id == "x"


# ── ToolDefinition ───────────────────────────────────────────────────

class TestToolDefinition:
    def test_to_dict(self):
        td = ToolDefinition(name="search", description="Search the web", parameters={"type": "object"})
        d = td.to_dict()
        assert d["type"] == "function"
        assert d["function"]["name"] == "search"
        assert d["function"]["parameters"] == {"type": "object"}


# ── ChatRequest ──────────────────────────────────────────────────────

class TestChatRequest:
    def test_to_dict_minimal(self):
        req = ChatRequest(model="gpt-4o", messages=[ChatMessage(role=Role.USER, content="hi")])
        d = req.to_dict()
        assert d["model"] == "gpt-4o"
        assert len(d["messages"]) == 1
        assert d["stream"] is False

    def test_to_dict_full(self):
        req = ChatRequest(
            model="gpt-4o",
            messages=[ChatMessage(role=Role.USER, content="hi")],
            tools=[ToolDefinition(name="fn", description="d", parameters={})],
            temperature=0.5,
            max_tokens=100,
            stream=True,
            stop=["\n"],
            top_p=0.9,
            presence_penalty=0.1,
            frequency_penalty=0.2,
        )
        d = req.to_dict()
        assert d["temperature"] == 0.5
        assert d["max_tokens"] == 100
        assert d["stream"] is True
        assert d["stop"] == ["\n"]
        assert d["top_p"] == 0.9
        assert d["presence_penalty"] == 0.1
        assert d["frequency_penalty"] == 0.2
        assert len(d["tools"]) == 1

    def test_from_dict(self):
        data = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.3,
            "max_tokens": 50,
            "stream": True,
        }
        req = ChatRequest.from_dict(data)
        assert req.model == "gpt-4o"
        assert req.temperature == 0.3
        assert req.max_tokens == 50
        assert req.stream is True

    def test_from_dict_with_tools(self):
        data = {
            "model": "gpt-4o",
            "messages": [],
            "tools": [
                {
                    "function": {
                        "name": "search",
                        "description": "search",
                        "parameters": {"type": "object"},
                    }
                }
            ],
        }
        req = ChatRequest.from_dict(data)
        assert req.tools is not None
        assert len(req.tools) == 1
        assert req.tools[0].name == "search"

    def test_from_dict_defaults(self):
        req = ChatRequest.from_dict({"model": "", "messages": []})
        assert req.temperature == 0.7
        assert req.max_tokens is None
        assert req.stream is False

    def test_roundtrip(self):
        original = ChatRequest(
            model="gpt-4o",
            messages=[ChatMessage(role=Role.USER, content="test")],
            temperature=0.5,
            max_tokens=100,
            top_p=0.8,
        )
        restored = ChatRequest.from_dict(original.to_dict())
        assert restored.model == original.model
        assert restored.temperature == original.temperature
        assert restored.max_tokens == original.max_tokens


# ── Usage ────────────────────────────────────────────────────────────

class TestUsage:
    def test_to_dict(self):
        u = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        d = u.to_dict()
        assert d == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

    def test_defaults(self):
        u = Usage()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0


# ── Choice ───────────────────────────────────────────────────────────

class TestChoice:
    def test_to_dict_with_enum(self):
        ch = Choice(index=0, message=ChatMessage(role=Role.ASSISTANT, content="hi"), finish_reason=FinishReason.STOP)
        d = ch.to_dict()
        assert d["finish_reason"] == "stop"

    def test_to_dict_with_string(self):
        ch = Choice(index=0, message=ChatMessage(role=Role.ASSISTANT, content="hi"), finish_reason="stop")
        d = ch.to_dict()
        assert d["finish_reason"] == "stop"

    def test_to_dict_no_finish_reason(self):
        ch = Choice(index=0, message=ChatMessage(role=Role.ASSISTANT, content="hi"))
        d = ch.to_dict()
        assert d["finish_reason"] is None


# ── ChatResponse ─────────────────────────────────────────────────────

class TestChatResponse:
    def _make_response(self) -> ChatResponse:
        return ChatResponse(
            id="resp-1",
            model="gpt-4o",
            choices=[Choice(index=0, message=ChatMessage(role=Role.ASSISTANT, content="hello"), finish_reason="stop")],
            usage=Usage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
        )

    def test_to_dict(self):
        resp = self._make_response()
        d = resp.to_dict()
        assert d["id"] == "resp-1"
        assert d["object"] == "chat.completion"
        assert d["model"] == "gpt-4o"
        assert len(d["choices"]) == 1
        assert d["usage"]["total_tokens"] == 7

    def test_message_property(self):
        resp = self._make_response()
        assert resp.message is not None
        assert resp.message.content == "hello"

    def test_content_property(self):
        resp = self._make_response()
        assert resp.content == "hello"

    def test_tool_calls_property(self):
        tc = ToolCall(id="c1", name="fn", arguments={})
        resp = ChatResponse(
            id="r1",
            model="gpt-4o",
            choices=[Choice(index=0, message=ChatMessage(role=Role.ASSISTANT, content=None, tool_calls=[tc]))],
        )
        assert resp.tool_calls is not None
        assert len(resp.tool_calls) == 1

    def test_empty_choices(self):
        resp = ChatResponse(id="r1", model="gpt-4o", choices=[])
        assert resp.message is None
        assert resp.content is None
        assert resp.tool_calls is None

    def test_no_usage(self):
        resp = ChatResponse(id="r1", model="gpt-4o", choices=[])
        d = resp.to_dict()
        assert "usage" not in d


# ── StreamChunk ──────────────────────────────────────────────────────

class TestStreamChunk:
    def test_to_dict_with_content(self):
        chunk = StreamChunk(id="ch1", model="gpt-4o", delta_content="hello")
        d = chunk.to_dict()
        assert d["id"] == "ch1"
        assert d["object"] == "chat.completion.chunk"
        assert d["choices"][0]["delta"]["content"] == "hello"
        assert d["choices"][0]["finish_reason"] is None

    def test_to_dict_with_tool_calls(self):
        tc = ToolCall(id="c1", name="fn", arguments={})
        chunk = StreamChunk(id="ch1", model="gpt-4o", delta_tool_calls=[tc])
        d = chunk.to_dict()
        assert "tool_calls" in d["choices"][0]["delta"]

    def test_to_dict_with_usage(self):
        chunk = StreamChunk(id="ch1", model="gpt-4o", usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2))
        d = chunk.to_dict()
        assert d["usage"]["total_tokens"] == 2

    def test_to_dict_no_usage(self):
        chunk = StreamChunk(id="ch1", model="gpt-4o")
        d = chunk.to_dict()
        assert d["usage"] is None


# ── ModelInfo ────────────────────────────────────────────────────────

class TestModelInfo:
    def test_to_dict(self):
        mi = ModelInfo(
            id="gpt-4o",
            name="GPT-4o",
            provider="openai",
            description="test",
            context_window=128000,
            max_output=16384,
            input_price=2.5,
            output_price=10.0,
            capabilities=["text"],
            aliases=["4o"],
        )
        d = mi.to_dict()
        assert d["id"] == "gpt-4o"
        assert d["aliases"] == ["4o"]
        assert d["context_window"] == 128000

    def test_defaults(self):
        mi = ModelInfo(id="x", name="X", provider="test")
        assert mi.capabilities == []
        assert mi.aliases == []
        assert mi.context_window == 0


# ── ErrorResponse ────────────────────────────────────────────────────

class TestErrorResponse:
    def test_to_dict(self):
        err = ErrorResponse(error="not found", code="model_not_found")
        d = err.to_dict()
        assert d["error"]["message"] == "not found"
        assert d["error"]["code"] == "model_not_found"
        assert "details" not in d["error"]

    def test_to_dict_with_details(self):
        err = ErrorResponse(error="fail", code="x", details={"foo": "bar"})
        d = err.to_dict()
        assert d["error"]["details"] == {"foo": "bar"}
