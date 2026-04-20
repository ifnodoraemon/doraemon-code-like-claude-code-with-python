"""Tests for gateway adapters — base, OpenAI, Anthropic, Google."""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.gateway.adapters.base import AdapterConfig, BaseAdapter
from src.gateway.adapters.openai_adapter import OPENAI_MODELS, OpenAIAdapter
from src.gateway.adapters.anthropic_adapter import ANTHROPIC_MODELS, AnthropicAdapter
from src.gateway.adapters.google_adapter import GOOGLE_MODELS, GoogleAdapter
from src.gateway.schema import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Choice,
    FinishReason,
    ModelInfo,
    Role,
    StreamChunk,
    ToolCall,
    ToolDefinition,
    Usage,
)


# ── AdapterConfig ────────────────────────────────────────────────────


class TestAdapterConfig:
    def test_defaults(self):
        cfg = AdapterConfig()
        assert cfg.api_key is None
        assert cfg.api_base is None
        assert cfg.timeout == 60.0
        assert cfg.max_retries == 3
        assert cfg.extra == {}

    def test_custom(self):
        cfg = AdapterConfig(
            api_key="k", api_base="http://x", timeout=30.0, max_retries=5, extra={"a": 1}
        )
        assert cfg.api_key == "k"
        assert cfg.extra == {"a": 1}


# ── BaseAdapter (concrete subclass for testing) ──────────────────────


class ConcreteAdapter(BaseAdapter):
    provider_name = "test"

    async def initialize(self):
        pass

    async def chat(self, request):
        return ChatResponse(id="r", model="m", choices=[])

    async def chat_stream(self, request):
        if False:
            yield StreamChunk(id="", model="")

    def get_models(self):
        return [ModelInfo(id="test-model", name="Test", provider="test", aliases=["tm"])]


class TestBaseAdapter:
    def _make_adapter(self):
        return ConcreteAdapter(AdapterConfig())

    def test_repr(self):
        a = self._make_adapter()
        assert "ConcreteAdapter" in repr(a)
        assert "test" in repr(a)

    def test_supports_model_by_id(self):
        a = self._make_adapter()
        assert a.supports_model("test-model") is True

    def test_supports_model_by_alias(self):
        a = self._make_adapter()
        assert a.supports_model("tm") is True

    def test_supports_model_unknown(self):
        a = self._make_adapter()
        assert a.supports_model("nonexistent") is False

    def test_resolve_model_by_id(self):
        a = self._make_adapter()
        assert a.resolve_model("test-model") == "test-model"

    def test_resolve_model_by_alias(self):
        a = self._make_adapter()
        assert a.resolve_model("tm") == "test-model"

    def test_resolve_model_unknown(self):
        a = self._make_adapter()
        assert a.resolve_model("unknown") is None

    @pytest.mark.asyncio
    async def test_health_check_caching(self):
        a = self._make_adapter()
        result = await a.health_check()
        assert result is True
        # Should be cached
        a._health_cache = False
        a._health_cache_time = time.monotonic()
        result2 = await a.health_check()
        assert result2 is False

    @pytest.mark.asyncio
    async def test_health_check_cache_expiry(self):
        a = self._make_adapter()
        a._health_cache = True
        a._health_cache_time = time.monotonic() - 31  # expired
        result = await a.health_check()
        assert result is True  # re-evaluated

    @pytest.mark.asyncio
    async def test_health_check_exception(self):
        class FailAdapter(BaseAdapter):
            provider_name = "fail"

            async def initialize(self):
                pass

            async def chat(self, request):
                pass

            async def chat_stream(self, request):
                if False:
                    yield

            def get_models(self):
                raise RuntimeError("boom")

        a = FailAdapter(AdapterConfig())
        result = await a.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_close(self):
        a = self._make_adapter()
        a._client = object()
        await a.close()
        assert a._client is None


# ── OpenAIAdapter ────────────────────────────────────────────────────


class TestOpenAIAdapter:
    def _make_adapter(self):
        return OpenAIAdapter(AdapterConfig(api_key="test-key"))

    def test_provider_name(self):
        assert self._make_adapter().provider_name == "openai"

    def test_get_models(self):
        adapter = self._make_adapter()
        models = adapter.get_models()
        assert len(models) == 4
        assert models[0].id == "gpt-4o"
        ids = [m.id for m in models]
        assert "o1" in ids
        assert "o3-mini" in ids

    def test_supports_model(self):
        adapter = self._make_adapter()
        assert adapter.supports_model("gpt-4o") is True
        assert adapter.supports_model("4o") is True
        assert adapter.supports_model("gpt4o-mini") is True
        assert adapter.supports_model("unknown") is False

    def test_resolve_model(self):
        adapter = self._make_adapter()
        assert adapter.resolve_model("4o") == "gpt-4o"
        assert adapter.resolve_model("o1-preview") == "o1"
        assert adapter.resolve_model("nonexistent") is None

    def test_convert_message_user(self):
        adapter = self._make_adapter()
        msg = ChatMessage(role=Role.USER, content="hello")
        result = adapter._convert_message(msg)
        assert result["role"] == "user"
        assert result["content"] == "hello"

    def test_convert_message_with_tool_calls(self):
        adapter = self._make_adapter()
        tc = ToolCall(id="c1", name="fn", arguments={"a": 1})
        msg = ChatMessage(role=Role.ASSISTANT, content=None, tool_calls=[tc])
        result = adapter._convert_message(msg)
        assert result["role"] == "assistant"
        assert "content" not in result
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "fn"

    def test_convert_message_tool_response(self):
        adapter = self._make_adapter()
        msg = ChatMessage(role=Role.TOOL, content="result", tool_call_id="c1", name="fn")
        result = adapter._convert_message(msg)
        assert result["tool_call_id"] == "c1"
        assert result["name"] == "fn"

    def test_convert_message_string_role(self):
        adapter = self._make_adapter()
        msg = ChatMessage(role="user", content="hi")
        result = adapter._convert_message(msg)
        assert result["role"] == "user"

    def test_convert_response(self):
        adapter = self._make_adapter()
        mock_resp = MagicMock()
        mock_resp.id = "resp-1"
        mock_resp.model = "gpt-4o"
        mock_resp.created = 12345
        mock_choice = MagicMock()
        mock_choice.index = 0
        mock_choice.finish_reason = "stop"
        mock_choice.message.content = "hello"
        mock_choice.message.tool_calls = None
        mock_resp.choices = [mock_choice]
        mock_resp.usage = MagicMock()
        mock_resp.usage.prompt_tokens = 10
        mock_resp.usage.completion_tokens = 5
        mock_resp.usage.total_tokens = 15

        result = adapter._convert_response(mock_resp)
        assert result.id == "resp-1"
        assert result.model == "gpt-4o"
        assert len(result.choices) == 1
        assert result.choices[0].message.content == "hello"
        assert result.usage.total_tokens == 15

    def test_convert_response_with_tool_calls(self):
        adapter = self._make_adapter()
        mock_resp = MagicMock()
        mock_resp.id = "r1"
        mock_resp.model = "gpt-4o"
        mock_resp.created = 0
        mock_tc = MagicMock()
        mock_tc.id = "c1"
        mock_tc.function.name = "fn"
        mock_tc.function.arguments = '{"a": 1}'
        mock_choice = MagicMock()
        mock_choice.index = 0
        mock_choice.finish_reason = "tool_calls"
        mock_choice.message.content = None
        mock_choice.message.tool_calls = [mock_tc]
        mock_resp.choices = [mock_choice]
        mock_resp.usage = None

        result = adapter._convert_response(mock_resp)
        assert result.choices[0].message.tool_calls is not None
        assert result.choices[0].message.tool_calls[0].name == "fn"
        assert result.choices[0].message.tool_calls[0].arguments == {"a": 1}

    def test_convert_response_malformed_tool_args(self):
        adapter = self._make_adapter()
        mock_resp = MagicMock()
        mock_resp.id = "r1"
        mock_resp.model = "gpt-4o"
        mock_resp.created = 0
        mock_tc = MagicMock()
        mock_tc.id = "c1"
        mock_tc.function.name = "fn"
        mock_tc.function.arguments = "not-json"
        mock_choice = MagicMock()
        mock_choice.index = 0
        mock_choice.finish_reason = "tool_calls"
        mock_choice.message.content = None
        mock_choice.message.tool_calls = [mock_tc]
        mock_resp.choices = [mock_choice]
        mock_resp.usage = None

        result = adapter._convert_response(mock_resp)
        assert result.choices[0].message.tool_calls[0].arguments == {}

    @pytest.mark.asyncio
    async def test_initialize_import_error(self):
        adapter = self._make_adapter()
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError, match="openai package"):
                await adapter.initialize()

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        adapter = self._make_adapter()
        with patch("openai.AsyncOpenAI") as mock_client:
            await adapter.initialize()
            mock_client.assert_called_once()
            assert adapter._client is not None

    @pytest.mark.asyncio
    async def test_chat_builds_correct_kwargs(self):
        adapter = self._make_adapter()
        mock_client = AsyncMock()
        adapter._client = mock_client

        mock_choice = MagicMock()
        mock_choice.index = 0
        mock_choice.finish_reason = "stop"
        mock_choice.message.content = "hi"
        mock_choice.message.tool_calls = None
        mock_resp = MagicMock()
        mock_resp.id = "r1"
        mock_resp.model = "gpt-4o"
        mock_resp.created = 0
        mock_resp.choices = [mock_choice]
        mock_resp.usage = None
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

        tool = ToolDefinition(name="fn", description="d", parameters={"type": "object"})
        req = ChatRequest(
            model="gpt-4o",
            messages=[ChatMessage(role=Role.USER, content="hello")],
            tools=[tool],
            max_tokens=100,
            stop=["\n"],
            top_p=0.9,
            presence_penalty=0.1,
            frequency_penalty=0.2,
        )
        await adapter.chat(req)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["stop"] == ["\n"]
        assert call_kwargs["top_p"] == 0.9
        assert len(call_kwargs["tools"]) == 1


# ── AnthropicAdapter ─────────────────────────────────────────────────


class TestAnthropicAdapter:
    def _make_adapter(self):
        return AnthropicAdapter(AdapterConfig(api_key="test-key"))

    def test_provider_name(self):
        assert self._make_adapter().provider_name == "anthropic"

    def test_get_models(self):
        models = self._make_adapter().get_models()
        assert len(models) == 4
        ids = [m.id for m in models]
        assert "claude-sonnet-4-20250514" in ids

    def test_supports_model(self):
        adapter = self._make_adapter()
        assert adapter.supports_model("sonnet") is True
        assert adapter.supports_model("claude-sonnet-4") is True
        assert adapter.supports_model("opus") is True

    def test_convert_message_tool_role(self):
        adapter = self._make_adapter()
        msg = ChatMessage(role=Role.TOOL, content="result", tool_call_id="c1")
        result = adapter._convert_message(msg)
        assert result["role"] == "user"
        assert result["content"][0]["type"] == "tool_result"

    def test_convert_message_assistant_with_tool_calls(self):
        adapter = self._make_adapter()
        tc = ToolCall(id="c1", name="fn", arguments={"a": 1})
        msg = ChatMessage(role=Role.ASSISTANT, content="thinking", tool_calls=[tc])
        result = adapter._convert_message(msg)
        assert result["role"] == "assistant"
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "text"
        assert result["content"][1]["type"] == "tool_use"

    def test_convert_message_user_role(self):
        adapter = self._make_adapter()
        msg = ChatMessage(role=Role.USER, content="hello")
        result = adapter._convert_message(msg)
        assert result["role"] == "user"
        assert result["content"][0]["text"] == "hello"

    def test_convert_tool(self):
        adapter = self._make_adapter()
        tool = ToolDefinition(name="fn", description="d", parameters={"type": "object"})
        result = adapter._convert_tool(tool)
        assert result["name"] == "fn"
        assert result["input_schema"] == {"type": "object"}

    def test_convert_finish_reason(self):
        adapter = self._make_adapter()
        assert adapter._convert_finish_reason("end_turn") == FinishReason.STOP.value
        assert adapter._convert_finish_reason("tool_use") == FinishReason.TOOL_CALLS.value
        assert adapter._convert_finish_reason("max_tokens") == FinishReason.LENGTH.value
        assert adapter._convert_finish_reason(None) == FinishReason.STOP.value
        assert adapter._convert_finish_reason("unknown") == FinishReason.STOP.value

    def test_convert_response(self):
        adapter = self._make_adapter()
        mock_resp = MagicMock()
        mock_resp.id = "msg-1"
        mock_resp.stop_reason = "end_turn"
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "hello"
        mock_resp.content = [text_block]
        mock_resp.usage = MagicMock()
        mock_resp.usage.input_tokens = 10
        mock_resp.usage.output_tokens = 5

        result = adapter._convert_response(mock_resp, "claude-sonnet-4-20250514")
        assert result.id == "msg-1"
        assert result.model == "claude-sonnet-4-20250514"
        assert result.choices[0].message.content == "hello"
        assert result.usage.total_tokens == 15

    def test_convert_response_with_tool_use(self):
        adapter = self._make_adapter()
        mock_resp = MagicMock()
        mock_resp.id = "msg-2"
        mock_resp.stop_reason = "tool_use"
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "thinking"
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "c1"
        tool_block.name = "fn"
        tool_block.input = {"a": 1}
        mock_resp.content = [text_block, tool_block]
        mock_resp.usage = MagicMock()
        mock_resp.usage.input_tokens = 20
        mock_resp.usage.output_tokens = 10

        result = adapter._convert_response(mock_resp, "claude-sonnet-4-20250514")
        assert result.choices[0].message.tool_calls is not None
        assert result.choices[0].message.tool_calls[0].name == "fn"

    @pytest.mark.asyncio
    async def test_chat_extracts_system_message(self):
        adapter = self._make_adapter()
        mock_client = AsyncMock()
        adapter._client = mock_client

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "hi"
        mock_resp = MagicMock()
        mock_resp.id = "r1"
        mock_resp.stop_reason = "end_turn"
        mock_resp.content = [mock_text_block]
        mock_resp.usage = MagicMock()
        mock_resp.usage.input_tokens = 5
        mock_resp.usage.output_tokens = 1
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        req = ChatRequest(
            model="claude-sonnet-4-20250514",
            messages=[
                ChatMessage(role=Role.SYSTEM, content="sys"),
                ChatMessage(role=Role.USER, content="hello"),
            ],
        )
        await adapter.chat(req)
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "sys"
        assert len(call_kwargs["messages"]) == 1


# ── GoogleAdapter ────────────────────────────────────────────────────


class TestGoogleAdapter:
    def _make_adapter(self):
        return GoogleAdapter(AdapterConfig(api_key="test-key"))

    def test_provider_name(self):
        assert self._make_adapter().provider_name == "google"

    def test_get_models(self):
        models = self._make_adapter().get_models()
        assert len(models) == 3
        ids = [m.id for m in models]
        assert "gemini-2.5-pro-preview" in ids

    def test_supports_model(self):
        adapter = self._make_adapter()
        assert adapter.supports_model("gemini-2.5-pro") is True
        assert adapter.supports_model("flash") is True

    def test_convert_messages_with_system(self):
        adapter = self._make_adapter()
        msgs = [
            ChatMessage(role=Role.SYSTEM, content="sys"),
            ChatMessage(role=Role.USER, content="hello"),
            ChatMessage(role=Role.ASSISTANT, content="hi"),
        ]
        contents, sys_instr = adapter._convert_messages(msgs)
        assert sys_instr == "sys"
        assert len(contents) == 2
        assert contents[0]["role"] == "user"
        assert contents[1]["role"] == "model"

    def test_convert_messages_tool_result(self):
        adapter = self._make_adapter()
        msgs = [
            ChatMessage(role=Role.TOOL, content="result", tool_call_id="c1", name="fn"),
        ]
        contents, sys_instr = adapter._convert_messages(msgs)
        assert sys_instr is None
        assert any("function_response" in p for p in contents[0]["parts"])

    def test_convert_messages_with_tool_calls(self):
        adapter = self._make_adapter()
        tc = ToolCall(id="c1", name="fn", arguments={"a": 1})
        msgs = [ChatMessage(role=Role.ASSISTANT, content="thinking", tool_calls=[tc])]
        contents, _ = adapter._convert_messages(msgs)
        assert any("function_call" in p for p in contents[0]["parts"])

    def test_convert_finish_reason(self):
        adapter = self._make_adapter()
        assert "stop" in adapter._convert_finish_reason("STOP")
        assert "length" in adapter._convert_finish_reason("MAX_TOKENS")
        assert "tool" in adapter._convert_finish_reason("FUNCTION_CALL")

    @pytest.mark.asyncio
    async def test_chat_api_error(self):
        adapter = self._make_adapter()
        mock_client = MagicMock()
        adapter._client = mock_client
        mock_client.aio.models.generate_content = AsyncMock(side_effect=Exception("API fail"))

        req = ChatRequest(
            model="gemini-2.5-pro-preview", messages=[ChatMessage(role=Role.USER, content="hi")]
        )
        with pytest.raises(RuntimeError, match="Google API request failed"):
            await adapter.chat(req)

    def test_convert_messages_user_no_content(self):
        adapter = self._make_adapter()
        msg = ChatMessage(role=Role.USER, content=None)
        contents, _ = adapter._convert_messages([msg])
        assert len(contents) == 0

    def test_convert_response_with_candidate_text(self):
        adapter = self._make_adapter()
        mock_resp = MagicMock()
        mock_resp.candidates = [MagicMock()]
        mock_resp.candidates[0].text = "hello"
        mock_resp.candidates[0].content = None
        mock_resp.candidates[0].finish_reason = None
        mock_resp.usage_metadata = MagicMock()
        mock_resp.usage_metadata.prompt_token_count = 5
        mock_resp.usage_metadata.candidates_token_count = 3
        mock_resp.usage_metadata.total_token_count = 8

        result = adapter._convert_response(mock_resp, "gemini-2.5-pro-preview")
        assert result.choices[0].message.content == "hello"

    def test_convert_response_no_candidates(self):
        adapter = self._make_adapter()
        mock_resp = MagicMock()
        mock_resp.candidates = None
        mock_resp.usage_metadata = None

        result = adapter._convert_response(mock_resp, "gemini-2.5-pro-preview")
        assert result.choices[0].message.content is None

    def test_convert_response_with_tool_call(self):
        adapter = self._make_adapter()
        mock_resp = MagicMock()
        candidate = MagicMock(spec=["content", "finish_reason"])
        part = MagicMock()
        part.text = None
        part.function_call = MagicMock()
        part.function_call.name = "fn"
        part.function_call.args = {"a": 1}
        candidate.content = MagicMock()
        candidate.content.parts = [part]
        candidate.finish_reason = None
        mock_resp.candidates = [candidate]
        mock_resp.usage_metadata = None

        result = adapter._convert_response(mock_resp, "gemini-2.5-pro-preview")
        assert result.choices[0].message.tool_calls is not None

    def test_convert_finish_reason_variants(self):
        adapter = self._make_adapter()
        assert "length" in adapter._convert_finish_reason("MAX_TOKENS")
        assert "stop" in adapter._convert_finish_reason("STOP")
        assert "tool" in adapter._convert_finish_reason("FUNCTION_CALL")
        assert "stop" in adapter._convert_finish_reason("UNKNOWN")


class TestAnthropicAdapterChat:
    def _make_adapter(self):
        return AnthropicAdapter(AdapterConfig(api_key="test-key"))

    @pytest.mark.asyncio
    async def test_chat_with_tools_and_params(self):
        adapter = self._make_adapter()
        mock_client = AsyncMock()
        adapter._client = mock_client

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "response"
        mock_resp = MagicMock()
        mock_resp.id = "r1"
        mock_resp.stop_reason = "end_turn"
        mock_resp.content = [mock_text_block]
        mock_resp.usage = MagicMock()
        mock_resp.usage.input_tokens = 10
        mock_resp.usage.output_tokens = 5
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        tool = ToolDefinition(name="fn", description="d", parameters={"type": "object"})
        req = ChatRequest(
            model="claude-sonnet-4-20250514",
            messages=[ChatMessage(role=Role.USER, content="hi")],
            tools=[tool],
            temperature=0.5,
            max_tokens=100,
            stop=["\n"],
            top_p=0.8,
        )
        await adapter.chat(req)
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["tools"] is not None
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["stop_sequences"] == ["\n"]
        assert call_kwargs["top_p"] == 0.8

    @pytest.mark.asyncio
    async def test_initialize_import_error(self):
        adapter = self._make_adapter()
        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(ImportError, match="anthropic package"):
                await adapter.initialize()

    def test_convert_message_string_role(self):
        adapter = self._make_adapter()
        msg = ChatMessage(role="user", content="hi")
        result = adapter._convert_message(msg)
        assert result["role"] == "user"

    def test_convert_message_assistant_no_content(self):
        adapter = self._make_adapter()
        msg = ChatMessage(role=Role.ASSISTANT, content=None)
        result = adapter._convert_message(msg)
        assert result["role"] == "assistant"


class TestOpenAIAdapterChatStream:
    def _make_adapter(self):
        return OpenAIAdapter(AdapterConfig(api_key="test-key"))

    @pytest.mark.asyncio
    async def test_chat_stream_produces_chunks(self):
        adapter = self._make_adapter()
        mock_client = AsyncMock()
        adapter._client = mock_client

        chunk1 = MagicMock()
        chunk1.id = "c1"
        chunk1.model = "gpt-4o"
        choice = MagicMock()
        choice.delta.content = "hello"
        choice.delta.tool_calls = None
        choice.finish_reason = None
        chunk1.choices = [choice]
        chunk1.usage = None

        chunk2 = MagicMock()
        chunk2.id = "c1"
        chunk2.model = "gpt-4o"
        choice2 = MagicMock()
        choice2.delta.content = None
        choice2.delta.tool_calls = None
        choice2.finish_reason = "stop"
        chunk2.choices = [choice2]
        chunk2.usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        class AsyncStream:
            def __init__(self, items):
                self._items = items

            def __aiter__(self):
                self._iter = iter(self._items)
                return self

            async def __anext__(self):
                try:
                    return next(self._iter)
                except StopIteration:
                    raise StopAsyncIteration

        stream = AsyncStream([chunk1, chunk2])
        mock_client.chat.completions.create = AsyncMock(return_value=stream)

        req = ChatRequest(
            model="gpt-4o",
            messages=[ChatMessage(role=Role.USER, content="hi")],
        )
        chunks = []
        async for chunk in adapter.chat_stream(req):
            chunks.append(chunk)
        assert len(chunks) == 2

    @pytest.mark.asyncio
    async def test_chat_stream_with_max_tokens_tools_stop(self):
        adapter = self._make_adapter()
        mock_client = AsyncMock()
        adapter._client = mock_client

        chunk = MagicMock()
        chunk.id = "c1"
        chunk.model = "gpt-4o"
        choice = MagicMock()
        choice.delta.content = None
        choice.delta.tool_calls = None
        choice.finish_reason = "stop"
        chunk.choices = [choice]
        chunk.usage = None

        class AsyncStream:
            def __init__(self, items):
                self._items = items
            def __aiter__(self):
                self._iter = iter(self._items)
                return self
            async def __anext__(self):
                try:
                    return next(self._iter)
                except StopIteration:
                    raise StopAsyncIteration

        stream = AsyncStream([chunk])
        mock_client.chat.completions.create = AsyncMock(return_value=stream)

        tool = ToolDefinition(name="fn", description="d", parameters={"type": "object"})
        req = ChatRequest(
            model="gpt-4o",
            messages=[ChatMessage(role=Role.USER, content="hi")],
            max_tokens=100,
            tools=[tool],
            stop=["\n"],
        )
        chunks = []
        async for c in adapter.chat_stream(req):
            chunks.append(c)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 100
        assert len(call_kwargs["tools"]) == 1
        assert call_kwargs["stop"] == ["\n"]

    @pytest.mark.asyncio
    async def test_chat_stream_with_tool_calls(self):
        adapter = self._make_adapter()
        mock_client = AsyncMock()
        adapter._client = mock_client

        tc = MagicMock()
        tc.id = "tc1"
        tc.function.name = "fn"
        tc.function.arguments = '{"a": 1}'

        chunk = MagicMock()
        chunk.id = "c1"
        chunk.model = "gpt-4o"
        choice = MagicMock()
        choice.delta.content = None
        choice.delta.tool_calls = [tc]
        choice.finish_reason = None
        chunk.choices = [choice]
        chunk.usage = None

        class AsyncStream:
            def __init__(self, items):
                self._items = items
            def __aiter__(self):
                self._iter = iter(self._items)
                return self
            async def __anext__(self):
                try:
                    return next(self._iter)
                except StopIteration:
                    raise StopAsyncIteration

        stream = AsyncStream([chunk])
        mock_client.chat.completions.create = AsyncMock(return_value=stream)

        req = ChatRequest(
            model="gpt-4o",
            messages=[ChatMessage(role=Role.USER, content="use tool")],
            tools=[ToolDefinition(name="fn", description="d", parameters={"type": "object"})],
        )
        chunks = []
        async for c in adapter.chat_stream(req):
            chunks.append(c)
        assert len(chunks) == 1
        assert chunks[0].delta_tool_calls is not None
        assert chunks[0].delta_tool_calls[0].name == "fn"
        assert chunks[0].delta_tool_calls[0].arguments == {"a": 1}

    @pytest.mark.asyncio
    async def test_chat_stream_tool_call_invalid_json(self):
        adapter = self._make_adapter()
        mock_client = AsyncMock()
        adapter._client = mock_client

        tc = MagicMock()
        tc.id = "tc2"
        tc.function.name = "fn2"
        tc.function.arguments = "not-json"

        chunk = MagicMock()
        chunk.id = "c1"
        chunk.model = "gpt-4o"
        choice = MagicMock()
        choice.delta.content = None
        choice.delta.tool_calls = [tc]
        choice.finish_reason = None
        chunk.choices = [choice]
        chunk.usage = None

        class AsyncStream:
            def __init__(self, items):
                self._items = items
            def __aiter__(self):
                self._iter = iter(self._items)
                return self
            async def __anext__(self):
                try:
                    return next(self._iter)
                except StopIteration:
                    raise StopAsyncIteration

        stream = AsyncStream([chunk])
        mock_client.chat.completions.create = AsyncMock(return_value=stream)

        req = ChatRequest(
            model="gpt-4o",
            messages=[ChatMessage(role=Role.USER, content="hi")],
            tools=[ToolDefinition(name="fn2", description="d", parameters={"type": "object"})],
        )
        chunks = []
        async for c in adapter.chat_stream(req):
            chunks.append(c)
        assert len(chunks) == 1
        assert chunks[0].delta_tool_calls[0].arguments == {}


class TestGoogleAdapterChatStream:
    def _make_adapter(self):
        return GoogleAdapter(AdapterConfig(api_key="test-key"))

    @pytest.mark.asyncio
    async def test_chat_stream_produces_chunks(self):
        adapter = self._make_adapter()
        mock_client = MagicMock()
        adapter._client = mock_client

        chunk1 = MagicMock()
        chunk1.text = "hello"
        chunk1.candidates = None

        fc = MagicMock()
        fc.name = "fn"
        fc.args = {"a": 1}

        part = MagicMock()
        part.function_call = fc
        part.text = None

        candidate = MagicMock()
        candidate.content = MagicMock()
        candidate.content.parts = [part]
        candidate.finish_reason = None

        chunk2 = MagicMock()
        chunk2.text = None
        chunk2.candidates = [candidate]

        chunk3 = MagicMock()
        chunk3.text = None
        finish_candidate = MagicMock()
        finish_candidate.finish_reason = "STOP"
        chunk3.candidates = [finish_candidate]

        class AsyncStream:
            def __init__(self, items):
                self._items = items

            def __aiter__(self):
                self._iter = iter(self._items)
                return self

            async def __anext__(self):
                try:
                    return next(self._iter)
                except StopIteration:
                    raise StopAsyncIteration

        stream = AsyncStream([chunk1, chunk2, chunk3])
        mock_client.aio.models.generate_content_stream = AsyncMock(return_value=stream)

        req = ChatRequest(
            model="gemini-2.5-flash-preview",
            messages=[ChatMessage(role=Role.USER, content="hi")],
        )
        chunks = []
        async for chunk in adapter.chat_stream(req):
            chunks.append(chunk)
        assert len(chunks) == 3
        assert chunks[0].delta_content == "hello"
        assert chunks[1].delta_tool_calls is not None
        assert chunks[2].finish_reason is not None

    @pytest.mark.asyncio
    async def test_chat_stream_api_error(self):
        adapter = self._make_adapter()
        mock_client = MagicMock()
        adapter._client = mock_client
        mock_client.aio.models.generate_content_stream = AsyncMock(
            side_effect=Exception("stream fail")
        )

        req = ChatRequest(
            model="gemini-2.5-flash-preview",
            messages=[ChatMessage(role=Role.USER, content="hi")],
        )
        with pytest.raises(RuntimeError, match="Google stream API request failed"):
            async for _ in adapter.chat_stream(req):
                pass


class TestGoogleAdapterChatSuccess:
    def _make_adapter(self):
        return GoogleAdapter(AdapterConfig(api_key="test-key"))

    @pytest.mark.asyncio
    async def test_chat_success_with_candidate_text(self):
        adapter = self._make_adapter()
        mock_client = MagicMock()
        adapter._client = mock_client

        mock_resp = MagicMock()
        mock_resp.candidates = [MagicMock()]
        mock_resp.candidates[0].text = "hello"
        mock_resp.candidates[0].content = None
        mock_resp.candidates[0].finish_reason = None
        mock_resp.usage_metadata = MagicMock()
        mock_resp.usage_metadata.prompt_token_count = 5
        mock_resp.usage_metadata.candidates_token_count = 3
        mock_resp.usage_metadata.total_token_count = 8

        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_resp)

        req = ChatRequest(
            model="gemini-2.5-flash-preview",
            messages=[ChatMessage(role=Role.USER, content="hi")],
        )
        result = await adapter.chat(req)
        assert result.choices[0].message.content == "hello"

    @pytest.mark.asyncio
    async def test_chat_with_tool_call_parts(self):
        adapter = self._make_adapter()
        mock_client = MagicMock()
        adapter._client = mock_client

        part = MagicMock()
        part.text = None
        part.function_call = MagicMock()
        part.function_call.name = "fn"
        part.function_call.args = {"a": 1}

        candidate = MagicMock(spec=["content", "finish_reason"])
        candidate.content = MagicMock()
        candidate.content.parts = [part]
        candidate.finish_reason = None

        mock_resp = MagicMock()
        mock_resp.candidates = [candidate]
        mock_resp.usage_metadata = None

        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_resp)

        req = ChatRequest(
            model="gemini-2.5-flash-preview",
            messages=[ChatMessage(role=Role.USER, content="hi")],
            tools=[ToolDefinition(name="fn", description="d", parameters={"type": "object"})],
        )
        result = await adapter.chat(req)
        assert result.choices[0].message.tool_calls is not None
        assert result.choices[0].finish_reason == FinishReason.TOOL_CALLS.value


class TestAnthropicAdapterChatStream:
    def _make_adapter(self):
        return AnthropicAdapter(AdapterConfig(api_key="test-key"))

    @pytest.mark.asyncio
    async def test_chat_stream_produces_chunks(self):
        adapter = self._make_adapter()
        mock_client = AsyncMock()
        adapter._client = mock_client

        class MockStream:
            def __init__(self):
                self.events = []

            def __aiter__(self):
                self._idx = 0
                return self

            async def __anext__(self):
                if self._idx >= len(self.events):
                    raise StopAsyncIteration
                event = self.events[self._idx]
                self._idx += 1
                return event

        msg_start = MagicMock()
        msg_start.type = "message_start"
        msg_start.message = MagicMock()
        msg_start.message.usage = MagicMock()
        msg_start.message.usage.input_tokens = 10

        text_delta = MagicMock()
        text_delta.type = "content_block_delta"
        text_delta.delta = MagicMock()
        text_delta.delta.text = "hello"

        msg_delta = MagicMock()
        msg_delta.type = "message_delta"
        msg_delta.usage = MagicMock()
        msg_delta.usage.output_tokens = 5
        msg_delta.delta = MagicMock()
        msg_delta.delta.stop_reason = "end_turn"

        stream = MockStream()
        stream.events = [msg_start, text_delta, msg_delta]

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=stream)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_client.messages.stream = MagicMock(return_value=mock_stream_ctx)

        req = ChatRequest(
            model="claude-sonnet-4-20250514",
            messages=[ChatMessage(role=Role.USER, content="hi")],
        )
        chunks = []
        async for chunk in adapter.chat_stream(req):
            chunks.append(chunk)
        assert len(chunks) == 3
        assert chunks[1].delta_content == "hello"
        assert chunks[2].finish_reason is not None

    @pytest.mark.asyncio
    async def test_chat_stream_with_tool_use(self):
        adapter = self._make_adapter()
        mock_client = AsyncMock()
        adapter._client = mock_client

        class MockStream:
            def __init__(self):
                self.events = []

            def __aiter__(self):
                self._idx = 0
                return self

            async def __anext__(self):
                if self._idx >= len(self.events):
                    raise StopAsyncIteration
                event = self.events[self._idx]
                self._idx += 1
                return event

        tool_start = MagicMock()
        tool_start.type = "content_block_start"
        tool_start.content_block = MagicMock()
        tool_start.content_block.type = "tool_use"
        tool_start.content_block.id = "t1"
        tool_start.content_block.name = "fn"
        tool_start.index = 0

        msg_delta = MagicMock()
        msg_delta.type = "message_delta"
        msg_delta.usage = MagicMock()
        msg_delta.usage.output_tokens = 10
        msg_delta.delta = MagicMock()
        msg_delta.delta.stop_reason = "tool_use"

        stream = MockStream()
        stream.events = [tool_start, msg_delta]

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=stream)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_client.messages.stream = MagicMock(return_value=mock_stream_ctx)

        req = ChatRequest(
            model="claude-sonnet-4-20250514",
            messages=[ChatMessage(role=Role.USER, content="use tool")],
        )
        chunks = []
        async for chunk in adapter.chat_stream(req):
            chunks.append(chunk)
        assert chunks[0].delta_tool_calls is not None
        assert chunks[0].delta_tool_calls[0].name == "fn"


class TestGoogleAdapterConvertResponseDetailed:
    def test_convert_response_with_usage(self):
        adapter = GoogleAdapter(AdapterConfig(api_key="k"))
        mock_resp = MagicMock()
        mock_resp.candidates = [MagicMock()]
        mock_resp.candidates[0].text = "hi"
        mock_resp.candidates[0].content = None
        mock_resp.candidates[0].finish_reason = "STOP"
        mock_resp.usage_metadata = MagicMock()
        mock_resp.usage_metadata.prompt_token_count = 10
        mock_resp.usage_metadata.candidates_token_count = 5
        mock_resp.usage_metadata.total_token_count = 15

        result = adapter._convert_response(mock_resp, "gemini-2.5-pro-preview")
        assert result.usage is not None
        assert result.usage.total_tokens == 15

    def test_convert_response_with_content_parts(self):
        adapter = GoogleAdapter(AdapterConfig(api_key="k"))
        part = MagicMock()
        part.text = "text part"
        part.function_call = None
        part.has_text = False

        candidate = MagicMock(spec=["content", "finish_reason"])
        candidate.content = MagicMock()
        candidate.content.parts = [part]
        candidate.finish_reason = None

        mock_resp = MagicMock()
        mock_resp.candidates = [candidate]
        mock_resp.usage_metadata = None

        result = adapter._convert_response(mock_resp, "gemini-2.5-pro-preview")
        assert result.choices[0].message.content == "text part"

    def test_convert_response_no_candidates_no_usage(self):
        adapter = GoogleAdapter(AdapterConfig(api_key="k"))
        mock_resp = MagicMock()
        mock_resp.candidates = None
        mock_resp.usage_metadata = None

        result = adapter._convert_response(mock_resp, "gemini-2.5-pro-preview")
        assert result.choices[0].message.content is None
        assert result.usage is None

    def test_convert_response_function_call_args_error(self):
        adapter = GoogleAdapter(AdapterConfig(api_key="k"))
        fc = MagicMock()
        fc.name = "fn"
        fc.args = MagicMock()
        fc.args.__iter__ = MagicMock(side_effect=TypeError("not iterable"))
        fc.args.items = MagicMock(side_effect=TypeError("not iterable"))

        part = MagicMock()
        part.text = None
        part.function_call = fc

        candidate = MagicMock(spec=["content", "finish_reason"])
        candidate.content = MagicMock()
        candidate.content.parts = [part]
        candidate.finish_reason = None

        mock_resp = MagicMock()
        mock_resp.candidates = [candidate]
        mock_resp.usage_metadata = None

        result = adapter._convert_response(mock_resp, "gemini-2.5-pro-preview")
        assert result.choices[0].message.tool_calls is not None
        assert result.choices[0].message.tool_calls[0].arguments == {}
