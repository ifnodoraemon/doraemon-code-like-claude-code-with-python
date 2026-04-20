"""Additional coverage tests for core.llm.model_client_direct - _chat_openai, _chat_anthropic, _chat_google, streaming."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.llm.model_client_direct import (
    DirectModelClient,
    _is_retryable,
    _retry_async,
)
from src.core.llm.model_utils import ChatResponse, ClientConfig, Provider, StreamChunk


def _make_config(**kw):
    defaults = {"model": "gpt-4o", "openai_api_key": "o"}
    defaults.update(kw)
    return ClientConfig(**defaults)


class TestChatOpenaiViaChatCompletions:
    @pytest.mark.asyncio
    async def test_chat_completions_normal_response(self):
        config = _make_config(openai_protocol="chat_completions")
        client = DirectModelClient(config)
        mock_resp = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="hello", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)
        client._providers = {Provider.OPENAI: mock_client}
        with patch("src.core.llm.model_client_direct.OpenAIAdapter") as MockAdapter:
            MockAdapter.convert_messages.return_value = [{"role": "user", "content": "hi"}]
            MockAdapter.build_params.return_value = {"model": "gpt-4o"}
            result = await client._chat_openai([{"role": "user", "content": "hi"}])
        assert result.content == "hello"
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_chat_completions_with_tool_calls(self):
        config = _make_config(openai_protocol="chat_completions")
        client = DirectModelClient(config)
        tc = SimpleNamespace(
            id="tc1",
            function=SimpleNamespace(name="fn", arguments='{"a":1}'),
            thought_signature=None,
        )
        mock_resp = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="used tool", tool_calls=[tc]),
                    finish_reason="tool_calls",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)
        client._providers = {Provider.OPENAI: mock_client}
        with patch("src.core.llm.model_client_direct.OpenAIAdapter") as MockAdapter:
            MockAdapter.convert_messages.return_value = [{"role": "user", "content": "hi"}]
            MockAdapter.build_params.return_value = {"model": "gpt-4o"}
            result = await client._chat_openai([{"role": "user", "content": "hi"}])
        assert result.tool_calls is not None
        assert result.tool_calls[0]["id"] == "tc1"

    @pytest.mark.asyncio
    async def test_chat_completions_string_body_raises(self):
        config = _make_config(openai_protocol="chat_completions")
        client = DirectModelClient(config)
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value="not a response")
        client._providers = {Provider.OPENAI: mock_client}
        with patch("src.core.llm.model_client_direct.OpenAIAdapter") as MockAdapter:
            MockAdapter.convert_messages.return_value = [{"role": "user", "content": "hi"}]
            MockAdapter.build_params.return_value = {"model": "gpt-4o"}
            with pytest.raises(RuntimeError, match="string body"):
                await client._chat_openai([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_chat_completions_no_choices(self):
        config = _make_config(openai_protocol="chat_completions")
        client = DirectModelClient(config)
        mock_resp = SimpleNamespace(choices=[], usage=None)
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)
        client._providers = {Provider.OPENAI: mock_client}
        with patch("src.core.llm.model_client_direct.OpenAIAdapter") as MockAdapter:
            MockAdapter.convert_messages.return_value = [{"role": "user", "content": "hi"}]
            MockAdapter.build_params.return_value = {"model": "gpt-4o"}
            result = await client._chat_openai([{"role": "user", "content": "hi"}])
        assert result.content is None
        assert result.finish_reason == "error"

    @pytest.mark.asyncio
    async def test_chat_completions_no_usage(self):
        config = _make_config(openai_protocol="chat_completions")
        client = DirectModelClient(config)
        mock_resp = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)
        client._providers = {Provider.OPENAI: mock_client}
        with patch("src.core.llm.model_client_direct.OpenAIAdapter") as MockAdapter:
            MockAdapter.convert_messages.return_value = [{"role": "user", "content": "hi"}]
            MockAdapter.build_params.return_value = {"model": "gpt-4o"}
            result = await client._chat_openai([{"role": "user", "content": "hi"}])
        assert result.usage is None

    @pytest.mark.asyncio
    async def test_chat_completions_thought_signature(self):
        config = _make_config(openai_protocol="chat_completions")
        client = DirectModelClient(config)
        sig = b"\x01\x02\x03"
        tc = SimpleNamespace(
            id="tc1",
            function=SimpleNamespace(name="fn", arguments='{}', thought_signature=sig),
            thought_signature=None,
        )
        mock_resp = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="", tool_calls=[tc]),
                    finish_reason="tool_calls",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)
        client._providers = {Provider.OPENAI: mock_client}
        with patch("src.core.llm.model_client_direct.OpenAIAdapter") as MockAdapter:
            MockAdapter.convert_messages.return_value = [{"role": "user", "content": "hi"}]
            MockAdapter.build_params.return_value = {"model": "gpt-4o"}
            with patch("src.core.llm.model_client_direct._serialize_gemini_thought_signature", return_value="ser_sig"):
                result = await client._chat_openai([{"role": "user", "content": "hi"}])
        assert result.tool_calls[0]["thought_signature"] == "ser_sig"


class TestChatOpenaiViaResponses:
    @pytest.mark.asyncio
    async def test_responses_success(self):
        config = _make_config(openai_protocol="responses")
        client = DirectModelClient(config)
        parsed = ChatResponse(content="resp", tool_calls=None, finish_reason="stop", usage=None, raw=None)
        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=MagicMock(
            error=None, status="completed", code=None, message=None, output=["something"],
        ))
        client._providers = {Provider.OPENAI: mock_client}
        with patch("src.core.llm.model_client_direct.OpenAIAdapter") as MockAdapter:
            MockAdapter.convert_messages.return_value = [{"role": "user", "content": "hi"}]
            MockAdapter.build_responses_tools.return_value = None
            MockAdapter.build_responses_params.return_value = {"model": "gpt-4o"}
            MockAdapter.summarize_responses_request_params.return_value = ""
            MockAdapter.summarize_responses_payload.return_value = ""
            MockAdapter.parse_responses_response.return_value = parsed
            result = await client._chat_openai([{"role": "user", "content": "hi"}])
        assert result.content == "resp"

    @pytest.mark.asyncio
    async def test_responses_string_body_raises(self):
        config = _make_config(openai_protocol="responses")
        client = DirectModelClient(config)
        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(return_value="bad")
        client._providers = {Provider.OPENAI: mock_client}
        with patch("src.core.llm.model_client_direct.OpenAIAdapter") as MockAdapter:
            MockAdapter.convert_messages.return_value = [{"role": "user", "content": "hi"}]
            MockAdapter.build_responses_tools.return_value = None
            MockAdapter.build_responses_params.return_value = {"model": "gpt-4o"}
            MockAdapter.summarize_responses_request_params.return_value = ""
            with pytest.raises(RuntimeError, match="string body"):
                await client._chat_openai([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_responses_error_status_raises(self):
        config = _make_config(openai_protocol="responses")
        client = DirectModelClient(config)
        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=MagicMock(
            error=None, status="failed", code=None, message=None, output=["something"],
        ))
        client._providers = {Provider.OPENAI: mock_client}
        with patch("src.core.llm.model_client_direct.OpenAIAdapter") as MockAdapter:
            MockAdapter.convert_messages.return_value = [{"role": "user", "content": "hi"}]
            MockAdapter.build_responses_tools.return_value = None
            MockAdapter.build_responses_params.return_value = {"model": "gpt-4o"}
            MockAdapter.summarize_responses_request_params.return_value = ""
            with pytest.raises(RuntimeError, match="failed"):
                await client._chat_openai([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_responses_empty_output_raises(self):
        config = _make_config(openai_protocol="responses")
        client = DirectModelClient(config)
        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=MagicMock(
            error=None, status=None, code=None, message=None, output=None,
        ))
        client._providers = {Provider.OPENAI: mock_client}
        with patch("src.core.llm.model_client_direct.OpenAIAdapter") as MockAdapter:
            MockAdapter.convert_messages.return_value = [{"role": "user", "content": "hi"}]
            MockAdapter.build_responses_tools.return_value = None
            MockAdapter.build_responses_params.return_value = {"model": "gpt-4o"}
            MockAdapter.summarize_responses_request_params.return_value = ""
            with pytest.raises(RuntimeError, match="empty response"):
                await client._chat_openai([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_responses_error_with_message(self):
        config = _make_config(openai_protocol="responses")
        client = DirectModelClient(config)
        err_obj = SimpleNamespace(message="custom error msg")
        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=MagicMock(
            error=err_obj, status=None, code=None, message=None, output=["something"],
        ))
        client._providers = {Provider.OPENAI: mock_client}
        with patch("src.core.llm.model_client_direct.OpenAIAdapter") as MockAdapter:
            MockAdapter.convert_messages.return_value = [{"role": "user", "content": "hi"}]
            MockAdapter.build_responses_tools.return_value = None
            MockAdapter.build_responses_params.return_value = {"model": "gpt-4o"}
            MockAdapter.summarize_responses_request_params.return_value = ""
            with pytest.raises(RuntimeError, match="custom error msg"):
                await client._chat_openai([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_responses_code_raises(self):
        config = _make_config(openai_protocol="responses")
        client = DirectModelClient(config)
        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=MagicMock(
            error=None, status=None, code="ERR", message=None, output=["something"],
        ))
        client._providers = {Provider.OPENAI: mock_client}
        with patch("src.core.llm.model_client_direct.OpenAIAdapter") as MockAdapter:
            MockAdapter.convert_messages.return_value = [{"role": "user", "content": "hi"}]
            MockAdapter.build_responses_tools.return_value = None
            MockAdapter.build_responses_params.return_value = {"model": "gpt-4o"}
            MockAdapter.summarize_responses_request_params.return_value = ""
            with pytest.raises(RuntimeError, match="ERR"):
                await client._chat_openai([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_responses_message_field_raises(self):
        config = _make_config(openai_protocol="responses")
        client = DirectModelClient(config)
        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=MagicMock(
            error=None, status=None, code=None, message="bad things", output=["something"],
        ))
        client._providers = {Provider.OPENAI: mock_client}
        with patch("src.core.llm.model_client_direct.OpenAIAdapter") as MockAdapter:
            MockAdapter.convert_messages.return_value = [{"role": "user", "content": "hi"}]
            MockAdapter.build_responses_tools.return_value = None
            MockAdapter.build_responses_params.return_value = {"model": "gpt-4o"}
            MockAdapter.summarize_responses_request_params.return_value = ""
            with pytest.raises(RuntimeError, match="bad things"):
                await client._chat_openai([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_responses_empty_parsed_result_warns(self):
        config = _make_config(openai_protocol="responses")
        client = DirectModelClient(config)
        parsed = ChatResponse(content=None, tool_calls=None, finish_reason="stop", usage=None, raw=None)
        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=MagicMock(
            error=None, status="completed", code=None, message=None, output=["x"],
        ))
        client._providers = {Provider.OPENAI: mock_client}
        with patch("src.core.llm.model_client_direct.OpenAIAdapter") as MockAdapter:
            MockAdapter.convert_messages.return_value = [{"role": "user", "content": "hi"}]
            MockAdapter.build_responses_tools.return_value = None
            MockAdapter.build_responses_params.return_value = {"model": "gpt-4o"}
            MockAdapter.summarize_responses_request_params.return_value = ""
            MockAdapter.summarize_responses_payload.return_value = ""
            MockAdapter.parse_responses_response.return_value = parsed
            result = await client._chat_openai([{"role": "user", "content": "hi"}])
        assert result.content is None


class TestChatOpenaiFallback:
    @pytest.mark.asyncio
    async def test_fallback_to_chat_completions(self):
        config = _make_config(openai_protocol="auto")
        client = DirectModelClient(config)
        client._openai_protocol = None
        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(side_effect=Exception("404 not found"))
        cc_resp = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="fb", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        mock_client.chat.completions.create = AsyncMock(return_value=cc_resp)
        client._providers = {Provider.OPENAI: mock_client}
        with patch("src.core.llm.model_client_direct.OpenAIAdapter") as MockAdapter:
            MockAdapter.convert_messages.return_value = [{"role": "user", "content": "hi"}]
            MockAdapter.build_responses_tools.return_value = None
            MockAdapter.build_responses_params.return_value = {"model": "gpt-4o"}
            MockAdapter.summarize_responses_request_params.return_value = ""
            MockAdapter.build_params.return_value = {"model": "gpt-4o"}
            MockAdapter._should_fallback_to_chat_completions = DirectModelClient._should_fallback_to_chat_completions
            with patch.object(client, "_should_use_openai_responses_api", return_value=True):
                result = await client._chat_openai([{"role": "user", "content": "hi"}])
        assert result.content == "fb"


class TestChatAnthropic:
    @pytest.mark.asyncio
    async def test_text_and_tool_use_blocks(self):
        config = ClientConfig(model="claude-3", anthropic_api_key="a")
        client = DirectModelClient(config)
        text_block = SimpleNamespace(type="text", text="hello ")
        text_block2 = SimpleNamespace(type="text", text="world")
        tool_block = SimpleNamespace(type="tool_use", id="tu1", name="fn", input={"x": 1})
        mock_resp = SimpleNamespace(
            content=[text_block, text_block2, tool_block],
            usage=SimpleNamespace(input_tokens=5, output_tokens=3),
        )
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_resp)
        client._providers = {Provider.ANTHROPIC: mock_client}
        with patch("src.core.llm.model_client_direct.AnthropicAdapter") as MockAdapter:
            MockAdapter.convert_messages.return_value = ("sys", [])
            MockAdapter.build_params.return_value = {"model": "claude-3"}
            result = await client._chat_anthropic([{"role": "user", "content": "hi"}])
        assert result.content == "hello world"
        assert result.tool_calls is not None
        assert result.tool_calls[0]["id"] == "tu1"
        assert result.finish_reason == "tool_calls"
        assert result.usage["total_tokens"] == 8


class TestChatGoogle:
    @pytest.mark.asyncio
    async def test_no_candidates_returns_error(self):
        config = ClientConfig(model="gemini-2.5-flash", google_api_key="g")
        client = DirectModelClient(config)
        mock_resp = SimpleNamespace(candidates=[])
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_resp)
        client._providers = {Provider.GOOGLE: mock_client}
        with patch("src.core.llm.model_client_direct.GoogleAdapter") as MockAdapter:
            MockAdapter.convert_messages.return_value = (None, [])
            MockAdapter.build_config.return_value = {}
            result = await client._chat_google([{"role": "user", "content": "hi"}])
        assert result.content is None
        assert result.finish_reason == "error"

    @pytest.mark.asyncio
    async def test_with_candidate_and_tool_calls(self):
        config = ClientConfig(model="gemini-2.5-flash", google_api_key="g")
        client = DirectModelClient(config)
        mock_resp = SimpleNamespace(candidates=[SimpleNamespace(finish_reason="STOP")])
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_resp)
        client._providers = {Provider.GOOGLE: mock_client}
        with patch("src.core.llm.model_client_direct.GoogleAdapter") as MockAdapter:
            MockAdapter.convert_messages.return_value = (None, [])
            MockAdapter.build_config.return_value = {}
            MockAdapter.parse_candidate.return_value = ("text", "thought", [{"id": "tc1"}])
            MockAdapter.parse_usage.return_value = {"prompt_tokens": 5, "completion_tokens": 3}
            result = await client._chat_google([{"role": "user", "content": "hi"}])
        assert result.content == "text"
        assert result.finish_reason == "tool_calls"
        assert result.thought == "thought"


class TestStreamOpenaiChatCompletions:
    @pytest.mark.asyncio
    async def test_stream_with_tool_calls_and_usage(self):
        config = _make_config(openai_protocol="chat_completions")
        client = DirectModelClient(config)
        tc_delta = SimpleNamespace(
            index=0,
            id="tc1",
            function=SimpleNamespace(name="fn", arguments='{"a":1}', thought_signature=None),
            thought_signature=None,
        )
        chunk1 = SimpleNamespace(
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content="hi", tool_calls=[tc_delta]),
                finish_reason=None,
            )],
            usage=None,
        )
        chunk2 = SimpleNamespace(
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content=None, tool_calls=None),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        mock_client = MagicMock()

        async def fake_create(**kw):
            return AsyncIterator([chunk1, chunk2])

        class AsyncIterator:
            def __init__(self, items):
                self._items = items
                self._idx = 0
            def __aiter__(self):
                return self
            async def __anext__(self):
                if self._idx >= len(self._items):
                    raise StopAsyncIteration
                item = self._items[self._idx]
                self._idx += 1
                return item

        mock_client.chat.completions.create = fake_create
        client._providers = {Provider.OPENAI: mock_client}
        with patch("src.core.llm.model_client_direct.OpenAIAdapter") as MockAdapter:
            MockAdapter.convert_messages.return_value = [{"role": "user", "content": "hi"}]
            MockAdapter.build_params.return_value = {"model": "gpt-4o"}
            chunks = []
            async for chunk in client._stream_openai([{"role": "user", "content": "hi"}]):
                chunks.append(chunk)
        assert any(c.content == "hi" for c in chunks)
        assert any(c.tool_calls is not None for c in chunks)
        assert any(c.usage is not None for c in chunks)

    @pytest.mark.asyncio
    async def test_stream_empty_choices_skipped(self):
        config = _make_config(openai_protocol="chat_completions")
        client = DirectModelClient(config)
        chunk_empty = SimpleNamespace(choices=[], usage=None)
        chunk_ok = SimpleNamespace(
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content="ok", tool_calls=None),
                finish_reason="stop",
            )],
            usage=None,
        )
        mock_client = MagicMock()

        async def fake_create(**kw):
            return AsyncIterator([chunk_empty, chunk_ok])

        class AsyncIterator:
            def __init__(self, items):
                self._items = items
                self._idx = 0
            def __aiter__(self):
                return self
            async def __anext__(self):
                if self._idx >= len(self._items):
                    raise StopAsyncIteration
                item = self._items[self._idx]
                self._idx += 1
                return item

        mock_client.chat.completions.create = fake_create
        client._providers = {Provider.OPENAI: mock_client}
        with patch("src.core.llm.model_client_direct.OpenAIAdapter") as MockAdapter:
            MockAdapter.convert_messages.return_value = [{"role": "user", "content": "hi"}]
            MockAdapter.build_params.return_value = {"model": "gpt-4o"}
            chunks = []
            async for chunk in client._stream_openai([{"role": "user", "content": "hi"}]):
                chunks.append(chunk)
        assert len(chunks) == 1
        assert chunks[0].content == "ok"


class TestStreamAnthropic:
    @pytest.mark.asyncio
    async def test_text_delta_and_tool_use_and_message_delta(self):
        config = ClientConfig(model="claude-3", anthropic_api_key="a")
        client = DirectModelClient(config)
        events = [
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(text="hi"),
                index=0,
            ),
            SimpleNamespace(
                type="content_block_start",
                content_block=SimpleNamespace(type="tool_use", id="tu1", name="fn"),
                index=1,
            ),
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(partial_json='{"a":1}'),
                index=1,
            ),
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="end_turn"),
                usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            ),
        ]
        mock_stream = MagicMock()
        mock_stream.__aenter__ = AsyncMock(return_self=True)
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.__aiter__ = MagicMock(return_value=iter(events))
        mock_stream.__anext__ = MagicMock(side_effect=[e for e in events] + [StopAsyncIteration])

        async def async_iter():
            for e in events:
                yield e

        mock_client = MagicMock()
        mock_client.messages.stream = MagicMock(return_value=mock_stream)
        mock_stream.__aenter__ = AsyncMock(return_value=async_iter())
        client._providers = {Provider.ANTHROPIC: mock_client}

        with patch("src.core.llm.model_client_direct.AnthropicAdapter") as MockAdapter:
            MockAdapter.convert_messages.return_value = ("sys", [])
            MockAdapter.build_params.return_value = {"model": "claude-3"}
            chunks = []
            async for chunk in client._stream_anthropic([{"role": "user", "content": "hi"}]):
                chunks.append(chunk)
        assert len(chunks) >= 2


class TestStreamGoogle:
    @pytest.mark.asyncio
    async def test_stream_with_candidate(self):
        config = ClientConfig(model="gemini-2.5-flash", google_api_key="g")
        client = DirectModelClient(config)
        resp1 = SimpleNamespace(candidates=[SimpleNamespace(finish_reason="STOP")])
        resp2 = SimpleNamespace(candidates=[])
        mock_client = MagicMock()
        async def fake_stream(*a, **kw):
            yield resp1
            yield resp2
        mock_client.aio.models.generate_content_stream = fake_stream
        client._providers = {Provider.GOOGLE: mock_client}
        with patch("src.core.llm.model_client_direct.GoogleAdapter") as MockAdapter:
            MockAdapter.convert_messages.return_value = (None, [])
            MockAdapter.build_config.return_value = {}
            MockAdapter.parse_candidate.return_value = ("chunk", None, None)
            MockAdapter.parse_usage.return_value = {"prompt_tokens": 5}
            chunks = []
            async for chunk in client._stream_google([{"role": "user", "content": "hi"}]):
                chunks.append(chunk)
        assert len(chunks) == 1
        assert chunks[0].content == "chunk"
        assert chunks[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_stream_with_tool_calls(self):
        config = ClientConfig(model="gemini-2.5-flash", google_api_key="g")
        client = DirectModelClient(config)
        resp1 = SimpleNamespace(candidates=[SimpleNamespace(finish_reason="STOP")])
        mock_client = MagicMock()
        async def fake_stream(*a, **kw):
            yield resp1
        mock_client.aio.models.generate_content_stream = fake_stream
        client._providers = {Provider.GOOGLE: mock_client}
        with patch("src.core.llm.model_client_direct.GoogleAdapter") as MockAdapter:
            MockAdapter.convert_messages.return_value = (None, [])
            MockAdapter.build_config.return_value = {}
            MockAdapter.parse_candidate.return_value = ("", None, [{"id": "tc1"}])
            MockAdapter.parse_usage.return_value = None
            chunks = []
            async for chunk in client._stream_google([{"role": "user", "content": "hi"}]):
                chunks.append(chunk)
        assert chunks[0].finish_reason == "tool_calls"


class TestStreamOpenaiViaResponses:
    @pytest.mark.asyncio
    async def test_text_delta_events(self):
        config = _make_config(openai_protocol="responses")
        client = DirectModelClient(config)
        events_list = [
            SimpleNamespace(type="response.output_text.delta", delta="Hello "),
            SimpleNamespace(type="response.output_text.delta", delta="World"),
            SimpleNamespace(type="response.completed"),
        ]

        class EventStream:
            def __init__(self, items):
                self._items = items
                self._idx = 0
            def __aiter__(self):
                return self
            async def __anext__(self):
                if self._idx >= len(self._items):
                    raise StopAsyncIteration
                item = self._items[self._idx]
                self._idx += 1
                return item

        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=EventStream(events_list))
        client._providers = {Provider.OPENAI: mock_client}

        with patch("src.core.llm.model_client_direct.OpenAIAdapter") as MockAdapter:
            MockAdapter.convert_messages.return_value = [{"role": "user", "content": "hi"}]
            MockAdapter.build_responses_tools.return_value = None
            MockAdapter.build_responses_params.return_value = {"model": "gpt-4o"}
            chunks = []
            async for chunk in client._stream_openai_via_responses(
                mock_client, "gpt-4o", [{"role": "user", "content": "hi"}], None, 0.7
            ):
                chunks.append(chunk)
        assert any(c.content == "Hello " for c in chunks)
        assert any(c.content == "World" for c in chunks)

    @pytest.mark.asyncio
    async def test_tool_call_events(self):
        config = _make_config(openai_protocol="responses")
        client = DirectModelClient(config)
        events_list = [
            SimpleNamespace(
                type="response.output_item.added",
                item={"type": "function_call", "call_id": "c1", "name": "fn"},
            ),
            SimpleNamespace(
                type="response.function_call_arguments.delta",
                delta='{"a":1}',
                call_id="c1",
            ),
            SimpleNamespace(type="response.completed"),
        ]

        class EventStream:
            def __init__(self, items):
                self._items = items
                self._idx = 0
            def __aiter__(self):
                return self
            async def __anext__(self):
                if self._idx >= len(self._items):
                    raise StopAsyncIteration
                item = self._items[self._idx]
                self._idx += 1
                return item

        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=EventStream(events_list))
        client._providers = {Provider.OPENAI: mock_client}

        with patch("src.core.llm.model_client_direct.OpenAIAdapter") as MockAdapter:
            MockAdapter.convert_messages.return_value = [{"role": "user", "content": "hi"}]
            MockAdapter.build_responses_tools.return_value = None
            MockAdapter.build_responses_params.return_value = {"model": "gpt-4o"}
            MockAdapter.serialize_response_output_item.return_value = {"type": "function_call", "call_id": "c1", "name": "fn"}
            chunks = []
            async for chunk in client._stream_openai_via_responses(
                mock_client, "gpt-4o", [{"role": "user", "content": "hi"}], None, 0.7
            ):
                chunks.append(chunk)
        assert any(c.tool_calls is not None and len(c.tool_calls) > 0 for c in chunks)

    @pytest.mark.asyncio
    async def test_failed_event_raises(self):
        config = _make_config(openai_protocol="responses")
        client = DirectModelClient(config)
        events_list = [
            SimpleNamespace(type="response.failed", response="error detail"),
        ]

        class EventStream:
            def __init__(self, items):
                self._items = items
                self._idx = 0
            def __aiter__(self):
                return self
            async def __anext__(self):
                if self._idx >= len(self._items):
                    raise StopAsyncIteration
                item = self._items[self._idx]
                self._idx += 1
                return item

        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=EventStream(events_list))
        client._providers = {Provider.OPENAI: mock_client}

        with patch("src.core.llm.model_client_direct.OpenAIAdapter") as MockAdapter:
            MockAdapter.convert_messages.return_value = [{"role": "user", "content": "hi"}]
            MockAdapter.build_responses_tools.return_value = None
            MockAdapter.build_responses_params.return_value = {"model": "gpt-4o"}
            with pytest.raises(RuntimeError, match="stream failed"):
                async for chunk in client._stream_openai_via_responses(
                    mock_client, "gpt-4o", [{"role": "user", "content": "hi"}], None, 0.7
                ):
                    pass


class TestRetryAsyncLastExcReraise:
    @pytest.mark.asyncio
    async def test_last_exc_reraise_on_exhausted(self, monkeypatch):
        monkeypatch.setattr("src.core.llm.model_client_direct.INITIAL_DELAY", 0.01)
        monkeypatch.setattr("src.core.llm.model_client_direct.MAX_DELAY", 0.01)
        async def always_rate_limit():
            raise Exception("429 rate limit exceeded")
        with pytest.raises(Exception, match="429"):
            await _retry_async(always_rate_limit)
