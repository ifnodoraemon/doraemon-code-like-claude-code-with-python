"""Tests for src.core.llm.providers.openai — OpenAIAdapter."""

import json
from unittest.mock import MagicMock

import pytest

from src.core.llm.model_utils import ChatResponse, ToolDefinition
from src.core.llm.providers.openai import (
    OpenAIAdapter,
    build_openai_content_parts,
)


class TestBuildOpenaiContentParts:
    def test_string_passthrough(self):
        assert build_openai_content_parts("hello") == "hello"

    def test_list_with_text(self):
        content = [{"type": "text", "text": "hi"}]
        result = build_openai_content_parts(content)
        assert result == [{"type": "text", "text": "hi"}]

    def test_list_with_image(self):
        content = [
            {
                "type": "image",
                "source": {
                    "media_type": "image/png",
                    "data": "base64data",
                },
            }
        ]
        result = build_openai_content_parts(content)
        assert len(result) == 1
        assert result[0]["type"] == "image_url"
        assert "data:image/png;base64,base64data" in result[0]["image_url"]["url"]

    def test_non_string_non_list(self):
        result = build_openai_content_parts(None)
        assert result == ""

    def test_other_type(self):
        result = build_openai_content_parts(42)
        assert result == 42


class TestOpenAIAdapterConvertMessages:
    def test_simple_message(self):
        msgs = [{"role": "user", "content": "hi"}]
        result = OpenAIAdapter.convert_messages(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "hi"

    def test_multimodal_content(self):
        msgs = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "describe this"}],
            }
        ]
        result = OpenAIAdapter.convert_messages(msgs)
        assert result[0]["content"] == [{"type": "text", "text": "describe this"}]

    def test_object_with_to_dict(self):
        class Obj:
            def to_dict(self):
                return {"role": "user", "content": "from_obj"}

        result = OpenAIAdapter.convert_messages([Obj()])
        assert result[0]["content"] == "from_obj"


class TestOpenAIAdapterBuildParams:
    def test_basic_params(self):
        params = OpenAIAdapter.build_params(
            model="gpt-4o",
            msg_list=[{"role": "user", "content": "hi"}],
        )
        assert params["model"] == "gpt-4o"
        assert params["temperature"] == 0.7
        assert "stream" not in params

    def test_stream_mode(self):
        params = OpenAIAdapter.build_params(
            model="gpt-4o",
            msg_list=[],
            stream=True,
        )
        assert params["stream"] is True

    def test_with_tools(self):
        tools = [
            ToolDefinition(name="fn", description="d", parameters={"type": "object"})
        ]
        params = OpenAIAdapter.build_params(
            model="gpt-4o",
            msg_list=[],
            tools=tools,
        )
        assert len(params["tools"]) == 1
        assert params["tools"][0]["type"] == "function"
        assert params["tools"][0]["function"]["name"] == "fn"

    def test_with_dict_tools(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "fn",
                    "description": "d",
                    "parameters": {"type": "object"},
                },
            }
        ]
        params = OpenAIAdapter.build_params(
            model="gpt-4o",
            msg_list=[],
            tools=tools,
        )
        assert len(params["tools"]) == 1

    def test_max_tokens(self):
        params = OpenAIAdapter.build_params(
            model="gpt-4o",
            msg_list=[],
            max_tokens=100,
        )
        assert params["max_tokens"] == 100

    def test_gemini_thought_signature_workaround(self):
        msg_list = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {"name": "fn", "arguments": "{}"},
                    }
                ],
            }
        ]
        params = OpenAIAdapter.build_params(
            model="gemini-3-flash",
            msg_list=msg_list,
        )
        tool_calls = params["messages"][0]["tool_calls"]
        assert tool_calls[0].get("thought_signature") is not None

    def test_gemini_with_existing_thought_signature(self):
        msg_list = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "thought_signature": "existing_sig",
                        "function": {
                            "name": "fn",
                            "arguments": "{}",
                            "thought_signature": "func_sig",
                        },
                    }
                ],
            }
        ]
        params = OpenAIAdapter.build_params(
            model="gemini-3-flash",
            msg_list=msg_list,
        )
        tool_calls = params["messages"][0]["tool_calls"]
        assert tool_calls[0]["thought_signature"] == "existing_sig"


class TestOpenAIAdapterBuildResponsesTools:
    def test_none_tools(self):
        assert OpenAIAdapter.build_responses_tools(None) is None

    def test_empty_tools(self):
        assert OpenAIAdapter.build_responses_tools([]) is None

    def test_function_tool(self):
        tools = [
            ToolDefinition(name="fn", description="d", parameters={"type": "object"})
        ]
        result = OpenAIAdapter.build_responses_tools(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["name"] == "fn"

    def test_non_function_tool(self):
        tools = [{"type": "web_search", "name": "search"}]
        result = OpenAIAdapter.build_responses_tools(tools)
        assert len(result) == 1
        assert result[0]["type"] == "web_search"


class TestOpenAIAdapterBuildResponsesParams:
    def test_basic_params(self):
        params = OpenAIAdapter.build_responses_params(
            model="gpt-4o",
            msg_list=[{"role": "user", "content": "hi"}],
            tools=None,
            temperature=0.5,
            max_output_tokens=100,
        )
        assert params["model"] == "gpt-4o"
        assert params["store"] is False
        assert len(params["input"]) == 1

    def test_assistant_message(self):
        params = OpenAIAdapter.build_responses_params(
            model="gpt-4o",
            msg_list=[
                {"role": "assistant", "content": "response text"},
            ],
            tools=None,
            temperature=0.7,
            max_output_tokens=None,
        )
        assert params["input"][0]["type"] == "message"
        assert params["input"][0]["role"] == "assistant"

    def test_assistant_with_tool_calls(self):
        params = OpenAIAdapter.build_responses_params(
            model="gpt-4o",
            msg_list=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {
                                "name": "fn",
                                "arguments": '{"a": 1}',
                            },
                        }
                    ],
                }
            ],
            tools=None,
            temperature=0.7,
            max_output_tokens=None,
        )
        function_call_item = params["input"][0]
        assert function_call_item["type"] == "function_call"

    def test_tool_message(self):
        params = OpenAIAdapter.build_responses_params(
            model="gpt-4o",
            msg_list=[
                {"role": "tool", "content": "result", "tool_call_id": "c1"},
            ],
            tools=None,
            temperature=0.7,
            max_output_tokens=None,
        )
        assert params["input"][0]["type"] == "function_call_output"

    def test_with_provider_items(self):
        params = OpenAIAdapter.build_responses_params(
            model="gpt-4o",
            msg_list=[
                {
                    "role": "assistant",
                    "content": "x",
                    "provider_items": [{"type": "reasoning", "id": "r1"}],
                }
            ],
            tools=None,
            temperature=0.7,
            max_output_tokens=None,
        )
        assert params["input"][0]["type"] == "reasoning"

    def test_with_tools(self):
        tools = [{"type": "function", "name": "fn"}]
        params = OpenAIAdapter.build_responses_params(
            model="gpt-4o",
            msg_list=[],
            tools=tools,
            temperature=0.7,
            max_output_tokens=None,
        )
        assert len(params["tools"]) == 1

    def test_with_include(self):
        params = OpenAIAdapter.build_responses_params(
            model="gpt-4o",
            msg_list=[],
            tools=None,
            temperature=0.7,
            max_output_tokens=None,
            include=["reasoning.encrypted_content"],
        )
        assert params["include"] == ["reasoning.encrypted_content"]

    def test_stream_mode(self):
        params = OpenAIAdapter.build_responses_params(
            model="gpt-4o",
            msg_list=[],
            tools=None,
            temperature=0.7,
            max_output_tokens=None,
            stream=True,
        )
        assert params["stream"] is True

    def test_multimodal_user_content(self):
        params = OpenAIAdapter.build_responses_params(
            model="gpt-4o",
            msg_list=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,abc"},
                        },
                    ],
                }
            ],
            tools=None,
            temperature=0.7,
            max_output_tokens=None,
        )
        item = params["input"][0]
        assert len(item["content"]) == 2
        assert item["content"][1]["type"] == "input_image"

    def test_non_string_content_value(self):
        params = OpenAIAdapter.build_responses_params(
            model="gpt-4o",
            msg_list=[{"role": "user", "content": 42}],
            tools=None,
            temperature=0.7,
            max_output_tokens=None,
        )
        assert params["input"][0]["content"][0]["text"] == "42"


class TestOpenAIAdapterParseResponsesResponse:
    def test_string_raises(self):
        with pytest.raises(RuntimeError, match="non-OpenAI responses payload"):
            OpenAIAdapter.parse_responses_response("not a response")

    def test_basic_response(self):
        response = MagicMock()
        response.output_text = "hello"
        response.output = []
        response.usage = None
        response.status = "completed"

        result = OpenAIAdapter.parse_responses_response(response)
        assert result.content == "hello"
        assert result.finish_reason == "completed"

    def test_with_usage(self):
        response = MagicMock()
        response.output_text = "hi"
        response.output = []
        response.usage = MagicMock()
        response.usage.input_tokens = 10
        response.usage.output_tokens = 5
        response.status = "completed"

        result = OpenAIAdapter.parse_responses_response(response)
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["total_tokens"] == 15

    def test_with_function_call(self):
        fc_item = MagicMock()
        fc_item.type = "function_call"
        fc_item.call_id = "c1"
        fc_item.name = "fn"
        fc_item.arguments = '{"a": 1}'

        response = MagicMock()
        response.output_text = None
        response.output = [fc_item]
        response.usage = None
        response.status = "completed"

        result = OpenAIAdapter.parse_responses_response(response)
        assert result.tool_calls is not None
        assert result.tool_calls[0]["function"]["name"] == "fn"

    def test_with_message_item(self):
        msg_item = MagicMock()
        msg_item.type = "message"
        text_part = MagicMock()
        text_part.type = "output_text"
        text_part.text = "from message"
        msg_item.content = [text_part]

        response = MagicMock()
        type(response).output_text = property(lambda self: None)
        response.output = [msg_item]
        response.usage = None
        response.status = "completed"

        result = OpenAIAdapter.parse_responses_response(response)
        assert result.content == "from message"


class TestOpenAIAdapterSerializeResponseOutputItem:
    def test_with_model_dump(self):
        item = MagicMock()
        item.model_dump = MagicMock(return_value={"type": "custom"})
        result = OpenAIAdapter.serialize_response_output_item(item)
        assert result == {"type": "custom"}

    def test_dict_passthrough(self):
        item = {"type": "test"}
        result = OpenAIAdapter.serialize_response_output_item(item)
        assert result == {"type": "test"}

    def test_function_call(self):
        item = MagicMock()
        item.model_dump = None
        del item.model_dump
        item.type = "function_call"
        item.call_id = "c1"
        item.name = "fn"
        item.arguments = '{"a": 1}'

        result = OpenAIAdapter.serialize_response_output_item(item)
        assert result["type"] == "function_call"
        assert result["name"] == "fn"

    def test_reasoning_item(self):
        item = MagicMock()
        item.model_dump = None
        del item.model_dump
        item.type = "reasoning"
        item.summary = "thinking"
        item.encrypted_content = "enc"
        item.id = "r1"

        result = OpenAIAdapter.serialize_response_output_item(item)
        assert result["type"] == "reasoning"
        assert result["summary"] == "thinking"

    def test_message_item(self):
        part = MagicMock()
        part.type = "output_text"
        part.text = "hello"

        item = MagicMock()
        item.model_dump = None
        del item.model_dump
        item.type = "message"
        item.role = "assistant"
        item.content = [part]

        result = OpenAIAdapter.serialize_response_output_item(item)
        assert result["type"] == "message"

    def test_unknown_type(self):
        item = MagicMock()
        item.model_dump = None
        del item.model_dump
        item.type = "unknown_type"

        result = OpenAIAdapter.serialize_response_output_item(item)
        assert result is None


class TestSummarizeResponsesRequestParams:
    def test_basic(self):
        params = {
            "model": "gpt-4o",
            "input": [{"type": "message", "role": "user"}],
            "tools": [{"name": "fn"}],
            "stream": True,
            "store": False,
            "include": ["reasoning"],
        }
        result = OpenAIAdapter.summarize_responses_request_params(params)
        assert result["model"] == "gpt-4o"
        assert result["tool_count"] == 1
        assert result["tool_names"] == ["fn"]
        assert result["input_count"] == 1


class TestSummarizeResponsesPayload:
    def test_basic(self):
        response = MagicMock()
        response.id = "r1"
        response.status = "completed"
        response.output_text = "hello"
        response.output = []
        response.usage = MagicMock()
        response.usage.input_tokens = 10
        response.usage.output_tokens = 5

        result = OpenAIAdapter.summarize_responses_payload(response)
        assert result["id"] == "r1"
        assert result["output_text_present"] is True
        assert result["usage"]["input_tokens"] == 10

    def test_with_include_items(self):
        item = MagicMock()
        item.model_dump = MagicMock(return_value={"type": "text"})

        response = MagicMock()
        response.id = "r1"
        response.status = "completed"
        response.output_text = None
        response.output = [item]
        response.usage = None

        result = OpenAIAdapter.summarize_responses_payload(
            response, include_items=True
        )
        assert "output_items" in result
