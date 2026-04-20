"""Tests for src.core.llm.providers.google — GoogleAdapter."""

import base64
import json
from unittest.mock import MagicMock, patch

import pytest

from src.core.llm.providers.google import (
    GoogleAdapter,
    _deserialize_gemini_thought_signature,
    _serialize_gemini_thought_signature,
    _strip_json_schema_metadata,
    build_google_content_parts,
)


class TestSerializeGeminiThoughtSignature:
    def test_bytes_value(self):
        data = b"hello"
        result = _serialize_gemini_thought_signature(data)
        assert result.startswith("base64:")
        assert base64.b64decode(result[7:]) == data

    def test_string_value(self):
        assert _serialize_gemini_thought_signature("abc") == "abc"

    def test_int_value(self):
        assert _serialize_gemini_thought_signature(42) == 42


class TestDeserializeGeminiThoughtSignature:
    def test_base64_prefix(self):
        encoded = "base64:" + base64.b64encode(b"hello").decode("ascii")
        result = _deserialize_gemini_thought_signature(encoded)
        assert result == b"hello"

    def test_plain_string(self):
        assert _deserialize_gemini_thought_signature("abc") == "abc"

    def test_non_string(self):
        assert _deserialize_gemini_thought_signature(42) == 42


class TestStripJsonSchemaMetadata:
    def test_removes_schema_key(self):
        data = {"$schema": "http://...", "type": "object"}
        result = _strip_json_schema_metadata(data)
        assert "$schema" not in result
        assert result["type"] == "object"

    def test_nested(self):
        data = {"outer": {"$schema": "http://...", "key": "val"}}
        result = _strip_json_schema_metadata(data)
        assert "$schema" not in result["outer"]

    def test_list(self):
        data = [{"$schema": "http://...", "name": "a"}, {"name": "b"}]
        result = _strip_json_schema_metadata(data)
        assert "$schema" not in result[0]
        assert result[1]["name"] == "b"

    def test_scalar(self):
        assert _strip_json_schema_metadata("hello") == "hello"
        assert _strip_json_schema_metadata(42) == 42


class TestBuildGoogleContentParts:
    def test_string_content(self):
        mock_types = MagicMock()
        mock_types.Part = MagicMock(return_value=MagicMock())
        result = build_google_content_parts("hello", mock_types)
        mock_types.Part.assert_called_once_with(text="hello")

    def test_list_with_text(self):
        mock_types = MagicMock()
        mock_types.Part = MagicMock(return_value=MagicMock())
        content = [{"type": "text", "text": "hi"}]
        result = build_google_content_parts(content, mock_types)
        mock_types.Part.assert_called_with(text="hi")

    def test_list_with_image(self):
        mock_types = MagicMock()
        mock_part = MagicMock()
        mock_types.Part.from_bytes = MagicMock(return_value=mock_part)
        content = [
            {
                "type": "image",
                "source": {"media_type": "image/png", "data": "aGVsbG8="},
            }
        ]
        result = build_google_content_parts(content, mock_types)
        mock_types.Part.from_bytes.assert_called_once()

    def test_empty_content(self):
        result = build_google_content_parts(42, MagicMock())
        assert result == []


class TestGoogleAdapterConvertMessages:
    def _make_types(self):
        types = MagicMock()
        types.Part = MagicMock(return_value=MagicMock())
        types.Content = MagicMock(return_value=MagicMock())
        types.FunctionCall = MagicMock(return_value=MagicMock())
        types.FunctionResponse = MagicMock(return_value=MagicMock())
        return types

    def test_system_message(self):
        types = self._make_types()
        msgs = [{"role": "system", "content": "sys"}]
        system_instr, contents = GoogleAdapter.convert_messages(
            msgs, types_module=types
        )
        assert system_instr == "sys"
        assert len(contents) == 0

    def test_user_message(self):
        types = self._make_types()
        msgs = [{"role": "user", "content": "hi"}]
        system_instr, contents = GoogleAdapter.convert_messages(
            msgs, types_module=types
        )
        assert len(contents) == 1

    def test_tool_message(self):
        types = self._make_types()
        msgs = [
            {
                "role": "tool",
                "content": "result",
                "tool_call_id": "c1",
                "name": "fn",
            }
        ]
        system_instr, contents = GoogleAdapter.convert_messages(
            msgs, types_module=types
        )
        assert len(contents) == 1
        types.FunctionResponse.assert_called()

    def test_thought_message(self):
        types = self._make_types()
        msgs = [{"role": "user", "content": "hi", "thought": "reasoning"}]
        system_instr, contents = GoogleAdapter.convert_messages(
            msgs, types_module=types
        )
        types.Part.assert_any_call(thought="reasoning")

    def test_tool_calls(self):
        types = self._make_types()
        msgs = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "function": {
                            "name": "fn",
                            "arguments": '{"a": 1}',
                        },
                        "thought_signature": None,
                    }
                ],
            }
        ]
        system_instr, contents = GoogleAdapter.convert_messages(
            msgs, types_module=types
        )
        types.FunctionCall.assert_called()

    def test_tool_calls_invalid_args(self):
        types = self._make_types()
        msgs = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "function": {
                            "name": "fn",
                            "arguments": "not-json",
                        },
                    }
                ],
            }
        ]
        system_instr, contents = GoogleAdapter.convert_messages(
            msgs, types_module=types
        )
        types.FunctionCall.assert_called_with(name="fn", args={})

    def test_object_with_to_dict(self):
        types = self._make_types()

        class Obj:
            def to_dict(self):
                return {"role": "user", "content": "from_obj"}

        system_instr, contents = GoogleAdapter.convert_messages(
            [Obj()], types_module=types
        )
        assert len(contents) == 1


class TestGoogleAdapterParseCandidate:
    def test_text_parts(self):
        part1 = MagicMock()
        part1.text = "hello"
        part1.thought = None
        part1.function_call = None

        candidate = MagicMock()
        candidate.content.parts = [part1]

        content, thought, tool_calls = GoogleAdapter.parse_candidate(candidate)
        assert content == "hello"
        assert thought is None
        assert tool_calls is None

    def test_thought_parts(self):
        part1 = MagicMock()
        part1.text = None
        part1.thought = "reasoning"
        part1.function_call = None

        candidate = MagicMock()
        candidate.content.parts = [part1]

        content, thought, tool_calls = GoogleAdapter.parse_candidate(candidate)
        assert content is None
        assert thought == "reasoning"

    def test_function_call_parts(self):
        fc = MagicMock()
        fc.name = "fn"
        fc.args = {"a": 1}

        part1 = MagicMock()
        part1.text = None
        part1.thought = None
        part1.function_call = fc
        part1.thought_signature = None

        candidate = MagicMock()
        candidate.content.parts = [part1]

        content, thought, tool_calls = GoogleAdapter.parse_candidate(candidate)
        assert content is None
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "fn"

    def test_function_call_with_thought_signature(self):
        fc = MagicMock()
        fc.name = "fn"
        fc.args = {"a": 1}

        part1 = MagicMock()
        part1.text = None
        part1.thought = None
        part1.function_call = fc
        part1.thought_signature = b"sig_data"

        candidate = MagicMock()
        candidate.content.parts = [part1]

        content, thought, tool_calls = GoogleAdapter.parse_candidate(candidate)
        assert tool_calls[0]["thought_signature"].startswith("base64:")

    def test_function_call_dict_with_thought_signature(self):
        fc = MagicMock()
        fc.name = "fn"
        fc.args = {"a": 1}

        part = MagicMock()
        part.text = None
        part.thought = None
        part.function_call = fc
        part.thought_signature = None

        candidate = MagicMock()
        candidate.content.parts = [part]

        content, thought, tool_calls = GoogleAdapter.parse_candidate(candidate)
        assert tool_calls is not None

    def test_function_call_args_error(self):
        fc = MagicMock()
        fc.name = "fn"
        fc.args = MagicMock()
        fc.args.__iter__ = MagicMock(side_effect=TypeError("not iterable"))
        fc.args.items = MagicMock(side_effect=TypeError("not iterable"))

        part = MagicMock()
        part.text = None
        part.thought = None
        part.function_call = fc
        part.thought_signature = None

        candidate = MagicMock()
        candidate.content.parts = [part]

        content, thought, tool_calls = GoogleAdapter.parse_candidate(candidate)
        assert tool_calls[0]["function"]["arguments"] == json.dumps({})


class TestGoogleAdapterParseUsage:
    def test_with_usage(self):
        response = MagicMock()
        response.usage_metadata = MagicMock()
        response.usage_metadata.prompt_token_count = 10
        response.usage_metadata.candidates_token_count = 5
        response.usage_metadata.total_token_count = 15

        usage = GoogleAdapter.parse_usage(response)
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 5
        assert usage["total_tokens"] == 15

    def test_no_usage(self):
        response = MagicMock()
        response.usage_metadata = None
        assert GoogleAdapter.parse_usage(response) is None

    def test_usage_with_none_counts(self):
        response = MagicMock()
        response.usage_metadata = MagicMock()
        response.usage_metadata.prompt_token_count = None
        response.usage_metadata.candidates_token_count = None
        response.usage_metadata.total_token_count = None

        usage = GoogleAdapter.parse_usage(response)
        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 0


class TestGoogleAdapterBuildParams:
    def test_basic_params(self):
        types = MagicMock()
        params = GoogleAdapter.build_params(
            model="gemini-3",
            messages=[],
            types_module=types,
        )
        assert params["model"] == "gemini-3"
        assert "config" in params


class TestGoogleAdapterBuildConfig:
    def test_with_tools(self):
        from src.core.llm.model_utils import ToolDefinition

        types = MagicMock()
        types.JSONSchema.model_validate = MagicMock(return_value=MagicMock())
        types.Schema.from_json_schema = MagicMock(return_value=MagicMock())
        types.FunctionDeclaration = MagicMock(return_value=MagicMock())
        types.Tool = MagicMock(return_value=MagicMock())
        types.AutomaticFunctionCallingConfig = MagicMock(return_value=MagicMock())
        types.GenerateContentConfig = MagicMock(return_value=MagicMock())

        tools = [
            ToolDefinition(name="fn", description="d", parameters={"type": "object"})
        ]
        config = GoogleAdapter.build_config(
            tools, None, 0.7, 100, types
        )
        types.GenerateContentConfig.assert_called_once()

    def test_with_system_instruction(self):
        types = MagicMock()
        types.GenerateContentConfig = MagicMock(return_value=MagicMock())

        config = GoogleAdapter.build_config(
            None, "sys", 0.7, 100, types
        )
        call_kwargs = types.GenerateContentConfig.call_args[1]
        assert call_kwargs["system_instruction"] == "sys"

    def test_with_max_tokens(self):
        types = MagicMock()
        types.GenerateContentConfig = MagicMock(return_value=MagicMock())

        config = GoogleAdapter.build_config(None, None, 0.7, 200, types)
        call_kwargs = types.GenerateContentConfig.call_args[1]
        assert call_kwargs["max_output_tokens"] == 200

    def test_no_max_tokens(self):
        types = MagicMock()
        types.GenerateContentConfig = MagicMock(return_value=MagicMock())

        config = GoogleAdapter.build_config(None, None, 0.7, None, types)
        call_kwargs = types.GenerateContentConfig.call_args[1]
        assert "max_output_tokens" not in call_kwargs

    def test_with_dict_tools(self):
        types = MagicMock()
        types.JSONSchema.model_validate = MagicMock(return_value=MagicMock())
        types.Schema.from_json_schema = MagicMock(return_value=MagicMock())
        types.FunctionDeclaration = MagicMock(return_value=MagicMock())
        types.Tool = MagicMock(return_value=MagicMock())
        types.AutomaticFunctionCallingConfig = MagicMock(return_value=MagicMock())
        types.GenerateContentConfig = MagicMock(return_value=MagicMock())

        tools = [
            {
                "function": {
                    "name": "fn",
                    "description": "d",
                    "parameters": {"type": "object"},
                }
            }
        ]
        config = GoogleAdapter.build_config(tools, None, 0.7, None, types)
        types.FunctionDeclaration.assert_called_once()
