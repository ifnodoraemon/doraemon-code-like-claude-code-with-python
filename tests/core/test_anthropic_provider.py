"""Tests for src.core.llm.providers.anthropic — AnthropicAdapter."""

import json
from unittest.mock import MagicMock

import pytest

from src.core.llm.providers.anthropic import (
    AnthropicAdapter,
    build_anthropic_content_parts,
)


class TestBuildAnthropicContentParts:
    def test_string_content(self):
        result = build_anthropic_content_parts("hello")
        assert result == [{"type": "text", "text": "hello"}]

    def test_list_with_text(self):
        content = [{"type": "text", "text": "hi"}]
        result = build_anthropic_content_parts(content)
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
        result = build_anthropic_content_parts(content)
        assert len(result) == 1
        assert result[0]["type"] == "image"
        assert result[0]["source"]["type"] == "base64"
        assert result[0]["source"]["media_type"] == "image/png"

    def test_non_string_content(self):
        result = build_anthropic_content_parts(42)
        assert result == [{"type": "text", "text": "42"}]

    def test_none_content(self):
        result = build_anthropic_content_parts(None)
        assert result == [{"type": "text", "text": ""}]


class TestAnthropicAdapterConvertMessages:
    def test_system_message(self):
        msgs = [{"role": "system", "content": "sys prompt"}]
        system, msg_list = AnthropicAdapter.convert_messages(msgs)
        assert system == "sys prompt"
        assert len(msg_list) == 0

    def test_tool_message(self):
        msgs = [
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": "tool result",
            }
        ]
        system, msg_list = AnthropicAdapter.convert_messages(msgs)
        assert len(msg_list) == 1
        assert msg_list[0]["role"] == "user"
        assert msg_list[0]["content"][0]["type"] == "tool_result"
        assert msg_list[0]["content"][0]["tool_use_id"] == "c1"

    def test_assistant_with_tool_calls(self):
        msgs = [
            {
                "role": "assistant",
                "content": "thinking",
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
        ]
        system, msg_list = AnthropicAdapter.convert_messages(msgs)
        assert len(msg_list) == 1
        assert msg_list[0]["role"] == "assistant"
        assert msg_list[0]["content"][0]["type"] == "text"
        assert msg_list[0]["content"][1]["type"] == "tool_use"
        assert msg_list[0]["content"][1]["input"] == {"a": 1}

    def test_assistant_with_invalid_tool_args(self):
        msgs = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {
                            "name": "fn",
                            "arguments": "not-json",
                        },
                    }
                ],
            }
        ]
        system, msg_list = AnthropicAdapter.convert_messages(msgs)
        assert msg_list[0]["content"][0]["input"] == {}

    def test_user_message_with_content(self):
        msgs = [{"role": "user", "content": "hello"}]
        system, msg_list = AnthropicAdapter.convert_messages(msgs)
        assert len(msg_list) == 1
        assert msg_list[0]["role"] == "user"
        assert msg_list[0]["content"] == [{"type": "text", "text": "hello"}]

    def test_user_message_no_content(self):
        msgs = [{"role": "user", "content": ""}]
        system, msg_list = AnthropicAdapter.convert_messages(msgs)
        assert len(msg_list) == 1
        assert msg_list[0]["content"] == ""

    def test_system_prompt_override(self):
        msgs = [{"role": "user", "content": "hi"}]
        system, msg_list = AnthropicAdapter.convert_messages(
            msgs, system_prompt="override"
        )
        assert system == "override"

    def test_object_with_to_dict(self):
        class Obj:
            def to_dict(self):
                return {"role": "user", "content": "from_obj"}

        system, msg_list = AnthropicAdapter.convert_messages([Obj()])
        assert msg_list[0]["content"] == [{"type": "text", "text": "from_obj"}]

    def test_assistant_with_dict_args(self):
        msgs = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {
                            "name": "fn",
                            "arguments": {"a": 1},
                        },
                    }
                ],
            }
        ]
        system, msg_list = AnthropicAdapter.convert_messages(msgs)
        assert msg_list[0]["content"][0]["input"] == {"a": 1}


class TestAnthropicAdapterBuildParams:
    def test_basic_params(self):
        params = AnthropicAdapter.build_params(
            model="claude-3",
            msg_list=[{"role": "user", "content": "hi"}],
        )
        assert params["model"] == "claude-3"
        assert params["max_tokens"] == 4096
        assert "system" not in params

    def test_with_system(self):
        params = AnthropicAdapter.build_params(
            model="claude-3",
            msg_list=[],
            system="sys prompt",
        )
        assert params["system"] == "sys prompt"

    def test_with_tools(self):
        from src.core.llm.model_utils import ToolDefinition

        tools = [
            ToolDefinition(name="fn1", description="d", parameters={"type": "object"})
        ]
        params = AnthropicAdapter.build_params(
            model="claude-3",
            msg_list=[],
            tools=tools,
        )
        assert len(params["tools"]) == 1
        assert params["tools"][0]["name"] == "fn1"
        assert params["tools"][0]["input_schema"] == {"type": "object"}

    def test_with_dict_tools(self):
        tools = [
            {
                "function": {
                    "name": "fn1",
                    "description": "d",
                    "parameters": {"type": "object"},
                }
            }
        ]
        params = AnthropicAdapter.build_params(
            model="claude-3",
            msg_list=[],
            tools=tools,
        )
        assert len(params["tools"]) == 1
        assert params["tools"][0]["name"] == "fn1"

    def test_max_tokens_override(self):
        params = AnthropicAdapter.build_params(
            model="claude-3",
            msg_list=[],
            max_tokens=8192,
        )
        assert params["max_tokens"] == 8192
