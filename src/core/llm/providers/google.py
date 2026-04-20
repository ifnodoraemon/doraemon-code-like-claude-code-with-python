"""
Google Gemini Provider Adapter

Converts messages, builds configs, and parses responses for the
Google Gemini (generativelanguage) API.
"""

import base64
import json
import logging
import uuid
from collections.abc import Sequence
from typing import Any

from src.core.llm.model_utils import ToolDefinition, get_content_text

logger = logging.getLogger(__name__)

_GEMINI_THOUGHT_SIGNATURE_PREFIX = "base64:"
_GEMINI_DUMMY_THOUGHT_SIGNATURE = "skip_thought_signature_validator"


def _serialize_gemini_thought_signature(value: Any) -> Any:
    if isinstance(value, bytes):
        return _GEMINI_THOUGHT_SIGNATURE_PREFIX + base64.b64encode(value).decode("ascii")
    return value


def _deserialize_gemini_thought_signature(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(_GEMINI_THOUGHT_SIGNATURE_PREFIX):
        return base64.b64decode(value[len(_GEMINI_THOUGHT_SIGNATURE_PREFIX) :])
    return value


def _strip_json_schema_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_json_schema_metadata(item)
            for key, item in value.items()
            if key != "$schema"
        }
    if isinstance(value, list):
        return [_strip_json_schema_metadata(item) for item in value]
    return value


def build_google_content_parts(content: Any, types_module: Any) -> list[Any]:
    if isinstance(content, str):
        return [types_module.Part(text=content)]
    if isinstance(content, list):
        parts = []
        for part in content:
            if part.get("type") == "text":
                parts.append(types_module.Part(text=part["text"]))
            elif part.get("type") == "image":
                source = part["source"]
                parts.append(
                    types_module.Part.from_bytes(
                        data=base64.b64decode(source["data"]),
                        mime_type=source["media_type"],
                    )
                )
        return parts
    return []


class GoogleAdapter:
    """Converts messages and configs for Google Gemini API."""

    @staticmethod
    def convert_messages(
        messages: Sequence[Any],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> tuple[str | None, list]:
        types_module = kwargs.get("types_module")
        contents = []
        system_instruction = system_prompt

        for m in messages:
            msg = m if isinstance(m, dict) else m.to_dict()
            role = msg.get("role", "user")

            if role == "system":
                system_instruction = get_content_text(msg.get("content", ""))
                continue

            gemini_role = "user" if role in {"user", "tool"} else "model"
            parts = []

            if role != "tool" and msg.get("content"):
                parts.extend(build_google_content_parts(msg["content"], types_module))

            if msg.get("thought"):
                parts.append(types_module.Part(thought=msg["thought"]))

            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    args_str = func.get("arguments", "{}")
                    try:
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse tool arguments: %s", args_str)
                        args = {}

                    fc_obj = types_module.FunctionCall(name=func.get("name"), args=args)
                    thought_sig = tc.get("thought_signature") or func.get("thought_signature")
                    thought_sig = _deserialize_gemini_thought_signature(thought_sig)
                    parts.append(
                        types_module.Part(function_call=fc_obj, thought_signature=thought_sig)
                    )

            if role == "tool" and msg.get("tool_call_id"):
                parts.append(
                    types_module.Part(
                        function_response=types_module.FunctionResponse(
                            name=msg.get("name", "function"),
                            response={"result": msg.get("content", "")},
                        )
                    )
                )

            if parts:
                contents.append(types_module.Content(role=gemini_role, parts=parts))

        return system_instruction, contents

    @staticmethod
    def build_params(
        model: str,
        messages: Any,
        tools: Sequence[ToolDefinition | dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        types_module = kwargs.get("types_module")
        system_instruction = kwargs.get("system_instruction")
        config = GoogleAdapter.build_config(
            tools, system_instruction, temperature, max_tokens, types_module
        )
        return {
            "model": model,
            "contents": messages,
            "config": config,
        }

    @staticmethod
    def build_config(
        tools: Sequence[Any] | None,
        system_instruction: str | None,
        temperature: float,
        max_tokens: int | None,
        types_module: Any,
    ) -> Any:
        gen_config_dict: dict[str, Any] = {
            "temperature": temperature,
        }

        if max_tokens:
            gen_config_dict["max_output_tokens"] = max_tokens

        if tools:
            function_declarations = []
            for t in tools:
                if isinstance(t, ToolDefinition):
                    function_declarations.append(t.to_genai_format())
                else:
                    func = t.get("function", t)
                    sanitized_parameters = _strip_json_schema_metadata(
                        func.get("parameters", {}) or {"type": "object"}
                    )
                    json_schema = types_module.JSONSchema.model_validate(sanitized_parameters)
                    function_declarations.append(
                        types_module.FunctionDeclaration(
                            name=func.get("name"),
                            description=func.get("description", ""),
                            parameters=types_module.Schema.from_json_schema(
                                json_schema=json_schema,
                                api_option="GEMINI_API",
                                raise_error_on_unsupported_field=False,
                            ),
                        )
                    )
            gen_config_dict["tools"] = [
                types_module.Tool(function_declarations=function_declarations)
            ]
            gen_config_dict["automatic_function_calling"] = (
                types_module.AutomaticFunctionCallingConfig(disable=True)
            )

        if system_instruction:
            gen_config_dict["system_instruction"] = system_instruction

        return types_module.GenerateContentConfig(**gen_config_dict)

    @staticmethod
    def parse_candidate(candidate: Any) -> tuple[str | None, str | None, list | None]:
        texts = []
        thoughts = []
        tool_calls = None

        for part in candidate.content.parts:
            if hasattr(part, "text") and part.text:
                texts.append(part.text)
            elif hasattr(part, "thought") and part.thought:
                thoughts.append(part.thought)
            elif hasattr(part, "function_call") and part.function_call:
                if tool_calls is None:
                    tool_calls = []
                fc = part.function_call
                try:
                    args = dict(fc.args) if fc.args else {}
                except (TypeError, ValueError):
                    args = {}
                tc_dict = {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": fc.name,
                        "arguments": json.dumps(args),
                    },
                }
                if hasattr(part, "thought_signature") and part.thought_signature:
                    tc_dict["thought_signature"] = _serialize_gemini_thought_signature(
                        part.thought_signature
                    )
                elif isinstance(part, dict) and part.get("thought_signature"):
                    tc_dict["thought_signature"] = _serialize_gemini_thought_signature(
                        part.get("thought_signature")
                    )
                tool_calls.append(tc_dict)

        content = "".join(texts) if texts else None
        thought = "".join(thoughts) if thoughts else None

        return content, thought, tool_calls

    @staticmethod
    def parse_usage(response: Any) -> dict | None:
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            return {
                "prompt_tokens": response.usage_metadata.prompt_token_count or 0,
                "completion_tokens": response.usage_metadata.candidates_token_count or 0,
                "total_tokens": response.usage_metadata.total_token_count or 0,
            }
        return None
