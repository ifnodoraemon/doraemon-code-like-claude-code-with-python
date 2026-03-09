"""
Provider Adapters

Extracts the message conversion, config building, and response parsing logic
from DirectModelClient into reusable adapter classes. This eliminates
duplication between _chat_xxx and _stream_xxx methods.
"""

import json
import logging
import uuid
from collections.abc import Sequence
from typing import Any

from src.core.model_utils import (
    ToolDefinition,
    get_content_text,
)

logger = logging.getLogger(__name__)


# ========================================
# Google Gemini Adapter
# ========================================


class GoogleAdapter:
    """Converts messages and configs for Google Gemini API."""

    @staticmethod
    def convert_messages(
        messages: Sequence[Any],
        system_prompt: str | None,
        types_module: Any,
    ) -> tuple[str | None, list]:
        """
        Convert unified messages to Google Gemini format.

        Args:
            messages: Sequence of Message or dict objects
            system_prompt: Default system prompt
            types_module: google.genai.types module

        Returns:
            Tuple of (system_instruction, contents list)
        """
        from src.core.model_client_direct import _build_google_content_parts

        contents = []
        system_instruction = system_prompt

        for m in messages:
            msg = m if isinstance(m, dict) else m.to_dict()
            role = msg.get("role", "user")

            if role == "system":
                system_instruction = get_content_text(msg.get("content", ""))
                continue

            gemini_role = "user" if role == "user" else "model"
            parts = []

            if msg.get("content"):
                parts.extend(_build_google_content_parts(msg["content"], types_module))

            if msg.get("thought"):
                parts.append(types_module.Part(thought=msg["thought"]))

            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    args_str = func.get("arguments", "{}")
                    try:
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse tool arguments: {args_str}")
                        args = {}

                    fc_obj = types_module.FunctionCall(name=func.get("name"), args=args)
                    thought_sig = tc.get("thought_signature") or func.get("thought_signature")
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
    def build_config(
        tools: Sequence[Any] | None,
        system_instruction: str | None,
        temperature: float,
        max_tokens: int | None,
        types_module: Any,
    ) -> Any:
        """
        Build Google GenAI GenerateContentConfig.

        Returns:
            types.GenerateContentConfig instance
        """
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
                    function_declarations.append(
                        types_module.FunctionDeclaration(
                            name=func.get("name"),
                            description=func.get("description", ""),
                            parameters=func.get("parameters", {}),
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
        """
        Parse a Google Gemini response candidate.

        Returns:
            Tuple of (content_text, thought_text, tool_calls_list)
        """
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
                    tc_dict["thought_signature"] = part.thought_signature
                elif isinstance(part, dict) and part.get("thought_signature"):
                    tc_dict["thought_signature"] = part.get("thought_signature")
                tool_calls.append(tc_dict)

        content = "".join(texts) if texts else None
        thought = "".join(thoughts) if thoughts else None

        return content, thought, tool_calls

    @staticmethod
    def parse_usage(response: Any) -> dict | None:
        """Extract usage metadata from a Google response."""
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            return {
                "prompt_tokens": response.usage_metadata.prompt_token_count or 0,
                "completion_tokens": response.usage_metadata.candidates_token_count or 0,
                "total_tokens": response.usage_metadata.total_token_count or 0,
            }
        return None


# ========================================
# OpenAI Adapter
# ========================================


class OpenAIAdapter:
    """Converts messages and configs for OpenAI API."""

    @staticmethod
    def convert_messages(messages: Sequence[Any]) -> list[dict]:
        """Convert unified messages to OpenAI format."""
        from src.core.model_client_direct import _build_openai_content_parts

        msg_list = []
        for m in messages:
            msg = m if isinstance(m, dict) else m.to_dict()
            if isinstance(msg.get("content"), list):
                msg = {**msg, "content": _build_openai_content_parts(msg["content"])}
            msg_list.append(msg)
        return msg_list

    @staticmethod
    def build_params(
        model: str,
        msg_list: list[dict],
        tools: Sequence[Any] | None,
        temperature: float,
        max_tokens: int | None,
        stream: bool = False,
    ) -> dict:
        """Build OpenAI API request parameters."""
        params: dict[str, Any] = {
            "model": model,
            "messages": msg_list,
            "temperature": temperature,
        }
        if stream:
            params["stream"] = True
        if tools:
            params["tools"] = [
                t.to_openai_format() if isinstance(t, ToolDefinition) else t for t in tools
            ]
        if max_tokens:
            params["max_tokens"] = max_tokens
        return params


# ========================================
# Anthropic Adapter
# ========================================


class AnthropicAdapter:
    """Converts messages and configs for Anthropic Claude API."""

    @staticmethod
    def convert_messages(
        messages: Sequence[Any],
        system_prompt: str | None,
    ) -> tuple[str | None, list[dict]]:
        """
        Convert unified messages to Anthropic format.

        Returns:
            Tuple of (system_text, msg_list)
        """
        from src.core.model_client_direct import _build_anthropic_content_parts

        system = system_prompt
        msg_list = []

        for m in messages:
            msg = m if isinstance(m, dict) else m.to_dict()
            role = msg.get("role", "user")

            if role == "system":
                system = get_content_text(msg.get("content", ""))
                continue

            if role == "tool":
                msg_list.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.get("tool_call_id"),
                                "content": get_content_text(msg.get("content", "")),
                            }
                        ],
                    }
                )
            else:
                content = []
                if msg.get("content"):
                    content.extend(_build_anthropic_content_parts(msg["content"]))
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        func = tc.get("function", {})
                        args_str = func.get("arguments", "{}")
                        try:
                            args = json.loads(args_str) if isinstance(args_str, str) else args_str
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse tool arguments: {args_str}")
                            args = {}
                        content.append(
                            {
                                "type": "tool_use",
                                "id": tc.get("id"),
                                "name": func.get("name"),
                                "input": args,
                            }
                        )
                msg_list.append(
                    {
                        "role": "assistant" if role == "assistant" else "user",
                        "content": content or msg.get("content"),
                    }
                )

        return system, msg_list

    @staticmethod
    def build_params(
        model: str,
        msg_list: list[dict],
        tools: Sequence[Any] | None,
        system: str | None,
        max_tokens: int | None,
    ) -> dict:
        """Build Anthropic API request parameters."""
        params: dict[str, Any] = {
            "model": model,
            "messages": msg_list,
            "max_tokens": max_tokens or 4096,
        }
        if system:
            params["system"] = system
        if tools:
            params["tools"] = [
                {
                    "name": t.name if isinstance(t, ToolDefinition) else t["function"]["name"],
                    "description": t.description
                    if isinstance(t, ToolDefinition)
                    else t["function"].get("description", ""),
                    "input_schema": t.parameters
                    if isinstance(t, ToolDefinition)
                    else t["function"].get("parameters", {}),
                }
                for t in tools
            ]
        return params
