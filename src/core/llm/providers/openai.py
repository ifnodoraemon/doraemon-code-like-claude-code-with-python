"""
OpenAI Provider Adapter

Converts messages and builds request parameters for the OpenAI API
(Chat Completions and Responses endpoints).
"""

import json
from collections.abc import Sequence
from typing import Any

from src.core.llm.model_utils import ChatResponse, ToolDefinition

_GEMINI_DUMMY_THOUGHT_SIGNATURE = "skip_thought_signature_validator"


def build_openai_content_parts(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if part.get("type") == "text":
                parts.append({"type": "text", "text": part["text"]})
            elif part.get("type") == "image":
                source = part["source"]
                data_url = f"data:{source['media_type']};base64,{source['data']}"
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    }
                )
        return parts
    return content or ""


class OpenAIAdapter:
    """Converts messages and configs for OpenAI API."""

    @staticmethod
    def convert_messages(
        messages: Sequence[Any],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> list[dict]:
        msg_list = []
        for m in messages:
            msg = m if isinstance(m, dict) else m.to_dict()
            if isinstance(msg.get("content"), list):
                msg = {**msg, "content": build_openai_content_parts(msg["content"])}
            msg_list.append(msg)
        return msg_list

    @staticmethod
    def build_params(
        model: str,
        msg_list: Any,
        tools: Sequence[ToolDefinition | dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict:
        normalized_messages: list[dict[str, Any]] = []
        use_gemini_thought_signature_workaround = model.startswith("gemini-3")

        for msg in msg_list:
            normalized = dict(msg)
            tool_calls = normalized.get("tool_calls")
            if (
                use_gemini_thought_signature_workaround
                and normalized.get("role") == "assistant"
                and tool_calls
            ):
                normalized_tool_calls = []
                first_call_needs_signature = True
                for tool_call in tool_calls:
                    normalized_call = dict(tool_call)
                    function = dict(normalized_call.get("function") or {})
                    thought_signature = normalized_call.get("thought_signature") or function.get(
                        "thought_signature"
                    )
                    if first_call_needs_signature and not thought_signature:
                        thought_signature = _GEMINI_DUMMY_THOUGHT_SIGNATURE
                    if thought_signature:
                        normalized_call["thought_signature"] = thought_signature
                        function["thought_signature"] = thought_signature
                    if function:
                        normalized_call["function"] = function
                    normalized_tool_calls.append(normalized_call)
                    first_call_needs_signature = False
                normalized["tool_calls"] = normalized_tool_calls
            normalized_messages.append(normalized)

        params: dict[str, Any] = {
            "model": model,
            "messages": normalized_messages,
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

    @staticmethod
    def build_responses_tools(
        tools: Sequence[ToolDefinition | dict] | None,
    ) -> list[dict] | None:
        """Convert chat-completions style tools into Responses API tools."""
        if not tools:
            return None

        response_tools: list[dict] = []
        for tool in tools:
            formatted = tool.to_openai_format() if isinstance(tool, ToolDefinition) else tool
            if formatted.get("type") != "function":
                response_tools.append(formatted)
                continue

            function = formatted.get("function", {})
            response_tools.append(
                {
                    "type": "function",
                    "name": function.get("name"),
                    "description": function.get("description", ""),
                    "parameters": function.get("parameters", {}),
                    "strict": False,
                }
            )
        return response_tools

    @staticmethod
    def build_responses_params(
        *,
        model: str,
        msg_list: list[dict],
        tools: list[dict] | None,
        temperature: float,
        max_output_tokens: int | None,
        include: Sequence[str] | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Build request params for the OpenAI Responses API."""
        input_items: list[dict[str, Any]] = []
        for msg in msg_list:
            role = msg.get("role", "user")
            content = msg.get("content")
            tool_calls = msg.get("tool_calls") or []
            provider_items = msg.get("provider_items") or []

            def _build_content_parts(
                content_value: Any, *, assistant: bool = False
            ) -> list[dict[str, Any]]:
                text_type = "output_text" if assistant else "input_text"
                if isinstance(content_value, str):
                    return [{"type": text_type, "text": content_value}]
                if isinstance(content_value, list):
                    content_parts: list[dict[str, Any]] = []
                    for part in content_value:
                        part_type = part.get("type")
                        if part_type == "text":
                            content_parts.append({"type": text_type, "text": part.get("text", "")})
                        elif part_type == "image_url":
                            content_parts.append(
                                {
                                    "type": "input_image",
                                    "image_url": part.get("image_url", {}).get("url", ""),
                                }
                            )
                    return content_parts
                return [{"type": text_type, "text": str(content_value or "")}]

            if role == "assistant":
                if provider_items:
                    input_items.extend(provider_items)
                    continue
                if content:
                    input_items.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": _build_content_parts(content, assistant=True),
                        }
                    )
                for tool_call in tool_calls:
                    function = tool_call.get("function", {})
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": tool_call.get("id") or "",
                            "name": function.get("name", ""),
                            "arguments": function.get("arguments", "{}"),
                        }
                    )
                continue

            if role == "tool":
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.get("tool_call_id") or "",
                        "output": str(content or ""),
                    }
                )
                continue

            input_items.append(
                {
                    "role": role,
                    "content": _build_content_parts(content),
                }
            )

        params: dict[str, Any] = {
            "model": model,
            "input": input_items,
            "store": False,
        }
        if tools:
            params["tools"] = tools
        if max_output_tokens:
            params["max_output_tokens"] = max_output_tokens
        if include:
            params["include"] = list(include)
        if stream:
            params["stream"] = True
        return params

    @staticmethod
    def parse_responses_response(response: Any) -> ChatResponse:
        """Parse an OpenAI Responses API object into the unified ChatResponse."""
        if isinstance(response, str):
            raise RuntimeError("Provider returned non-OpenAI responses payload (string body)")

        try:
            output_text = getattr(response, "output_text", None)
        except TypeError:
            output_text = None
        content = output_text or None
        tool_calls: list[dict] | None = None
        provider_items: list[dict[str, Any]] = []

        for item in getattr(response, "output", None) or []:
            item_type = getattr(item, "type", None)
            serialized_item = OpenAIAdapter.serialize_response_output_item(item)
            if serialized_item is not None:
                provider_items.append(serialized_item)
            if item_type == "message" and not content:
                for part in getattr(item, "content", []) or []:
                    if getattr(part, "type", None) in {"output_text", "text"}:
                        text = getattr(part, "text", None)
                        if text:
                            content = (content or "") + text
            elif item_type in {"function_call", "tool_call"}:
                if tool_calls is None:
                    tool_calls = []
                arguments = getattr(item, "arguments", None)
                tool_calls.append(
                    {
                        "id": getattr(item, "call_id", None) or getattr(item, "id", ""),
                        "type": "function",
                        "function": {
                            "name": getattr(item, "name", ""),
                            "arguments": arguments
                            if isinstance(arguments, str)
                            else json.dumps(arguments or {}),
                        },
                    }
                )

        usage = None
        if getattr(response, "usage", None):
            usage = {
                "prompt_tokens": getattr(response.usage, "input_tokens", 0),
                "completion_tokens": getattr(response.usage, "output_tokens", 0),
                "total_tokens": (
                    getattr(response.usage, "input_tokens", 0)
                    + getattr(response.usage, "output_tokens", 0)
                ),
            }

        return ChatResponse(
            content=content,
            provider_items=provider_items or None,
            tool_calls=tool_calls,
            finish_reason=getattr(response, "status", None),
            usage=usage,
            raw=response,
        )

    @staticmethod
    def summarize_responses_request_params(params: dict[str, Any]) -> dict[str, Any]:
        """Build a safe request summary for Responses API diagnostics."""
        input_items = params.get("input") or []
        return {
            "model": params.get("model"),
            "stream": params.get("stream", False),
            "store": params.get("store"),
            "include": params.get("include", []),
            "tool_count": len(params.get("tools") or []),
            "tool_names": [tool.get("name") for tool in (params.get("tools") or [])],
            "input_count": len(input_items),
            "input_types": [item.get("type", "message") for item in input_items[:8]],
            "input_roles": [item.get("role") for item in input_items[:8] if item.get("role")],
        }

    @staticmethod
    def summarize_responses_payload(
        response: Any,
        *,
        include_items: bool = False,
    ) -> dict[str, Any]:
        """Build a safe summary for Responses API payloads."""
        output_items = []
        for item in getattr(response, "output", None) or []:
            serialized = OpenAIAdapter.serialize_response_output_item(item)
            if serialized is not None:
                output_items.append(serialized)

        summary: dict[str, Any] = {
            "id": getattr(response, "id", None),
            "status": getattr(response, "status", None),
            "output_text_present": bool(getattr(response, "output_text", None)),
            "output_count": len(output_items),
            "output_types": [item.get("type") for item in output_items],
            "usage": (
                {
                    "input_tokens": getattr(getattr(response, "usage", None), "input_tokens", None),
                    "output_tokens": getattr(
                        getattr(response, "usage", None), "output_tokens", None
                    ),
                }
                if getattr(response, "usage", None)
                else None
            ),
        }
        if include_items:
            summary["output_items"] = output_items
        return summary

    @staticmethod
    def serialize_response_output_item(item: Any) -> dict[str, Any] | None:
        """Convert an OpenAI Responses output item into a JSON-safe dict."""
        if hasattr(item, "model_dump"):
            return item.model_dump(exclude_none=True)
        if isinstance(item, dict):
            return item

        item_type = getattr(item, "type", None)
        if item_type == "function_call":
            arguments = getattr(item, "arguments", None)
            return {
                "type": "function_call",
                "call_id": getattr(item, "call_id", None) or getattr(item, "id", ""),
                "name": getattr(item, "name", ""),
                "arguments": arguments
                if isinstance(arguments, str)
                else json.dumps(arguments or {}),
            }
        if item_type == "reasoning":
            serialized = {"type": "reasoning"}
            if summary := getattr(item, "summary", None):
                serialized["summary"] = summary
            if encrypted := getattr(item, "encrypted_content", None):
                serialized["encrypted_content"] = encrypted
            if item_id := getattr(item, "id", None):
                serialized["id"] = item_id
            return serialized
        if item_type == "message":
            content_parts = []
            for part in getattr(item, "content", []) or []:
                part_type = getattr(part, "type", None)
                if part_type in {"output_text", "text"}:
                    content_parts.append({"type": part_type, "text": getattr(part, "text", "")})
            return {
                "type": "message",
                "role": getattr(item, "role", "assistant"),
                "content": content_parts,
            }
        return None
