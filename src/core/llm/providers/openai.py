"""
OpenAI Provider Adapter

Converts messages and builds request parameters for the OpenAI API
(Chat Completions and Responses endpoints).
"""

from collections.abc import Sequence
from typing import Any

from src.core.llm.model_utils import ToolDefinition

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
