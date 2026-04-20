"""
Anthropic Provider Adapter

Converts messages and builds request parameters for the Anthropic
Messages API.
"""

import json
import logging
from collections.abc import Sequence
from typing import Any

from src.core.llm.model_utils import ToolDefinition, get_content_text

logger = logging.getLogger(__name__)


def build_anthropic_content_parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        parts = []
        for part in content:
            if part.get("type") == "text":
                parts.append({"type": "text", "text": part["text"]})
            elif part.get("type") == "image":
                source = part["source"]
                parts.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": source["media_type"],
                            "data": source["data"],
                        },
                    }
                )
        return parts
    return [{"type": "text", "text": str(content or "")}]


class AnthropicAdapter:
    """Converts messages and configs for Anthropic Claude API."""

    @staticmethod
    def convert_messages(
        messages: Sequence[Any],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> tuple[str | None, list[dict]]:
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
                    content.extend(build_anthropic_content_parts(msg["content"]))
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        func = tc.get("function", {})
                        args_str = func.get("arguments", "{}")
                        try:
                            args = json.loads(args_str) if isinstance(args_str, str) else args_str
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse tool arguments: %s", args_str)
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
        msg_list: Any,
        tools: Sequence[ToolDefinition | dict] | None = None,
        system: str | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict:
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
