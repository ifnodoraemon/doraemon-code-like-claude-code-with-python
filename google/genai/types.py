"""Minimal stand-ins for ``google.genai.types`` used by the test suite.

These classes intentionally implement only the small surface area exercised by
this repository so tests can run in environments where ``google-genai`` is not
installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class FunctionDeclaration:
    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Tool:
    function_declarations: list[FunctionDeclaration] = field(default_factory=list)


@dataclass(slots=True)
class AutomaticFunctionCallingConfig:
    disable: bool = False


@dataclass(slots=True)
class GenerateContentConfig:
    temperature: float | None = None
    candidate_count: int | None = None
    max_output_tokens: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    tools: list[Tool] | None = None
    automatic_function_calling: AutomaticFunctionCallingConfig | None = None
    system_instruction: str | None = None

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


@dataclass(slots=True)
class FunctionResponse:
    name: str
    response: dict[str, Any]


@dataclass(slots=True)
class Part:
    text: str | None = None
    inline_data: bytes | None = None
    mime_type: str | None = None
    function_response: FunctionResponse | None = None

    @classmethod
    def from_bytes(cls, data: bytes, mime_type: str) -> Part:
        return cls(inline_data=data, mime_type=mime_type)

    @classmethod
    def from_function_response(cls, name: str, response: dict[str, Any]) -> Part:
        return cls(function_response=FunctionResponse(name=name, response=response))


@dataclass(slots=True)
class Content:
    parts: list[Part] = field(default_factory=list)
    role: str | None = None
