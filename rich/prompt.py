from __future__ import annotations

from typing import Any


class Prompt:
    @staticmethod
    def ask(_prompt: str, **kwargs: Any) -> str:
        default = kwargs.get("default")
        if default is not None:
            return str(default)
        choices = kwargs.get("choices")
        if choices:
            return str(choices[0])
        return ""


class Confirm:
    @staticmethod
    def ask(_prompt: str, **kwargs: Any) -> bool:
        default = kwargs.get("default")
        return bool(default) if default is not None else False
