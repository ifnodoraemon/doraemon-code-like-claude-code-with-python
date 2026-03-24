from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Syntax:
    code: str
    lexer: str
    theme: str | None = None
    line_numbers: bool = False

    def __str__(self) -> str:
        return self.code
