from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Markdown:
    text: str

    def __str__(self) -> str:
        return self.text
