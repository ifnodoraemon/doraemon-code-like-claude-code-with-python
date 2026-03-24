from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Panel:
    renderable: Any
    title: str | None = None
    border_style: str | None = None
    expand: bool = False

    def __str__(self) -> str:
        return str(self.renderable)
