"""Minimal local fallback for ``google.genai`` in offline test environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from . import types


class _ModelsAPI:
    def generate_content(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(
            "google-genai is not installed; patch Client in tests or install the dependency"
        )

    async def generate_content_stream(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(
            "google-genai is not installed; patch Client in tests or install the dependency"
        )


@dataclass(slots=True)
class Client:
    api_key: str | None = None
    models: Any = field(default_factory=_ModelsAPI)
    aio: Any = field(default_factory=lambda: SimpleNamespace(models=_ModelsAPI()))


__all__ = ["Client", "types"]
