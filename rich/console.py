from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


class Console:
    def print(self, *args: Any, **kwargs: Any) -> None:
        end = kwargs.get("end", "\n")
        text = " ".join(str(arg) for arg in args)
        if text:
            print(text, end=end)
        elif end:
            print(end=end)

    @contextmanager
    def status(self, *_args: Any, **_kwargs: Any) -> Iterator[Console]:
        yield self
