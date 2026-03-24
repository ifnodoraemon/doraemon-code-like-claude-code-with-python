from __future__ import annotations

from typing import Any


class Table:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.columns: list[tuple[Any, ...]] = []
        self.rows: list[tuple[Any, ...]] = []

    def add_column(self, *args: Any, **kwargs: Any) -> None:
        self.columns.append(args)

    def add_row(self, *args: Any, **kwargs: Any) -> None:
        self.rows.append(tuple(args))
