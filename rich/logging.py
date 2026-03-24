from __future__ import annotations

import logging
from typing import Any


class RichHandler(logging.StreamHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.kwargs = kwargs
