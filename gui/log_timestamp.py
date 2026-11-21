"""Date and time at the start of every log line (no Qt)."""

from __future__ import annotations

import threading
import time
from typing import Callable

LOG_TIMESTAMP_FORMAT: str = "%Y.%m.%d %H:%M:%S"


class LineTimestamper:
    """Adds 'YYYY.MM.DD HH:MM:SS ' before every line of text written in pieces (print writes text and newline separately)."""

    def __init__(self, now: Callable[[], str] | None = None) -> None:
        self._now = now or (lambda: time.strftime(LOG_TIMESTAMP_FORMAT))
        self._at_line_start = True
        self._lock = threading.Lock()

    def stamp(self, text: str) -> str:
        with self._lock:
            pieces = []
            for line in text.splitlines(keepends=True):
                if self._at_line_start:
                    pieces.append(self._now() + " ")
                pieces.append(line)
                self._at_line_start = line.endswith("\n")
            return "".join(pieces)
