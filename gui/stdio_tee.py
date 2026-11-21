"""Stdout/stderr tee into a QPlainTextEdit."""

from __future__ import annotations

from typing import TextIO

from PyQt5.QtCore import pyqtSignal
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from gui.log_timestamp import LineTimestamper


class StreamToPlainText(QtCore.QObject):
    """Mirror writes to a QPlainTextEdit while keeping the original stream (e.g. terminal), each line prefixed with
    date and time (YYYY.MM.DD HH:MM:SS). Thread-safe."""

    _append = pyqtSignal(str)

    def __init__(
        self, original_stream: TextIO, plain_widget: QtWidgets.QPlainTextEdit, parent: QtCore.QObject | None = None
    ) -> None:
        super().__init__(parent)
        self._orig = original_stream
        self._plain = plain_widget
        self._append.connect(self._append_slot)
        self.encoding = getattr(original_stream, "encoding", None) or "utf-8"
        self._timestamper = LineTimestamper()

    def _append_slot(self, text: str) -> None:
        self._plain.moveCursor(QtGui.QTextCursor.End)
        self._plain.insertPlainText(text)
        sb = self._plain.verticalScrollBar()
        sb.setValue(sb.maximum())

    def write(self, s: object) -> int:
        if not isinstance(s, str):
            s = str(s)
        n_written = len(s)
        s = self._timestamper.stamp(s)
        try:
            self._orig.write(s)
        except Exception:
            pass
        try:
            self._orig.flush()
        except Exception:
            pass
        if s:
            self._append.emit(s)
        return n_written

    def flush(self) -> None:
        try:
            self._orig.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        return False
