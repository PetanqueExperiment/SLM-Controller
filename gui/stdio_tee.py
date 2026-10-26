"""Stdout/stderr tee into a QPlainTextEdit."""
from PyQt5.QtCore import pyqtSignal
from pyqtgraph.Qt import QtCore, QtGui


class StreamToPlainText(QtCore.QObject):
    """Mirror writes to a QPlainTextEdit while keeping the original stream (e.g. terminal). Thread-safe."""

    _append = pyqtSignal(str)

    def __init__(self, original_stream, plain_widget, parent=None):
        super().__init__(parent)
        self._orig = original_stream
        self._plain = plain_widget
        self._append.connect(self._append_slot)
        self.encoding = getattr(original_stream, "encoding", None) or "utf-8"

    def _append_slot(self, text):
        self._plain.moveCursor(QtGui.QTextCursor.End)
        self._plain.insertPlainText(text)
        sb = self._plain.verticalScrollBar()
        sb.setValue(sb.maximum())

    def write(self, s):
        if not isinstance(s, str):
            s = str(s)
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
        return len(s)

    def flush(self):
        try:
            self._orig.flush()
        except Exception:
            pass

    def isatty(self):
        return False
