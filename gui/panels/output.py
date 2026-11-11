"""Output dock (stdout/stderr log)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyqtgraph as pg
import pyqtgraph.dockarea as dockarea
from pyqtgraph.Qt import QtGui, QtWidgets

if TYPE_CHECKING:
    from gui.main_window import SLMGUI


class OutputDock:
    """Creates the bottom Output dock and attaches it to ``gui.area``."""

    def __init__(self, gui: SLMGUI) -> None:
        self.gui = gui
        self._build()

    def _build(self) -> None:
        """Dock showing print() / stdout / stderr (see _install_gui_stdio)."""
        self.gui.d_output = dockarea.Dock("Output", size=(800, 160))
        self.gui.d_output.hideTitleBar()
        out_wrap = pg.LayoutWidget()
        self.gui.btn_clear_output_log = QtWidgets.QPushButton("Clear log")
        self.gui.output_log = QtWidgets.QPlainTextEdit()
        self.gui.output_log.setReadOnly(True)
        try:
            self.gui.output_log.setMaximumBlockCount(8000)
        except Exception:
            pass
        _mono = QtGui.QFont("Consolas")
        if not _mono.exactMatch():
            _mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.gui.output_log.setFont(_mono)
        self.gui.btn_clear_output_log.clicked.connect(self.gui.output_log.clear)
        self.gui.btn_copy_output_log = QtWidgets.QPushButton("Copy log")
        self.gui.btn_copy_output_log.setToolTip("Copy the whole log to the clipboard")
        self.gui.btn_copy_output_log.clicked.connect(self._copy_log)
        out_wrap.addWidget(self.gui.output_log, row=0, col=0, colspan=2)
        out_wrap.addWidget(self.gui.btn_clear_output_log, row=1, col=1)
        out_wrap.addWidget(self.gui.btn_copy_output_log, row=1, col=0)
        out_wrap.layout.setColumnStretch(0, 1)  # Copy log | Clear log: equal widths
        out_wrap.layout.setColumnStretch(1, 1)
        self.gui.d_output.addWidget(out_wrap)
        self.gui.area.addDock(self.gui.d_output, "bottom")

    def _copy_log(self) -> None:
        """Copy log: put the whole log text on the clipboard."""
        QtWidgets.QApplication.clipboard().setText(self.gui.output_log.toPlainText())
