"""Output dock (stdout/stderr log)."""
import pyqtgraph as pg
import pyqtgraph.dockarea as dockarea
from pyqtgraph.Qt import QtGui, QtWidgets


class OutputDock:
    """Creates the bottom Output dock and attaches it to ``gui.area``."""

    def __init__(self, gui):
        self.gui = gui
        self._build()

    def _build(self):
        """Dock showing print() / stdout / stderr (see _install_gui_stdio)."""
        self.gui.d_output = dockarea.Dock("Output", size=(800, 160))
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
        out_wrap.addWidget(self.gui.btn_clear_output_log, row=0, col=0)
        out_wrap.addWidget(self.gui.output_log, row=1, col=0)
        self.gui.d_output.addWidget(out_wrap)
        self.gui.area.addDock(self.gui.d_output, "bottom")

