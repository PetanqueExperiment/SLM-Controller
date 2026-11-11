"""Static mode page: the hologram file buttons and the displayed hologram (the mode itself is in logic.py)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

if TYPE_CHECKING:
    from gui.main_window import SLMGUI


class StaticModePage(pg.LayoutWidget):
    """Static mode controls: the hologram file buttons (same actions as in the Hologram Files panel)."""

    def __init__(self, gui: SLMGUI) -> None:
        super().__init__()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.blankBtn = QtWidgets.QPushButton('Blank')
        self.loadBtn = QtWidgets.QPushButton('Load Hologram')
        self.saveBtn = QtWidgets.QPushButton('Save Hologram')
        self.displayedHologramLabel = QtWidgets.QLabel()
        self.displayedHologramLabel.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.addWidget(self.blankBtn, row=0, col=0)
        self.addWidget(self.loadBtn, row=0, col=1)
        self.addWidget(self.saveBtn, row=0, col=2)
        self.addWidget(self.displayedHologramLabel, row=1, col=0, colspan=4)
        self.layout.setColumnStretch(3, 1)
        self.layout.setRowStretch(2, 1)
        gui._arbitrary.addDisplayedHologramLabel(self.displayedHologramLabel)
        self.blankBtn.clicked.connect(gui._arbitrary.generateBlank)
        self.loadBtn.clicked.connect(gui._arbitrary.loadPattern_clicked)
        self.saveBtn.clicked.connect(gui._arbitrary.savePattern_clicked)
