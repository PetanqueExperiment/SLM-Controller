"""Static mode page: the hologram file buttons and the displayed hologram (the mode itself is in logic.py)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

if TYPE_CHECKING:
    from gui.main_window import SLMGUI


class StaticModePage(pg.LayoutWidget):
    """Static mode controls: display a blank or a hologram file, and the name of the displayed hologram."""

    def __init__(self, gui: SLMGUI) -> None:
        super().__init__()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.displayLabel = QtWidgets.QLabel('Display hologram:')
        self.blankBtn = QtWidgets.QPushButton('Blank')
        self.loadBtn = QtWidgets.QPushButton('From file...')
        self.displayedHologramLabel = QtWidgets.QLabel()
        self.displayedHologramLabel.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.addWidget(self.displayLabel, row=0, col=0, colspan=3)
        self.addWidget(self.blankBtn, row=1, col=0)
        self.addWidget(self.loadBtn, row=1, col=1)
        self.addWidget(self.displayedHologramLabel, row=2, col=0, colspan=3)
        self.layout.setColumnStretch(2, 1)
        self.layout.setRowStretch(3, 1)
        gui._arbitrary.addDisplayedHologramLabel(self.displayedHologramLabel)
        self.blankBtn.clicked.connect(gui._arbitrary.generateBlank)
        self.loadBtn.clicked.connect(gui._arbitrary.loadPattern_clicked)
