"""Dynamic mode page: the hologram list, one frame per line, each assigned by hand from a file.

The list itself is in logic.py (DynamicMode). Driving the SLM with it is not implemented yet; loading files only
stores them there.
"""

from __future__ import annotations

import os
from typing import Any

from pyqtgraph.Qt import QtCore, QtWidgets

from gui.constants import SLM_PATTERNS_DIR
from gui.panels.modes.logic import DynamicMode, HologramSlot, ordinal, readHologramFile

NO_HOLOGRAM_TEXT: str = 'none'
NUMBER_PADDING_PX: int = 16
LIST_WIDTH_FRACTION: float = 0.5


def chooseHologramFile(parent: QtWidgets.QWidget, title: str, start: str | None) -> tuple[str, dict[str, Any]] | None:
    """Ask for a hologram file and read it; None if cancelled or unreadable (the error is printed)."""
    path, _ = QtWidgets.QFileDialog.getOpenFileName(parent, title, start or SLM_PATTERNS_DIR, 'Holograms (*.pkl *.npy *.bmp);;All files (*.*)')
    if not path:
        return None
    try:
        return path, readHologramFile(path)
    except Exception as e:
        print('Could not load hologram %s: %s' % (path, e))
        return None


class HologramSlotBlock(QtWidgets.QWidget):
    """One frame of the dynamic list on one line: number, Load and the loaded file name."""

    def __init__(self, mode: DynamicMode, index: int, number_width: int) -> None:
        super().__init__()
        self.mode = mode
        self.index = index
        self.titleLabel = QtWidgets.QLabel(str(index + 1))
        self.loadBtn = QtWidgets.QPushButton('Load')
        self.nameLabel = QtWidgets.QLabel(NO_HOLOGRAM_TEXT)
        self.nameLabel.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.titleLabel.setMinimumWidth(number_width)

        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setColumnMinimumWidth(1, NUMBER_PADDING_PX)
        grid.addWidget(self.titleLabel, 0, 0)
        grid.addWidget(self.loadBtn, 0, 2)
        grid.addWidget(self.nameLabel, 0, 3)
        grid.setColumnStretch(3, 1)

        self.loadBtn.clicked.connect(self.loadHologram_clicked)
        self.refresh()

    @property
    def slot(self) -> HologramSlot:
        return self.mode.holograms[self.index]

    def refresh(self) -> None:
        """Show the name of the frame's file."""
        path = self.slot.path
        self.nameLabel.setText(NO_HOLOGRAM_TEXT if path is None else os.path.basename(path))
        self.nameLabel.setToolTip('' if path is None else os.path.abspath(path))

    def loadHologram_clicked(self) -> None:
        """Choose the hologram file of this frame (stored in the dynamic mode; nothing is sent to the SLM)."""
        chosen = chooseHologramFile(self, 'Load %s hologram' % ordinal(self.index + 1), self.slot.path)
        if chosen is None:
            return
        self.mode.setHologram(self.index, *chosen)
        self.refresh()
        print('%s hologram: %s' % (ordinal(self.index + 1), chosen[0]))


class DynamicModePage(QtWidgets.QWidget):
    """Dynamic mode controls: Add / Remove / Interpolate buttons and the scrollable hologram list."""

    def __init__(self, mode: DynamicMode) -> None:
        super().__init__()
        self.mode = mode
        self.blocks: list[HologramSlotBlock] = []
        metrics = self.fontMetrics()
        self._number_width = metrics.horizontalAdvance('000')

        self.addSlotBtn = QtWidgets.QPushButton('Add hologram')
        self.removeSlotBtn = QtWidgets.QPushButton('Remove last')
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.addSlotBtn)
        buttons.addWidget(self.removeSlotBtn)
        buttons.addWidget(QtWidgets.QLabel('|'))
        self.interpolateBtn = QtWidgets.QPushButton('Interpolate')
        self.interpolateBtn.setToolTip('Not implemented yet')
        buttons.addWidget(self.interpolateBtn)
        buttons.addStretch(1)
        list_widget = QtWidgets.QWidget()
        list_lay = QtWidgets.QVBoxLayout(list_widget)
        list_lay.setContentsMargins(4, 4, 4, 4)
        self.listLayout = QtWidgets.QVBoxLayout()
        self.listLayout.setContentsMargins(0, 0, 0, 0)
        list_lay.addLayout(self.listLayout)
        list_lay.addStretch(1)
        self.listScroll = QtWidgets.QScrollArea()
        self.listScroll.setWidgetResizable(True)
        self.listScroll.setWidget(list_widget)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        list_row = QtWidgets.QHBoxLayout()
        list_row.addWidget(self.listScroll, round(100 * LIST_WIDTH_FRACTION))
        list_row.addStretch(round(100 * (1.0 - LIST_WIDTH_FRACTION)))
        lay.addLayout(buttons)
        lay.addLayout(list_row, 1)

        self._scroll_to_end = False
        self.listScroll.verticalScrollBar().rangeChanged.connect(self._listRangeChanged)
        self.addSlotBtn.clicked.connect(self.addSlot_clicked)
        self.removeSlotBtn.clicked.connect(self.removeSlot_clicked)
        self.interpolateBtn.clicked.connect(self.interpolate_clicked)
        for index in range(len(self.mode.holograms)):
            self._addBlock(index)
        self._updateRemoveButton()

    def _addBlock(self, index: int) -> None:
        block = HologramSlotBlock(self.mode, index, self._number_width)
        self.listLayout.addWidget(block)
        self.blocks.append(block)

    def _listRangeChanged(self, _minimum: int, maximum: int) -> None:
        if self._scroll_to_end:
            self._scroll_to_end = False
            self.listScroll.verticalScrollBar().setValue(maximum)

    def interpolate_clicked(self) -> None:
        """Placeholder: interpolation between holograms is not implemented yet."""
        print('Interpolate: not implemented yet')

    def _clearScrollToEnd(self) -> None:
        self._scroll_to_end = False

    def _updateRemoveButton(self) -> None:
        self.removeSlotBtn.setEnabled(len(self.mode.holograms) > 1)

    def addSlot_clicked(self) -> None:
        self._scroll_to_end = True
        self._addBlock(self.mode.addSlot())
        QtCore.QTimer.singleShot(200, self._clearScrollToEnd)
        self._updateRemoveButton()

    def removeSlot_clicked(self) -> None:
        if not self.mode.removeLastSlot():
            return
        block = self.blocks.pop()
        self.listLayout.removeWidget(block)
        block.deleteLater()
        self._updateRemoveButton()
