"""The Operation mode panel, shared by all modes: mode selector, status, and the page of the selected mode.

The panel is given the list of modes (logic.OPERATION_MODES); each mode builds its own page of controls
(static.py / dynamic.py) and may show a status next to the selector.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from pyqtgraph.Qt import QtWidgets

from gui.panels.modes.logic import OperationMode

if TYPE_CHECKING:
    from gui.main_window import SLMGUI


class OperationModePanel(QtWidgets.QWidget):
    """'Operation mode' selector and the page of the selected mode. The active mode object is ``gui.operation_mode``."""

    def __init__(self, gui: SLMGUI, modes: Sequence[type[OperationMode]], default: type[OperationMode]) -> None:
        super().__init__()
        self.gui = gui
        self._mode_classes: tuple[type[OperationMode], ...] = tuple(modes)
        self._modes: dict[type[OperationMode], OperationMode] = {cls: cls() for cls in self._mode_classes}

        self.operationModeLabel = QtWidgets.QLabel('Operation mode')
        self.operationModeCB = QtWidgets.QComboBox()
        for mode_cls in self._mode_classes:
            self.operationModeCB.addItem(mode_cls.name, mode_cls)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.operationModeLabel)
        top.addWidget(self.operationModeCB)
        self.dynamicStatusSeparator = QtWidgets.QLabel('|')
        self.dynamicStatusLabel = QtWidgets.QLabel()
        top.addWidget(self.dynamicStatusSeparator)
        top.addWidget(self.dynamicStatusLabel)
        top.addStretch(1)

        self.pages = QtWidgets.QStackedWidget()
        self._pages: dict[type[OperationMode], QtWidgets.QWidget] = {cls: self._modes[cls].buildPage(gui) or QtWidgets.QWidget() for cls in self._mode_classes}
        for mode_cls in self._mode_classes:
            self.pages.addWidget(self._pages[mode_cls])

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(top)
        lay.addWidget(self.pages, 1)

        self.gui.operation_mode = self._modes[default]
        self.gui.operation_mode.activate()
        self.operationModeCB.setCurrentIndex(self._mode_classes.index(default))
        self._showPage(default)
        self.refreshStatus()
        self.operationModeCB.currentIndexChanged.connect(self._modeSelected)

    def refreshStatus(self) -> None:
        """Show the active mode's status text (e.g. Ready / Editing for dynamic) after '|', or hide both."""
        text = self.gui.operation_mode.statusText(self.gui)
        self.dynamicStatusLabel.setText(text or '')
        self.dynamicStatusSeparator.setVisible(text is not None)
        self.dynamicStatusLabel.setVisible(text is not None)

    def _modeSelected(self, index: int) -> None:
        mode_cls = self.operationModeCB.itemData(index)
        if mode_cls is None or isinstance(self.gui.operation_mode, mode_cls):
            return
        self.setOperationMode(mode_cls)

    def setOperationMode(self, mode_cls: type[OperationMode]) -> None:
        """Deactivate the current mode, activate ``mode_cls`` and show its controls and name in the selector."""
        self.gui.operation_mode.deactivate()
        mode = self._modes[mode_cls]
        self.gui.operation_mode = mode
        mode.activate()
        index = self._mode_classes.index(mode_cls)
        if index != self.operationModeCB.currentIndex():
            self.operationModeCB.blockSignals(True)
            self.operationModeCB.setCurrentIndex(index)
            self.operationModeCB.blockSignals(False)
        self._showPage(mode_cls)
        self.refreshStatus()
        print('Operation mode:', mode.name)

    def _showPage(self, mode_cls: type[OperationMode]) -> None:
        """Show the page of ``mode_cls``; hidden pages do not count for the panel height."""
        current = self._pages[mode_cls]
        for page in self._pages.values():
            vertical = QtWidgets.QSizePolicy.Preferred if page is current else QtWidgets.QSizePolicy.Ignored
            page.setSizePolicy(QtWidgets.QSizePolicy.Preferred, vertical)
        self.pages.setCurrentWidget(current)
        self.pages.adjustSize()
        self.updateGeometry()
