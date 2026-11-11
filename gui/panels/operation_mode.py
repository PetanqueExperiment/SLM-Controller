"""Operation mode panel (top of the right column): mode selector and the controls of the selected mode.

The modes themselves (no Qt) are in gui/operation_modes.py; each mode has one page of controls here.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from gui.constants import SLM_PATTERNS_DIR
from gui.hologram_handler import readLegacyPattern, readTrapsDict
from gui.panels.operation_mode import DEFAULT_OPERATION_MODE, OPERATION_MODES, SLOT_START_STOP, SLOT_TIMINGS, DynamicMode, HologramSlot, OperationMode, StaticMode, ordinal

if TYPE_CHECKING:
    from gui.main_window import SLMGUI

NO_HOLOGRAM_TEXT: str = 'none'
NUMBER_PADDING_PX: int = 16
DYNAMIC_READY_TEXT: str = 'Ready'
DYNAMIC_EDITING_TEXT: str = 'Editing'


def readHologramFile(path: str) -> dict[str, Any]:
    """Contents of a hologram file as a dict with at least 'phasePattern' (.pkl traps_dict, or old .npy / .bmp)."""
    low = path.lower()
    if low.endswith('.pkl'):
        data = readTrapsDict(path)
        if not isinstance(data, dict) or 'phasePattern' not in data:
            raise ValueError('not a hologram file (no phasePattern)')
        return data
    if low.endswith('.npy') or low.endswith('.bmp'):
        return {'phasePattern': readLegacyPattern(path)}
    raise ValueError('unsupported file type (use .pkl, .npy or .bmp)')


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
    """One entry of the dynamic list: number, timing selector, Load and file name; for start-stop a stop line below."""

    def __init__(self, mode: DynamicMode, index: int, number_width: int) -> None:
        super().__init__()
        self.mode = mode
        self.index = index
        self.titleLabel = QtWidgets.QLabel(str(index + 1))
        self.loadBtn = QtWidgets.QPushButton('Load')
        self.nameLabel = QtWidgets.QLabel(NO_HOLOGRAM_TEXT)
        self.timingCB = QtWidgets.QComboBox()
        self.timingCB.addItems(list(SLOT_TIMINGS))
        self.stopLoadBtn = QtWidgets.QPushButton('Load')
        self.stopLoadBtn.setToolTip('Stop hologram')
        self.stopNameLabel = QtWidgets.QLabel(NO_HOLOGRAM_TEXT)
        for label in (self.nameLabel, self.stopNameLabel):
            label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.titleLabel.setMinimumWidth(number_width)

        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setColumnMinimumWidth(1, NUMBER_PADDING_PX)
        grid.addWidget(self.titleLabel, 0, 0)
        grid.addWidget(self.timingCB, 0, 2)
        grid.addWidget(self.loadBtn, 0, 3)
        grid.addWidget(self.nameLabel, 0, 4)
        grid.addWidget(self.stopLoadBtn, 1, 3)
        grid.addWidget(self.stopNameLabel, 1, 4)
        grid.setColumnStretch(4, 1)

        self.timingCB.setCurrentText(self.slot.timing)
        self.loadBtn.clicked.connect(self.loadHologram_clicked)
        self.stopLoadBtn.clicked.connect(self.loadStopHologram_clicked)
        self.timingCB.currentTextChanged.connect(self._timingSelected)
        self.refresh()

    @property
    def slot(self) -> HologramSlot:
        return self.mode.holograms[self.index]

    def refresh(self) -> None:
        """Show the slot's file names, and the stop hologram line only for start-stop."""
        slot = self.slot
        for label, path in ((self.nameLabel, slot.path), (self.stopNameLabel, slot.stop_path)):
            label.setText(NO_HOLOGRAM_TEXT if path is None else os.path.basename(path))
            label.setToolTip('' if path is None else os.path.abspath(path))
        start_stop = slot.timing == SLOT_START_STOP
        for widget in (self.stopLoadBtn, self.stopNameLabel):
            widget.setVisible(start_stop)

    def _timingSelected(self, timing: str) -> None:
        self.mode.setTiming(self.index, timing)
        self.refresh()

    def loadHologram_clicked(self) -> None:
        """Choose the hologram of this entry (stored in the dynamic mode; nothing is sent to the SLM)."""
        chosen = chooseHologramFile(self, 'Load %s hologram' % ordinal(self.index + 1), self.slot.path)
        if chosen is None:
            return
        self.mode.setHologram(self.index, *chosen)
        self.refresh()
        print('%s hologram: %s' % (ordinal(self.index + 1), chosen[0]))

    def loadStopHologram_clicked(self) -> None:
        """Choose the stop hologram of this entry (stored in the dynamic mode; nothing is sent to the SLM)."""
        chosen = chooseHologramFile(self, 'Load stop hologram of the %s hologram' % ordinal(self.index + 1), self.slot.stop_path or self.slot.path)
        if chosen is None:
            return
        self.mode.setStopHologram(self.index, *chosen)
        self.refresh()
        print('%s hologram, stop hologram: %s' % (ordinal(self.index + 1), chosen[0]))


class DynamicModePage(QtWidgets.QWidget):
    """Dynamic mode controls: Add / Remove buttons and the scrollable hologram list (one HologramSlotBlock per entry)."""

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
        lay.addLayout(buttons)
        lay.addWidget(self.listScroll, 1)

        self._scroll_to_end = False
        self.listScroll.verticalScrollBar().rangeChanged.connect(self._listRangeChanged)
        self.addSlotBtn.clicked.connect(self.addSlot_clicked)
        self.removeSlotBtn.clicked.connect(self.removeSlot_clicked)
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


class OperationModePanel(QtWidgets.QWidget):
    """'Operation mode' selector and the page of the selected mode. The active mode object is ``gui.operation_mode``."""

    def __init__(self, gui: SLMGUI) -> None:
        super().__init__()
        self.gui = gui
        self._modes: dict[type[OperationMode], OperationMode] = {cls: cls() for cls in OPERATION_MODES}

        self.operationModeLabel = QtWidgets.QLabel('Operation mode')
        self.operationModeCB = QtWidgets.QComboBox()
        for mode_cls in OPERATION_MODES:
            self.operationModeCB.addItem(mode_cls.name, mode_cls)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.operationModeLabel)
        top.addWidget(self.operationModeCB)
        self.dynamicStatusLabel = QtWidgets.QLabel()
        top.addWidget(self.dynamicStatusLabel)
        top.addStretch(1)

        self.pages = QtWidgets.QStackedWidget()
        self._pages: dict[type[OperationMode], QtWidgets.QWidget] = {
            StaticMode: StaticModePage(gui),
            DynamicMode: DynamicModePage(self._modes[DynamicMode]),
        }
        for mode_cls in OPERATION_MODES:
            self.pages.addWidget(self._pages[mode_cls])

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(top)
        lay.addWidget(self.pages, 1)

        self.gui.operation_mode = self._modes[DEFAULT_OPERATION_MODE]
        self.gui.operation_mode.activate()
        self.operationModeCB.setCurrentIndex(OPERATION_MODES.index(DEFAULT_OPERATION_MODE))
        self._showPage(DEFAULT_OPERATION_MODE)
        self.refreshDynamicStatus()
        self.operationModeCB.currentIndexChanged.connect(self._modeSelected)

    def refreshDynamicStatus(self) -> None:
        """Show ``gui.dynamicIsReady`` as Ready / Editing next to the selector, only while the dynamic mode is selected."""
        self.dynamicStatusLabel.setText('| %s' % (DYNAMIC_READY_TEXT if self.gui.dynamicIsReady else DYNAMIC_EDITING_TEXT))
        self.dynamicStatusLabel.setVisible(isinstance(self.gui.operation_mode, DynamicMode))

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
        index = OPERATION_MODES.index(mode_cls)
        if index != self.operationModeCB.currentIndex():
            self.operationModeCB.blockSignals(True)
            self.operationModeCB.setCurrentIndex(index)
            self.operationModeCB.blockSignals(False)
        self._showPage(mode_cls)
        self.refreshDynamicStatus()
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
