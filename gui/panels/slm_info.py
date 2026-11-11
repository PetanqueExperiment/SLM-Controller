"""SLM info strip (top of the window): board number, serial number and connection state of the SLM."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyqtgraph.dockarea as dockarea
from pyqtgraph.Qt import QtCore, QtWidgets

from gui.settings import SIMULATED_BOARD, findDevice

if TYPE_CHECKING:
    from gui.main_window import SLMGUI


class SlmInfoDock:
    """Creates the top SLM info dock (no title bar) and attaches it to ``gui.area``. Call :meth:`update` on changes."""

    def __init__(self, gui: SLMGUI) -> None:
        self.gui = gui
        self._build()

    def _build(self) -> None:
        self.gui.d_slm_info = dockarea.Dock("SLM", size=(800, 20))
        self.gui.d_slm_info.hideTitleBar()
        self.label = QtWidgets.QLabel()
        self.label.setContentsMargins(6, 0, 6, 0)
        self.label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.gui.d_slm_info.addWidget(self.label)
        self.gui.d_slm_info.layout.setContentsMargins(0, 0, 0, 0)
        self.gui.area.addDock(self.gui.d_slm_info, "top")
        self.update()

    def update(self) -> None:
        """Refresh from ``gui.SLM`` / ``gui.board``; the serial comes from the SLM when connected, else from devices.json."""
        slm = getattr(self.gui, "SLM", None)
        board = getattr(self.gui, "board", None)
        if slm is not None and slm.simulate:
            serial, state = "—", "Simulated"
        elif slm is not None:
            serial = "not readable" if slm.serial is None else str(slm.serial)
            state = "Connected"
        else:
            device = None if board is None else findDevice(board)
            serial = "—" if device is None or device.serial is None else f"{device.serial} (from device list)"
            state = "Not connected"
        device = "Board: —" if board is None else "Simulated SLM" if board == SIMULATED_BOARD else f"Board: {board}"
        self.label.setText(f"{device}  |  Serial number: {serial}  |  {state}")
