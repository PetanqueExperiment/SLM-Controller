"""Info strip (top of the window): active profile (left, 20%) and the SLM's board, serial number and state (right, 80%)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyqtgraph.dockarea as dockarea
from pyqtgraph.Qt import QtCore, QtWidgets

from gui.settings import SIMULATED_BOARD, findDevice

if TYPE_CHECKING:
    from gui.main_window import SLMGUI

PROFILE_WIDTH_STRETCH: int = 1  # profile : SLM width ratio = 1 : 4 → 20 % : 80 %
SLM_WIDTH_STRETCH: int = 4


def _infoPanel() -> tuple[QtWidgets.QFrame, QtWidgets.QLabel]:
    """A bordered panel (same look as the other GUI panels) holding one line of selectable text."""
    frame = QtWidgets.QFrame()
    frame.setObjectName("info_panel_frame")
    frame.setStyleSheet("#info_panel_frame { border: 1px solid #8a8a8a; border-radius: 4px; background: transparent; }")
    label = QtWidgets.QLabel()
    label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
    lay = QtWidgets.QHBoxLayout(frame)
    lay.setContentsMargins(6, 0, 6, 0)
    lay.addWidget(label)
    return frame, label


class SlmInfoDock:
    """Creates the top info dock (no title bar) and attaches it to ``gui.area``. Call :meth:`update` on changes."""

    def __init__(self, gui: SLMGUI, gap_px: int) -> None:
        self.gui = gui
        self._build(gap_px)

    def _build(self, gap_px: int) -> None:
        self.gui.d_slm_info = dockarea.Dock("SLM", size=(800, 20))
        self.gui.d_slm_info.nStyle = "Dock > QWidget { border: none; }"  # only the two panels draw a border, not the dock
        self.gui.d_slm_info.hideTitleBar()
        row = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(gap_px)
        profile_frame, self.profile_label = _infoPanel()
        slm_frame, self.label = _infoPanel()
        lay.addWidget(profile_frame, PROFILE_WIDTH_STRETCH)
        lay.addWidget(slm_frame, SLM_WIDTH_STRETCH)
        self.gui.d_slm_info.addWidget(row)
        self.gui.d_slm_info.layout.setContentsMargins(0, 0, 0, 0)
        self.gui.area.addDock(self.gui.d_slm_info, "top")
        self.update()

    def update(self) -> None:
        """Refresh the profile name and the SLM info (serial from the SLM when connected, else from devices.json)."""
        self.profile_label.setText(f"Profile: {getattr(self.gui, 'current_profile', '—')}")
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
