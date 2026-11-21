"""Device ▸ Set device parameters…: the device list entry of one real SLM, in the order of devices.json.

Name and Zernike correction radius are edited here and saved with Save. WFC and LUT are changed with their own
Load… buttons (file choice, copy to devices/, loading into the controller), which apply at once. Board, serial
number, image size and pixel pitch are shown only: the SLM reports them at connect.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pyqtgraph.Qt import QtWidgets

from gui.settings import MAX_SLM_RADIUS_M, Device, deviceLabel, findDevice, setDeviceName

if TYPE_CHECKING:
    from gui.main_window import SLMGUI

NOT_KNOWN_TEXT: str = 'not known yet (read from the SLM at connect)'
NONE_SET_TEXT: str = 'none set'


class DeviceParametersDialog(QtWidgets.QDialog):
    """Parameters of the device on ``board`` (must be in the device list)."""

    def __init__(self, gui: SLMGUI, board: int) -> None:
        super().__init__(gui)
        self.gui = gui
        self.board = board
        self.setWindowTitle('Device parameters')

        self.nameEdit = QtWidgets.QLineEdit()
        self.boardLabel = QtWidgets.QLabel()
        self.serialLabel = QtWidgets.QLabel()
        self.wfcLabel = QtWidgets.QLabel()
        self.wfcBtn = QtWidgets.QPushButton('Load…')
        self.lutLabel = QtWidgets.QLabel()
        self.lutBtn = QtWidgets.QPushButton('Load…')
        self.widthLabel = QtWidgets.QLabel()
        self.heightLabel = QtWidgets.QLabel()
        self.pitchLabel = QtWidgets.QLabel()
        self.radiusSpin = QtWidgets.QDoubleSpinBox()
        self.radiusSpin.setRange(0.001, MAX_SLM_RADIUS_M * 1e3)
        self.radiusSpin.setDecimals(4)
        self.radiusSpin.setSuffix(' mm')
        self.radiusSpin.setToolTip('Radius of the Zernike unit disk on the SLM, centred on the SLM.\nZernike coefficients are waves RMS over this disk.')

        form = QtWidgets.QFormLayout()
        form.addRow('Name', self.nameEdit)
        form.addRow('Board', self.boardLabel)
        form.addRow('Serial number', self.serialLabel)
        form.addRow('WFC', self._withButton(self.wfcLabel, self.wfcBtn))
        form.addRow('LUT', self._withButton(self.lutLabel, self.lutBtn))
        form.addRow('Width (px)', self.widthLabel)
        form.addRow('Height (px)', self.heightLabel)
        form.addRow('Pixel pitch (µm)', self.pitchLabel)
        form.addRow('Zernike correction radius', self.radiusSpin)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Close)
        buttons.accepted.connect(self.save_clicked)
        buttons.rejected.connect(self.reject)
        self.wfcBtn.clicked.connect(self.wfc_clicked)
        self.lutBtn.clicked.connect(self.lut_clicked)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(buttons)

        self._radius_m = 0.0
        self._shown_radius_mm = 0.0
        self.refresh(edits=True)

    @staticmethod
    def _withButton(label: QtWidgets.QLabel, button: QtWidgets.QPushButton) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(label, 1)
        lay.addWidget(button)
        return row

    def device(self) -> Device | None:
        return findDevice(self.board)

    def refresh(self, edits: bool = False) -> None:
        """Show the stored values; ``edits`` also resets the editable fields (name, radius)."""
        device = self.device()
        if device is None:
            return
        self.boardLabel.setText(str(device.board))
        self.serialLabel.setText(NOT_KNOWN_TEXT if device.serial is None else str(device.serial))
        self.wfcLabel.setText(device.factory_lut or NONE_SET_TEXT)
        self.lutLabel.setText(device.lut or NONE_SET_TEXT)
        self.widthLabel.setText(NOT_KNOWN_TEXT if device.width_px is None else str(device.width_px))
        self.heightLabel.setText(NOT_KNOWN_TEXT if device.height_px is None else str(device.height_px))
        self.pitchLabel.setText(NOT_KNOWN_TEXT if device.pixel_pitch_um is None else format(device.pixel_pitch_um, 'g'))
        if edits:
            self.nameEdit.setText(device.name or '')
            self.nameEdit.setPlaceholderText('set at the first connect (slm<serial number>)')
            self._radius_m = device.slmRadiusM()
            self.radiusSpin.setValue(self._radius_m * 1e3)
            self._shown_radius_mm = self.radiusSpin.value()

    def wfc_clicked(self) -> None:
        self.gui._setFactoryLutDialog()
        self.refresh()

    def lut_clicked(self) -> None:
        self.gui._setLutDialog()
        self.refresh()

    def save_clicked(self) -> None:
        device = self.device()
        if device is None:
            QtWidgets.QMessageBox.warning(self, 'Device parameters', 'This device is no longer in the device list.')
            self.reject()
            return
        name = self.nameEdit.text().strip()
        if name != (device.name or ''):
            if not name:
                QtWidgets.QMessageBox.warning(self, 'Device parameters', 'The name cannot be empty.')
                return
            try:
                setDeviceName(self.board, name)
            except (OSError, ValueError) as e:
                QtWidgets.QMessageBox.warning(self, 'Device parameters', 'Could not save the name:\n%s' % e)
                return
            print('%s: name set to %r' % (deviceLabel(device), name))
        value_mm = self.radiusSpin.value()
        if not math.isclose(value_mm, self._shown_radius_mm, abs_tol=1e-12):
            if not self.gui._applySlmRadiusM(self.board, value_mm / 1e3):
                return
        self.accept()
