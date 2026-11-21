"""Settings ▸ Parameters…: edit the SLM / optics parameters in parameters/parameters.json.

The values are read once at start (see gui/parameters.py), so saved changes apply after a restart. Values are shown
in convenient units (mm, µm) and stored in the file's units (m). A field that is not edited keeps its exact value.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from pyqtgraph.Qt import QtWidgets

from gui.parameters import JSON_KEYS, ParametersError, readParameters, readTemplate, writeParameters
from gui.paths import PARAMETERS_FILE

FIELDS: tuple[tuple[str, str, float], ...] = (
    ('f', 'Objective focal length (mm)', 1e3),
    ('M', 'Telescope magnification', 1.0),
)


def _formatValue(value: float, scale: float) -> str:
    return format(value * scale, '.10g')


class ParametersDialog(QtWidgets.QDialog):
    """One line per parameter; Save writes parameters.json, Defaults fills in the repository template values."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle('Parameters')
        self._edits: dict[str, QtWidgets.QLineEdit] = {}
        self._shown: dict[str, str] = {}
        self._values: dict[str, Any] = {}

        form = QtWidgets.QFormLayout()
        for name, label, _scale in FIELDS:
            edit = QtWidgets.QLineEdit()
            edit.setToolTip('%s in parameters.json' % JSON_KEYS[name])
            form.addRow(label, edit)
            self._edits[name] = edit


        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        self.defaultsBtn = buttons.addButton('Defaults', QtWidgets.QDialogButtonBox.ResetRole)
        buttons.accepted.connect(self.save_clicked)
        buttons.rejected.connect(self.reject)
        self.defaultsBtn.clicked.connect(self.defaults_clicked)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(buttons)

        try:
            values, warnings = readParameters(PARAMETERS_FILE, readTemplate())
        except ParametersError as e:
            QtWidgets.QMessageBox.critical(self, 'Parameters', str(e))
            values, warnings = None, []
        if values is not None:
            self._show(asdict(values))
        if warnings:
            QtWidgets.QMessageBox.warning(self, 'Parameters', 'Problems in the parameters file (default values shown):\n- ' + '\n- '.join(warnings))

    def _show(self, values: dict[str, Any]) -> None:
        for name, _label, scale in FIELDS:
            text = _formatValue(values[name], scale)
            self._edits[name].setText(text)
            self._shown[name] = text
            self._values[name] = values[name]

    def defaults_clicked(self) -> None:
        try:
            self._show(asdict(readTemplate()))
        except ParametersError as e:
            QtWidgets.QMessageBox.critical(self, 'Parameters', str(e))

    def editedValues(self) -> dict[str, Any]:
        """Values in the file's units; raises ValueError naming the field that is not a number."""
        values: dict[str, Any] = {}
        for name, label, scale in FIELDS:
            text = self._edits[name].text().strip()
            if text == self._shown.get(name) and name in self._values:
                values[name] = self._values[name]
                continue
            try:
                number = float(text)
            except ValueError:
                raise ValueError('%s: %r is not a number' % (label, text)) from None
            values[name] = number / scale
        return values

    def save_clicked(self) -> None:
        try:
            writeParameters(self.editedValues(), PARAMETERS_FILE)
        except (ValueError, OSError) as e:
            QtWidgets.QMessageBox.warning(self, 'Parameters', 'Not saved:\n%s' % e)
            return
        print('Parameters saved to %s (used from the next start)' % PARAMETERS_FILE)
        QtWidgets.QMessageBox.information(self, 'Parameters', 'Saved.\nRestart the SLM Controller to use the new parameters.')
        self.accept()
