"""Aberration / Zernike control panel."""

import json
import os

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from MyQtLineEdit.myqtLineEdit import MyQtLineEdit
from gui.constants import ZERNIKE_CORRECTIONS_DIR, ZERNIKE_DEFAULT_COEFFS_JSON



class AberrationControlPanel(QtWidgets.QWidget):
    """Owns this panel's widgets and handlers. Shared phase state is on ``self.gui``."""

    def __init__(self, gui, parent=None):
        super().__init__(parent)
        self.gui = gui
        self._build()
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.root)

    def _build(self):
        """Zernike / aberration sliders and save (`self.Abberation_control_layout`)."""
        self.root = pg.LayoutWidget()
        self.Abberation_control_layout = self.root

        self.label_Abberation_control_layout = QtWidgets.QLabel(""" -- ABERRATION CONTROL -- """)
        self.label_Abberation_control_layout.setAlignment(QtCore.Qt.AlignCenter)
        self.Abberation_control_layout.addWidget(self.label_Abberation_control_layout, row=0, colspan = 2)

        self.aber_lineEdit_list = []
        for i, aberration_name in enumerate(self.gui.aberration_label_list):
            label = QtWidgets.QLabel(aberration_name)
            lineEdit = MyQtLineEdit('0')
            lineEdit.setFixedWidth(50)
            self.Abberation_control_layout.addWidget(label, row=1+i, col = 0)
            self.Abberation_control_layout.addWidget(lineEdit, row=1+i, col = 1)

            lineEdit.returnPressed.connect(self.updateAberration)

            self.aber_lineEdit_list.append(lineEdit)

        self.loadDefaultZernikeCoeffsBtn = QtWidgets.QPushButton('Load default coefficients')
        _zernike_save_row = 1 + len(self.gui.aberration_label_list)
        self.Abberation_control_layout.addWidget(self.loadDefaultZernikeCoeffsBtn, row=_zernike_save_row, col=0, colspan=2)
        self.loadDefaultZernikeCoeffsBtn.clicked.connect(self._load_default_zernike_coefficients_json)

        _coef_row = _zernike_save_row + 1
        self.loadZernikeCoeffsBtn = QtWidgets.QPushButton('Load coefficients…')
        self.saveZernikeCoeffsBtn = QtWidgets.QPushButton('Save coefficients…')
        self.Abberation_control_layout.addWidget(self.loadZernikeCoeffsBtn, row=_coef_row, col=0)
        self.Abberation_control_layout.addWidget(self.saveZernikeCoeffsBtn, row=_coef_row, col=1)
        self.loadZernikeCoeffsBtn.clicked.connect(self._load_zernike_coefficients_json)
        self.saveZernikeCoeffsBtn.clicked.connect(self._save_zernike_coefficients_json)

    def updateAberration(self, aber_list= []):
        zernike_fit = np.zeros_like(self.gui.zernike_polynomials[0])
        zernike_coeff = []
        N_poly = len(self.gui.aber_lineEdit_list)

        for i in range(N_poly):
            if aber_list is not None and len(aber_list) > i:
                value = aber_list[i]
                self.gui.aber_lineEdit_list[i].setText(str(value))
            else:
                value = float(self.gui.aber_lineEdit_list[i].text())

            self.gui.aberration_values[i] = value
            zernike_coeff.append(value)
            zernike_fit += value * self.gui.zernike_polynomials[i]

        self.gui.im_zernike = zernike_fit

        self.gui.updatePhase()

    def updateAberration_ext(self,aberration_index, aberration_value):

        zernike_fit = np.zeros_like(self.gui.zernike_polynomials[0])
        zernike_coeff = []
        N_poly = len(self.gui.aber_lineEdit_list)


        for i in range(N_poly):
            if i != aberration_index:
                value = float(self.gui.aber_lineEdit_list[i].text())
            else:
                value = aberration_value
                self.gui.aber_lineEdit_list[i].setText(str(value))
                self.gui.aberration_values[i] = value
            zernike_coeff.append(value)
            zernike_fit += value * self.gui.zernike_polynomials[i]

        self.gui.im_zernike = zernike_fit

        self.gui.updatePhase()

    def updateAberration_defocus(self,value):
        self.updateAberration_ext(aberration_index = 2, aberration_value = value)

    def updateAberration_asti0(self,value):
        self.updateAberration_ext(aberration_index = 3, aberration_value = value)

    def updateAberration_asti45(self,value):
        self.updateAberration_ext(aberration_index = 4, aberration_value = value)

    def updateAberration_coma0(self,value):
        self.updateAberration_ext(aberration_index = 5, aberration_value = value)

    def updateAberration_coma90(self,value):
        self.updateAberration_ext(aberration_index = 6, aberration_value = value)

    def updateAberration_spherical(self,value):
        self.updateAberration_ext(aberration_index = 7, aberration_value = value)

    def updateAberration_trefoil0(self,value):
        self.updateAberration_ext(aberration_index = 8, aberration_value = value)

    def updateAberration_trefoil90(self,value):
        self.updateAberration_ext(aberration_index = 9, aberration_value = value)

    def updateAberration_asti2_0(self,value):
        self.updateAberration_ext(aberration_index = 10, aberration_value = value)

    def updateAberration_asti2_45(self,value):
        self.updateAberration_ext(aberration_index = 11, aberration_value = value)

    def updateAberration_coma2_0(self,value):
        self.updateAberration_ext(aberration_index = 12, aberration_value = value)

    def updateAberration_coma2_90(self,value):
        self.updateAberration_ext(aberration_index = 13, aberration_value = value)

    def updateAberration_second_sph(self,value):
        self.updateAberration_ext(aberration_index = 14, aberration_value = value)
    def _save_zernike_coefficients_json(self):
        """Save aberration line-edit values as JSON under ``Zernike_corrections/``."""
        os.makedirs(ZERNIKE_CORRECTIONS_DIR, exist_ok=True)
        default_path = os.path.join(ZERNIKE_CORRECTIONS_DIR, ZERNIKE_DEFAULT_COEFFS_JSON)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Zernike coefficients",
            default_path,
            "JSON (*.json)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path = path + ".json"
        try:
            coeffs = []
            for edt in self.gui.aber_lineEdit_list:
                t = edt.text().strip()
                if t in ("", "-", ".", "-."):
                    coeffs.append(0.0)
                else:
                    coeffs.append(float(t))
        except ValueError as e:
            QtWidgets.QMessageBox.warning(
                self, "Invalid coefficient", "Enter numeric values in all aberration fields.\n" + str(e)
            )
            return
        doc = {"format_version": 2}
        for label, c in zip(self.gui.aberration_label_list, coeffs):
            doc[label] = c
        try:
            _parent = os.path.dirname(os.path.abspath(path))
            if _parent:
                os.makedirs(_parent, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2, ensure_ascii=False)
                f.write("\n")
            print("Saved Zernike coefficients:", path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save error", str(e))

    def _load_zernike_coefficients_from_path(self, path):
        """
        Load aberration coefficients from JSON.

        * **format_version 2 (current):** each aberration name is a key, value is the coefficient.
        * **Legacy:** ``labels`` + ``coefficients`` array (format_version 1).
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load error", str(e))
            return

        legacy_coeffs = doc.get("coefficients")
        if isinstance(legacy_coeffs, list):
            labels = doc.get("labels")
            if isinstance(labels, list) and labels != list(self.gui.aberration_label_list):
                print(
                    "Warning: JSON labels differ from current GUI order; applying values by index only."
                )
            N = len(self.gui.aber_lineEdit_list)
            vals = []
            for i in range(N):
                if i < len(legacy_coeffs):
                    vals.append(float(legacy_coeffs[i]))
                else:
                    vals.append(0.0)
            if len(legacy_coeffs) > N:
                print("Warning: file had %d coefficients; only first %d used." % (len(legacy_coeffs), N))
            self.updateAberration(aber_list=vals)
            print("Loaded Zernike coefficients (legacy array format) from:", path)
            return

        _reserved = {"format_version", "labels", "coefficients"}
        vals = []
        for label in self.gui.aberration_label_list:
            if label not in doc:
                vals.append(0.0)
            else:
                try:
                    vals.append(float(doc[label]))
                except (TypeError, ValueError):
                    vals.append(0.0)
        for k in doc:
            if k in _reserved or k in self.gui.aberration_label_list:
                continue
            print("Warning: unknown key in JSON ignored:", k)
        self.updateAberration(aber_list=vals)
        print("Loaded Zernike coefficients from:", path)

    def _load_zernike_coefficients_json(self):
        """Load coefficients from JSON via file dialog (default folder: ``Zernike_corrections/``)."""
        os.makedirs(ZERNIKE_CORRECTIONS_DIR, exist_ok=True)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Zernike coefficients",
            ZERNIKE_CORRECTIONS_DIR,
            "JSON (*.json);;All files (*.*)",
        )
        if not path:
            return
        self._load_zernike_coefficients_from_path(path)

    def _load_default_zernike_coefficients_json(self):
        """Load coefficients from the default JSON in ``Zernike_corrections/`` (no dialog)."""
        path = os.path.join(ZERNIKE_CORRECTIONS_DIR, ZERNIKE_DEFAULT_COEFFS_JSON)
        if not os.path.isfile(path):
            QtWidgets.QMessageBox.warning(
                self,
                "Default coefficients",
                "File not found:\n%s\n\nUse “Save coefficients…” to create it first."
                % path,
            )
            return
        self._load_zernike_coefficients_from_path(path)
