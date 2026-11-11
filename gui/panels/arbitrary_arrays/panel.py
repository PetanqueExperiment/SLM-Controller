"""Arbitrary Arrays panel: widgets and button handlers.

The work is done in the modules next to this one: rois.py (rois.json files), tweezer_array.py (lattice geometry),
traps.py (k-vectors / phases / pupils), gui/hologram_generator.py (WGS, GS), gui/hologram_handler.py (Save / Load Hologram),
trap_plot.py (Plot traps file), equalization.py (dormant atom equalization) and legacy.py (old unused code).
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

import libslm.parameters as param
from MyQtLineEdit.myqtLineEdit import MyQtLineEdit
from gui.constants import SLM_PATTERNS_DIR, SYSTEM_MAGNIFICATION, TWEEZER_ROIS_DIR
from gui.helpers import framed_panel, parse_pasted_coordinates
from gui import hologram_generator
from gui.panels.arbitrary_arrays import rois
from gui.panels.arbitrary_arrays.equalization import EqualizationMixin
from gui.hologram_handler import readLegacyPattern, readTrapsDict, writeTrapsDict
from gui.panels.arbitrary_arrays.legacy import LegacyAlgorithmsMixin
from gui.panels.arbitrary_arrays.trap_plot import plotTrapCenters, plotTrapsFile
from gui.panels.arbitrary_arrays.traps import trapArrays
from gui.panels.arbitrary_arrays.tweezer_array import coords_npy_to_traps_pos, rectangular_lattice_coords

if TYPE_CHECKING:
    from gui.main_window import SLMGUI

NO_ROIS_LOADED_TEXT: str = 'Loaded ROIs: none'
HOLOGRAM_BLANK_TEXT: str = 'Displayed hologram: blank'
HOLOGRAM_IN_MEMORY_TEXT: str = 'Displayed hologram: in local memory (not saved)'
COMPUTED_NONE_TEXT: str = 'Computed hologram: none pending'


def _pump_gui() -> None:
    """Flush log output and let Qt redraw while a long algorithm runs on the GUI thread."""
    try:
        sys.stdout.flush()
    except Exception:
        pass
    try:
        sys.stderr.flush()
    except Exception:
        pass
    app = QtWidgets.QApplication.instance()
    if app is not None:
        app.processEvents(QtCore.QEventLoop.AllEvents)


class ArbitraryArraysPanel(EqualizationMixin, LegacyAlgorithmsMixin, QtWidgets.QWidget):
    """Owns this panel's widgets and handlers. Shared phase state is on ``self.gui``."""

    def __init__(self, gui: SLMGUI, parent: QtWidgets.QWidget | None = None, gap_px: int = 4) -> None:
        super().__init__(parent)
        self.gui = gui
        self._rois_path: str | None = None
        self._loaded_rois_name: str | None = None
        self._hologram_source: tuple[object, str] | None = None
        self._computed_hologram: np.ndarray | None = None
        self._computed_saved_as: str | None = None
        self._build()
        # Two framed panels stacked: ROI generator (top) | loaded traps + hologram generator + hologram files (bottom).
        traps = QtWidgets.QWidget()
        traps_lay = QtWidgets.QVBoxLayout(traps)
        traps_lay.setContentsMargins(0, 0, 0, 0)
        traps_lay.setSpacing(gap_px)
        traps_lay.addWidget(self.root)
        traps_lay.addWidget(self.generator_root)
        traps_lay.addWidget(self.files_root)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(gap_px)
        lay.addWidget(framed_panel(self.roi_generator_root))
        lay.addWidget(framed_panel(traps))

    def _build(self) -> None:
        """Build the ROI generator (`self.roi_generator_root`), loaded traps (`self.root`), hologram generator and files sections."""
        self.roi_generator_root = pg.LayoutWidget()
        self.root = pg.LayoutWidget()
        self.arbitrary_arrays_panel = self.root

        self.label_roi_generator = QtWidgets.QLabel(""" -- ROI GENERATOR -- """)
        self.label_roi_generator.setAlignment(QtCore.Qt.AlignCenter)
        self.label_Layout_Arbitrary_Array = QtWidgets.QLabel(""" -- ARBITRARY ARRAYS -- """)
        self.label_Layout_Arbitrary_Array.setAlignment(QtCore.Qt.AlignCenter)
        self.generateBlankBtn = QtWidgets.QPushButton('Blank')
        self.loadBtn = QtWidgets.QPushButton('Load Hologram')
        self.saveBtn = QtWidgets.QPushButton('Save Hologram')
        self.latticeGridLabel = QtWidgets.QLabel('Array generator')
        self.latticeRowsLabel = QtWidgets.QLabel('rows')
        self.latticeRowsSpin = QtWidgets.QSpinBox()
        self.latticeRowsSpin.setRange(1, 4096)
        self.latticeRowsSpin.setValue(5)
        self.latticeColsLabel = QtWidgets.QLabel('cols')
        self.latticeColsSpin = QtWidgets.QSpinBox()
        self.latticeColsSpin.setRange(1, 4096)
        self.latticeColsSpin.setValue(5)
        self.latticeSepXLabel = QtWidgets.QLabel('Δx (µm)')
        self.latticeSepXEdt = MyQtLineEdit('10')
        self.latticeSepYLabel = QtWidgets.QLabel('Δy (µm)')
        self.latticeSepYEdt = MyQtLineEdit('10')
        self.latticeSepZLabel = QtWidgets.QLabel('Δz (µm)')
        self.latticeSepZEdt = MyQtLineEdit('0')
        self.latticeRotLabel = QtWidgets.QLabel('Rot z (°)')
        self.latticeRotEdt = MyQtLineEdit('90')
        self.latticeGapLabel = QtWidgets.QLabel('Center gap (µm)')
        self.latticeGapEdt = MyQtLineEdit('0')
        self.loadArrayCoordinatesBtn = QtWidgets.QPushButton('Use')
        self.loadArrayCoordinatesBtn.setToolTip('Use the array as the current traps for Run, without writing a file.')
        self.saveArrayCoordinatesBtn = QtWidgets.QPushButton('Save')
        self.saveArrayCoordinatesBtn.setToolTip('Write the array to a rois.json without loading it.')
        self.saveAndLoadArrayCoordinatesBtn = QtWidgets.QPushButton('Save && use')
        self.saveAndLoadArrayCoordinatesBtn.setToolTip('Write the array to a rois.json and use it as the current traps.')
        self.generateTrapsBtn = QtWidgets.QPushButton('Load ROIs')
        self.generateTrapsBtn.setToolTip('Choose a rois.json and load it as the current traps.')
        self.loadedRoisLabel = QtWidgets.QLabel(NO_ROIS_LOADED_TEXT)
        self.loadedRoisLabel.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.plotTrapsBtn = QtWidgets.QPushButton('Plot from file...')
        self.plotTrapsBtn.setToolTip('Choose one or more rois.json files and plot their trap centres (nothing is loaded).')
        self.plotLoadedTrapsBtn = QtWidgets.QPushButton('Plot')
        self.plotLoadedTrapsBtn.setToolTip('Plot the currently loaded traps (camera pixels).')
        self.coordsToJsonBtn = QtWidgets.QPushButton('Coords → JSON')
        self.pastedCoordsEdt = QtWidgets.QLineEdit()
        self.pastedCoordsEdt.setPlaceholderText('[[x, y], ...] or np.array([[x, y, z], ...])')
        self.pastedCoordsUnitsCB = QtWidgets.QComboBox()
        self.pastedCoordsUnitsCB.addItem('px (camera)')
        self.pastedCoordsUnitsCB.addItem('µm (focal plane)')
        self.pastedCoordsUnitsCB.setToolTip(
            'px: values are written as-is into "centers".\n'
            'µm: same convention as the array generator (origin at image center, × SYSTEM_MAGNIFICATION).'
        )
        self.runBtn = QtWidgets.QPushButton('Run && display')
        self.runBtn.setToolTip('Compute the hologram and send it to the SLM.')
        self.runOnlyBtn = QtWidgets.QPushButton('Run')
        self.runOnlyBtn.setToolTip('Compute the hologram without sending it to the SLM; Save Hologram saves it.')
        self.computedHologramLabel = QtWidgets.QLabel(COMPUTED_NONE_TEXT)
        self.computedHologramLabel.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.iterationsEdit = MyQtLineEdit(str(self.gui.GS_maxit))
        self.methodCB = QtWidgets.QComboBox()
        for key in getattr(self.gui, 'methodList', {}) or []:
            self.methodCB.addItem(key)

        self.equalize_with_atoms_Btn = QtWidgets.QPushButton('Equalize with atoms')
        self.computeHologram_Btn = QtWidgets.QPushButton('Compute hologram')
        self.devide_slm_area_Btn = QtWidgets.QPushButton('Partial Phase Correction')

        gen = self.roi_generator_root
        line_idx = 0
        gen.addWidget(self.label_roi_generator, row=line_idx, col=0, colspan=4)

        line_idx += 1
        gen.addWidget(self.coordsToJsonBtn, row=line_idx, col=0)
        gen.addWidget(self.pastedCoordsEdt, row=line_idx, col=1, colspan=2)
        gen.addWidget(self.pastedCoordsUnitsCB, row=line_idx, col=3)

        line_idx += 1
        gen.addWidget(self.latticeGridLabel, row=line_idx, col=0)

        line_idx += 1
        gen.addWidget(self.latticeRowsLabel, row=line_idx, col=0)
        gen.addWidget(self.latticeRowsSpin, row=line_idx, col=1)
        gen.addWidget(self.latticeColsLabel, row=line_idx, col=2)
        gen.addWidget(self.latticeColsSpin, row=line_idx, col=3)

        line_idx += 1
        gen.addWidget(self.latticeSepXLabel, row=line_idx, col=0)
        gen.addWidget(self.latticeSepXEdt, row=line_idx, col=1)
        gen.addWidget(self.latticeSepYLabel, row=line_idx, col=2)
        gen.addWidget(self.latticeSepYEdt, row=line_idx, col=3)

        line_idx += 1
        gen.addWidget(self.latticeSepZLabel, row=line_idx, col=0)
        gen.addWidget(self.latticeSepZEdt, row=line_idx, col=1)
        gen.addWidget(self.latticeRotLabel, row=line_idx, col=2)
        gen.addWidget(self.latticeRotEdt, row=line_idx, col=3)

        line_idx += 1
        gen.addWidget(self.latticeGapLabel, row=line_idx, col=0)
        gen.addWidget(self.latticeGapEdt, row=line_idx, col=1)

        line_idx += 1
        save_row = QtWidgets.QWidget()
        save_row_lay = QtWidgets.QHBoxLayout(save_row)
        save_row_lay.setContentsMargins(0, 0, 0, 0)
        save_row_lay.addWidget(self.loadArrayCoordinatesBtn)
        save_row_lay.addWidget(self.saveArrayCoordinatesBtn)
        save_row_lay.addWidget(self.saveAndLoadArrayCoordinatesBtn)
        gen.addWidget(save_row, row=line_idx, col=0, colspan=4)

        line_idx = 0
        self.arbitrary_arrays_panel.addWidget(self.label_Layout_Arbitrary_Array, row=line_idx, col=0, colspan=3)

        line_idx += 1
        self.arbitrary_arrays_panel.addWidget(self.loadedRoisLabel, row=line_idx, col=0, colspan=3)

        line_idx += 1
        self.arbitrary_arrays_panel.addWidget(self.generateTrapsBtn, row=line_idx, col=0)
        self.arbitrary_arrays_panel.addWidget(self.plotLoadedTrapsBtn, row=line_idx, col=1)
        self.arbitrary_arrays_panel.addWidget(self.plotTrapsBtn, row=line_idx, col=2)

        # Hologram generator (gui/hologram_generator.py) and hologram files (gui/hologram_handler.py): own panels below
        self.generator_root = pg.LayoutWidget()
        self.label_hologram_generator = QtWidgets.QLabel(""" -- HOLOGRAM GENERATOR -- """)
        self.label_hologram_generator.setAlignment(QtCore.Qt.AlignCenter)
        self.generator_root.addWidget(self.label_hologram_generator, row=0, col=0, colspan=5)
        self.generator_root.addWidget(self.runOnlyBtn, row=1, col=0)
        self.generator_root.addWidget(self.runBtn, row=1, col=1, colspan=4)
        self.generator_root.addWidget(self.methodCB, row=2, col=0)
        self.generator_root.addWidget(self.iterationsEdit, row=2, col=1)
        self.generator_root.addWidget(self.computedHologramLabel, row=3, col=0, colspan=5)
        for w in (self.runOnlyBtn, self.runBtn, self.methodCB, self.iterationsEdit):
            w.setSizePolicy(QtWidgets.QSizePolicy.Ignored, w.sizePolicy().verticalPolicy())
        for col in range(5):
            self.generator_root.layout.setColumnStretch(col, 1)

        self.files_root = pg.LayoutWidget()
        self.label_hologram_files = QtWidgets.QLabel(""" -- HOLOGRAM FILES -- """)
        self.label_hologram_files.setAlignment(QtCore.Qt.AlignCenter)
        self.files_root.addWidget(self.label_hologram_files, row=0, col=0, colspan=3)
        self.files_root.addWidget(self.generateBlankBtn, row=1, col=0)
        self.files_root.addWidget(self.loadBtn, row=1, col=1)
        self.files_root.addWidget(self.saveBtn, row=1, col=2)
        self.displayedHologramLabel = QtWidgets.QLabel(HOLOGRAM_BLANK_TEXT)
        self._displayed_hologram_labels: list[QtWidgets.QLabel] = [self.displayedHologramLabel]
        self.displayedHologramLabel.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.files_root.addWidget(self.displayedHologramLabel, row=2, col=0, colspan=3)

        # line_idx += 1
        # self.arbitrary_arrays_panel.addWidget(self.equalize_with_atoms_Btn, row=line_idx, col=0)
        # self.arbitrary_arrays_panel.addWidget(self.computeHologram_Btn, row=line_idx, col=1)
        # self.arbitrary_arrays_panel.addWidget(self.devide_slm_area_Btn, row=line_idx, col=2)

        self.generateBlankBtn.clicked.connect(self.generateBlank)
        self.loadBtn.clicked.connect(self.loadPattern_clicked)
        self.saveBtn.clicked.connect(self.savePattern_clicked)
        self.generateTrapsBtn.clicked.connect(self.generateTraps)
        self.loadArrayCoordinatesBtn.clicked.connect(self.loadArrayCoordinates_clicked)
        self.saveArrayCoordinatesBtn.clicked.connect(self.saveArrayCoordinates_clicked)
        self.saveAndLoadArrayCoordinatesBtn.clicked.connect(self.saveAndLoadArrayCoordinates_clicked)
        self.plotTrapsBtn.clicked.connect(self.plotTrapsFile_clicked)
        self.plotLoadedTrapsBtn.clicked.connect(self.plotLoadedTraps_clicked)
        self.coordsToJsonBtn.clicked.connect(self.pastedCoordsToJson_clicked)
        self.runBtn.clicked.connect(self.runAlgorithm_clicked)
        self.runOnlyBtn.clicked.connect(self.runOnly_clicked)
        # self.equalize_with_atoms_Btn.clicked.connect(self.equalize_with_atoms)
        # self.devide_slm_area_Btn.clicked.connect(self.devide_slm_area)
        # self.computeHologram_Btn.clicked.connect(self.computeHologram_clicked)

    # ==============================================================================
    # Algorithms (Run Algorithm → gui.methodList → WGS)
    # ==============================================================================
    def runAlgorithm_clicked(self) -> None:
        method = str(self.methodCB.currentText())
        self.runAlgorithm(method)

    def runOnly_clicked(self) -> None:
        method = str(self.methodCB.currentText())
        self.runAlgorithm(method, display=False)

    def runAlgorithm(self, method: str, display: bool = True) -> None:
        '''
        This function chooses the method to calculate the phase mask for
        the desired trap positions. With ``display`` False the result is kept as the computed hologram
        (saved by Save Hologram) and nothing is sent to the SLM.
        '''
        if method not in self.gui.methodList:
            print('Run aborted: unknown method %r' % method)
            return
        try:
            iterations = int(self.iterationsEdit.text())
        except ValueError:
            print('Run aborted: the number of iterations must be an integer.')
            return
        if iterations < 1:
            print('Run aborted: the number of iterations must be at least 1.')
            return
        # Each run starts from the loaded trap positions, with fresh k-vectors, phases, weights and pupils
        if self._rois_path is not None:
            if not self._load_traps_from_pattern_file(self._rois_path, verbose=True):
                print('Run aborted: could not reload', self._rois_path)
                return
        elif not self._traps_ready():
            print('Run aborted: no ROIs loaded. Use Load ROIs, or Use / Save & use in the ROI generator.')
            return
        if not self._prepareTrapArrays():
            print('Run aborted: could not prepare the traps.')
            return

        image = self.gui.methodList[method]()

        if image is None:
            return

        if not display:
            self._computed_hologram = image
            self._computed_saved_as = None
            self._refreshComputedLabel()
            print('Hologram computed, not sent to the SLM (Save Hologram saves it).')
            return
        self._clearComputedHologram()

        # Put the phase pattern on the GUI and on the SLM
        self.gui.phasePattern = image
        self.gui.updatePhase()

    def computeHologram_clicked(self) -> None:
        print('Computed hologram')
        phi = self.computeHologram()
        if phi is None:
            return
        self.gui.phasePattern = phi
        self.gui.updatePhase()

    def computeHologram(self) -> np.ndarray | None:
        if not self._ensure_traps_loaded():
            return None
        return hologram_generator.computeHologram(self.gui.traps_k, self.gui.theta_traps, self.gui.weight_traps, self.gui.pupil_traps, self.gui.SLM_shape)

    def WGS(self, correct_diffraction_efficiency: bool = False) -> np.ndarray | None:
        '''
        Weighted Gerchberg−Saxton (GS). Otpimize the phase and weigth of each grating.
        '''
        print('Starting weighted Gerchberg-Saxton calculation')
        _pump_gui()
        if not self._ensure_traps_loaded():
            return None
        result = hologram_generator.wgs(self.gui.traps_pos, self.gui.traps_k, self.gui.theta_traps, self.gui.weight_traps, self.gui.pupil_traps, self.gui.SLM_shape, int(self.iterationsEdit.text()), self.gui.dx, self.gui.dy, correct_diffraction_efficiency, progress=_pump_gui)
        if result is None:
            return None
        phi, self.gui.theta_traps, self.gui.weight_traps = result
        return phi

    def GS(self) -> np.ndarray | None:
        '''
        Gerchberg−Saxton (GS). Optimize the phase of each grating. (Not in the Method menu.)
        '''
        print('Starting Gerchberg-Saxton calculation')
        if not self._ensure_traps_loaded():
            return None
        phi, self.gui.theta_traps, self.gui.weight_traps = hologram_generator.gs(self.gui.traps_k, self.gui.pupil_traps, self.gui.SLM_shape, int(self.iterationsEdit.text()), progress=_pump_gui)
        return phi

    # ==============================================================================
    # Traps (rois.json → traps_pos → k-vectors, phases, weights, pupils)
    # ==============================================================================
    def generateTraps(self) -> None:
        """Choose a PyCam ``rois.json`` and load it as the current traps (see ``_load_traps_from_pattern_file``)."""
        os.makedirs(TWEEZER_ROIS_DIR, exist_ok=True)
        start = self._rois_path or TWEEZER_ROIS_DIR
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load trap coordinates (rois.json)', start, 'PyCam ROIs (*.json);;All files (*.*)')
        if not path:
            return
        print('Loading ROIs')
        self._load_traps_from_pattern_file(path, verbose=True)

    def _setLoadedRois(self, path: str | None, source: str | None = None, description: str | None = None) -> None:
        """Remember the loaded ``rois.json`` (None if traps came from elsewhere) and show it in `loadedRoisLabel`."""
        self._rois_path = path
        if path is not None:
            self._loaded_rois_name = os.path.basename(path)
        elif source is not None:
            self._loaded_rois_name = 'from %s' % os.path.basename(source)
        else:
            self._loaded_rois_name = description
        if path is None and source is None and description is not None:
            n = getattr(self.gui, 'N_traps', 0)
            self.loadedRoisLabel.setText('Loaded ROIs: %s (%d traps)' % (description, n))
            self.loadedRoisLabel.setToolTip('')
        elif path is not None:
            n = getattr(self.gui, 'N_traps', 0)
            self.loadedRoisLabel.setText('Loaded ROIs: %s (%d traps)' % (os.path.basename(path), n))
            self.loadedRoisLabel.setToolTip(os.path.abspath(path))
        elif source is not None:
            n = getattr(self.gui, 'N_traps', 0)
            self.loadedRoisLabel.setText('Loaded ROIs: from %s (%d traps)' % (os.path.basename(source), n))
            self.loadedRoisLabel.setToolTip(os.path.abspath(source))
        else:
            self.loadedRoisLabel.setText(NO_ROIS_LOADED_TEXT)
            self.loadedRoisLabel.setToolTip('')

    def _traps_ready(self) -> bool:
        """True if trap positions are loaded (from a rois.json, the ROI generator or a hologram file)."""
        tp = getattr(self.gui, 'traps_pos', None)
        if not isinstance(tp, np.ndarray) or tp.ndim != 2 or tp.shape[1] != 4:
            return False
        return tp.shape[0] >= 1

    def _trapArraysReady(self) -> bool:
        """True if the loaded positions also have k-vectors, phases, weights and pupils (prepared, or from a hologram file)."""
        if not self._traps_ready():
            return False
        n = self.gui.traps_pos.shape[0]
        tk = getattr(self.gui, 'traps_k', None)
        theta = getattr(self.gui, 'theta_traps', None)
        weight = getattr(self.gui, 'weight_traps', None)
        if not isinstance(tk, np.ndarray) or tk.shape != (n, 3):
            return False
        if not isinstance(theta, np.ndarray) or theta.shape != (n,):
            return False
        if not isinstance(weight, np.ndarray) or weight.shape != (n,):
            return False
        return isinstance(getattr(self.gui, 'pupil_traps', None), dict)

    def _prepareTrapArrays(self) -> bool:
        """Build k-vectors (current wavelength), random phases, unit weights and pupils for the loaded positions."""
        try:
            self._finalize_trap_generation()
        except ValueError as e:
            print('Could not prepare traps (check the pupil field): %s' % e)
            return False
        return True

    def _load_traps_from_pattern_file(self, path: str, verbose: bool = False) -> bool:
        """
        Load trap coordinates from the PyCam ``rois.json`` at ``path`` and make it the loaded ROIs file.
        Pixel centers are converted to focal-plane positions using ``ROI_JSON_UM_PER_CAMERA_PIXEL``
        and the same origin as the array generator. Axial ``z`` is set to 0 (not stored in JSON).

        Returns True on success. Leaves the current traps unchanged if the file is missing, unreadable or has no centers.
        """
        if not os.path.isfile(path):
            print('rois.json not found:', path)
            return False
        if verbose:
            print('Loading traps from', path)
        try:
            coords_m = rois.readTrapCoordsM(path, verbose=verbose)
        except (OSError, ValueError, TypeError) as e:
            print('Could not read %s: %s' % (path, e))
            return False
        if coords_m is None:
            return False
        if not self._loadTrapsFromCoords(coords_m, path):
            return False
        self._setLoadedRois(path)
        return True

    def _loadTrapsFromCoords(self, coords_m: np.ndarray, source: str) -> bool:
        """Make the (N, 3) positions in metres the current trap positions; Run Algorithm prepares the rest. False if invalid."""
        try:
            traps_pos = coords_npy_to_traps_pos(coords_m)
        except ValueError as e:
            print('Could not build traps from %s: %s' % (source, e))
            return False
        self.gui.traps_pos = traps_pos
        self.gui.N_traps = int(traps_pos.shape[0])
        self._clearComputedHologram()
        self.gui.traps_k = None
        self.gui.theta_traps = None
        self.gui.weight_traps = None
        self.gui.pupil_traps = None
        return True

    def _ensure_traps_loaded(self) -> bool:
        """
        Ensure traps are ready for the algorithms: positions loaded (reloading the ROIs file if needed) and prepared
        (after loading a .pkl or a Run Algorithm this is a no-op).
        """
        if not self._traps_ready():
            if self._rois_path is None:
                print('No ROIs loaded. Use Load ROIs, or Use / Save & use in the ROI generator.')
                return False
            print('Trap coordinates not in memory; reloading', self._rois_path)
            if not self._load_traps_from_pattern_file(self._rois_path, verbose=False):
                return False
        if self._trapArraysReady():
            return True
        return self._prepareTrapArrays()

    def _finalize_trap_generation(self) -> None:
        """Build k-vectors, random phases, pupils, and traps_dict from self.gui.traps_pos (meters)."""
        arrays = trapArrays(self.gui.traps_pos, self.gui.wavelength_m, self.gui.pupilEdt.text())
        self.gui.N_traps = arrays['N_traps']
        self.gui.traps_k = arrays['traps_k']
        self.gui.theta_traps = arrays['theta_traps']
        self.gui.weight_traps = arrays['weight_traps']
        self.gui.pupil_traps = arrays['pupil_traps']
        self.gui.traps_dict = {
            'traps_pos': self.gui.traps_pos,
            'traps_k': self.gui.traps_k,
            'theta_traps': self.gui.theta_traps,
            'weight_traps': self.gui.weight_traps,
            'pupil_traps': self.gui.pupil_traps,
        }

    # ==============================================================================
    # rois.json: array generator, pasted coordinates, plot
    # ==============================================================================
    def loadArrayCoordinates_clicked(self) -> None:
        """Load the rectangular lattice as the current traps without writing a file (same conversion as a saved rois.json)."""
        lattice = self._latticeFromFields()
        if lattice is None:
            return
        centers, _side, _rot = lattice
        coords_m = rois.centersToCoordsM(centers)
        if coords_m is None or not self._loadTrapsFromCoords(coords_m, 'the array generator'):
            return
        self._setLoadedRois(None, description='generated array, not saved')
        print('Loaded generated array into memory (%d traps, not saved)' % coords_m.shape[0])

    def _latticeFromFields(self) -> tuple[list[list[float]], int, float] | None:
        """Rectangular lattice from the generator fields as rois.json camera centres, ROI side [px] and rotation [deg]."""
        try:
            spacing_cal = float(SYSTEM_MAGNIFICATION)
            sx = float(self.latticeSepXEdt.text()) * spacing_cal
            sy = float(self.latticeSepYEdt.text()) * spacing_cal
            sz = float(self.latticeSepZEdt.text())
            rot = float(self.latticeRotEdt.text())
            gap = float(self.latticeGapEdt.text()) * spacing_cal
        except ValueError:
            print('Invalid lattice parameter (need numbers for Δx, Δy, Δz, rotation, center gap).')
            return None
        arr = rectangular_lattice_coords(self.latticeRowsSpin.value(), self.latticeColsSpin.value(), sx, sy, sep_z_um=sz, rotation_deg_z=rot, center_gap_um=gap)
        centers, side = rois.latticeCenters(arr, sx, sy)
        return centers, side, rot

    def saveArrayCoordinates_clicked(self) -> None:
        """Save the rectangular lattice as PyCam / ROIfinder `rois.json` without loading it."""
        self._saveArrayCoordinates()

    def saveAndLoadArrayCoordinates_clicked(self) -> None:
        """Save the rectangular lattice as `rois.json` and load it as the current traps."""
        path = self._saveArrayCoordinates()
        if path is not None:
            self._load_traps_from_pattern_file(path, verbose=True)

    def _saveArrayCoordinates(self) -> str | None:
        """Ask for a file name and write the rectangular lattice there. Returns the written path, or None."""
        lattice = self._latticeFromFields()
        if lattice is None:
            return None
        centers, side, rot = lattice
        os.makedirs(TWEEZER_ROIS_DIR, exist_ok=True)
        default_path = os.path.join(TWEEZER_ROIS_DIR, 'rois.json')
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save trap coordinates (rois.json)', default_path, 'PyCam ROIs (*.json)')
        if not path:
            return None

        path = path.strip()
        low = path.lower()
        _parent = os.path.dirname(os.path.abspath(path))
        if _parent:
            os.makedirs(_parent, exist_ok=True)

        if not low.endswith('.json'):
            root, ext = os.path.splitext(path)
            path = root + '.json' if ext else path + '.json'
        try:
            rois.writeRoisJson(path, centers, side, rot)
        except OSError as e:
            print('Could not save %s: %s' % (path, e))
            return None
        print('Saved PyCam ROIs:', path, '(%d centers)' % len(centers))
        return path

    def pastedCoordsToJson_clicked(self) -> None:
        """Convert the pasted coordinate list / np.array into a PyCam ``rois.json`` chosen via a save dialog."""
        try:
            coords = parse_pasted_coordinates(self.pastedCoordsEdt.text())
        except ValueError as e:
            print('Invalid pasted coordinates:', e)
            return
        if coords.shape[1] == 3 and np.any(coords[:, 2] != 0):
            print('Note: z column ignored (rois.json stores only x, y).')
        centers, side = rois.pastedCenters(coords, in_um=self.pastedCoordsUnitsCB.currentIndex() == 1)

        os.makedirs(TWEEZER_ROIS_DIR, exist_ok=True)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save pasted coordinates (rois.json)', os.path.join(TWEEZER_ROIS_DIR, 'rois_custom.json'), 'PyCam ROIs (*.json)')
        if not path:
            return
        path = path.strip()
        if not path.lower().endswith('.json'):
            path = os.path.splitext(path)[0] + '.json'
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        try:
            rois.writeRoisJson(path, centers, side, 0.0)
        except OSError as e:
            print('Could not save %s: %s' % (path, e))
            return
        print('Saved PyCam ROIs:', path, '(%d centers, roi size %d px)' % (len(centers), side))

    def plotLoadedTraps_clicked(self) -> None:
        """Plot the loaded traps (from a rois.json, the ROI generator or a hologram file) in the trap plot window."""
        if not self._traps_ready():
            print('No ROIs loaded: nothing to plot.')
            return
        centers = rois.trapsPosToCentersPx(self.gui.traps_pos)
        label = '%s, loaded' % (self._loaded_rois_name or 'traps')
        plotTrapCenters(self.gui, centers, label)
        print('Plotted %d loaded traps' % centers.shape[0])

    def plotTrapsFile_clicked(self) -> None:
        """Pick one or more ``rois.json`` files and overlay their trap centers (camera pixels) in one window."""
        start = self._rois_path or TWEEZER_ROIS_DIR
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, 'Plot trap coordinates', start, 'PyCam ROIs (*.json);;All files (*.*)')
        for path in paths:
            plotTrapsFile(self.gui, path)

    # ==============================================================================
    # Hologram files (Save / Load Hologram) and Blank
    # ==============================================================================
    def generateBlank(self) -> None:
        '''Generates a blank phase mask'''
        self.gui.phasePattern = np.zeros([param.ImgResY, param.ImgResX])
        self.gui.updatePhase()

    def savePattern_clicked(self) -> None:
        '''
        Opens the dialog and saves the selected file
        '''
        os.makedirs(SLM_PATTERNS_DIR, exist_ok=True)
        self.gui.file = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Hologram', SLM_PATTERNS_DIR)[0]
        if not self.gui.file:
            return
        self.savePattern(name=self.gui.file)

    def savePattern(self, name: str = '') -> None:
        '''
        Saves the hologram and its traps to ``name`` (.pkl added if missing)
        '''
        if not self._ensure_traps_loaded():
            print('Save pattern aborted: trap coordinates are not available.')
            return

        self.gui.file = name
        if not (self.gui.file[-4:] == '.pkl'):
            self.gui.file += '.pkl'

        pending = self._computed_hologram
        self.pack_traps_dict(pending if pending is not None else self.gui.phasePattern)
        try:
            writeTrapsDict(self.gui.file, self.gui.traps_dict)
        except OSError as e:
            print('Could not save hologram %s: %s' % (self.gui.file, e))
            return
        if pending is not None:
            self._computed_saved_as = self.gui.file
            self._refreshComputedLabel()
            print('Saved the computed (not displayed) hologram to', self.gui.file)
        else:
            self._setHologramSource(self.gui.file)

    def loadPattern_clicked(self) -> None:
        '''
        Open the dialog and gets the selected file
        '''
        os.makedirs(SLM_PATTERNS_DIR, exist_ok=True)
        self.gui.file = QtWidgets.QFileDialog.getOpenFileName(self, 'Load Hologram', SLM_PATTERNS_DIR)[0]
        if not self.gui.file:
            return
        self.loadPattern(name=self.gui.file)

    def loadPattern(self, name: str = '') -> None:
        print('SLM file')
        print(name)
        self.gui.file = name

        # current
        if name[-3:] == 'pkl':
            try:
                self.gui.traps_dict = readTrapsDict(self.gui.file)
                self.unpack_traps_dict()
                self._setLoadedRois(None, source=name)
                self._setHologramSource(name)
            except Exception as e:
                print('Could not load traps_dict')
                raise e
        # obsolete
        elif name[-3:] in ('npy', 'bmp'):
            self.gui.phasePattern = readLegacyPattern(name)
            self._setHologramSource(name)
        elif name == None:
            pass
        else:
            print('Incorrect file type')
            print('You are too stupid!!!!!!')  ### Chew and Sylvain.

        self.gui.updatePhase()

    def _clearComputedHologram(self) -> None:
        """Drop the computed-but-not-displayed hologram (it no longer matches the loaded traps, or it was displayed)."""
        self._computed_hologram = None
        self._computed_saved_as = None
        self._refreshComputedLabel()

    def _refreshComputedLabel(self) -> None:
        """Show whether a computed hologram is waiting (not displayed) and whether it was saved."""
        if self._computed_hologram is None:
            self.computedHologramLabel.setText(COMPUTED_NONE_TEXT)
            self.computedHologramLabel.setToolTip('')
        elif self._computed_saved_as is None:
            self.computedHologramLabel.setText('Computed hologram: not displayed, not saved (Save Hologram saves it)')
            self.computedHologramLabel.setToolTip('')
        else:
            self.computedHologramLabel.setText('Computed hologram: not displayed, saved to %s' % os.path.basename(self._computed_saved_as))
            self.computedHologramLabel.setToolTip(os.path.abspath(self._computed_saved_as))

    def _setHologramSource(self, path: str) -> None:
        """Mark the current `gui.phasePattern` as coming from the hologram file ``path`` and update the label."""
        self._hologram_source = (self.gui.phasePattern, path)
        self.refreshHologramLabel()

    def addDisplayedHologramLabel(self, label: QtWidgets.QLabel) -> None:
        """Keep ``label`` showing the same 'Displayed hologram' text as `displayedHologramLabel`."""
        self._displayed_hologram_labels.append(label)
        label.setText(self.displayedHologramLabel.text())
        label.setToolTip(self.displayedHologramLabel.toolTip())

    def refreshHologramLabel(self) -> None:
        """Show which hologram is displayed: its file if it is unchanged since it was saved / loaded, else blank or in memory."""
        pattern = getattr(self.gui, 'phasePattern', None)
        if self._hologram_source is not None and self._hologram_source[0] is pattern:
            path = self._hologram_source[1]
            text, tooltip = 'Displayed hologram: %s' % os.path.basename(path), os.path.abspath(path)
        else:
            self._hologram_source = None
            tooltip = ''
            text = HOLOGRAM_BLANK_TEXT if pattern is None or not np.any(pattern) else HOLOGRAM_IN_MEMORY_TEXT
        for label in self._displayed_hologram_labels:
            label.setText(text)
            label.setToolTip(tooltip)

    def pack_traps_dict(self, phasePattern: np.ndarray | None = None) -> None:
        self.gui.traps_dict['traps_pos'] = self.gui.traps_pos
        self.gui.traps_dict['traps_k'] = self.gui.traps_k
        self.gui.traps_dict['theta_traps'] = self.gui.theta_traps
        self.gui.traps_dict['weight_traps'] = self.gui.weight_traps
        self.gui.traps_dict['pupil_traps'] = self.gui.pupil_traps
        self.gui.traps_dict['phasePattern'] = self.gui.phasePattern if phasePattern is None else phasePattern

    def unpack_traps_dict(self) -> None:
        self._clearComputedHologram()
        self.gui.traps_pos = self.gui.traps_dict['traps_pos']
        self.gui.traps_k = self.gui.traps_dict['traps_k']
        self.gui.N_traps = int(self.gui.traps_pos.shape[0])
        self.gui.theta_traps = self.gui.traps_dict['theta_traps']
        self.gui.weight_traps = self.gui.traps_dict['weight_traps']
        self.gui.pupil_traps = self.gui.traps_dict['pupil_traps']
        self.gui.phasePattern = self.gui.traps_dict['phasePattern']

        # Added in v29 to allow recompute the hologram based on updated traps_dict parameters
        if 'computeHologram' in self.gui.traps_dict.keys():
            print(self.gui.traps_dict['computeHologram'])

            if self.gui.traps_dict['computeHologram']:
                ph = self.computeHologram()
                if ph is not None:
                    self.gui.phasePattern = ph
                    self.gui.updatePhase()
