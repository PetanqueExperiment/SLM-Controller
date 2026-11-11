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
from gui.panels.arbitrary_arrays.trap_plot import plotTrapsFile
from gui.panels.arbitrary_arrays.traps import trapArrays
from gui.panels.arbitrary_arrays.tweezer_array import coords_npy_to_traps_pos, rectangular_lattice_coords

if TYPE_CHECKING:
    from gui.main_window import SLMGUI

# Width split of the two panels: trap tools (left) and the combined hologram generator / hologram files panel (right).
ARBITRARY_ARRAYS_WIDTH_FRACTION: float = 0.5
HOLOGRAM_PANEL_WIDTH_FRACTION: float = 1.0 - ARBITRARY_ARRAYS_WIDTH_FRACTION


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
        self._build()
        # Two framed panels side by side: trap tools (left) | hologram generator + hologram files combined (right).
        hologram = QtWidgets.QWidget()
        hologram_lay = QtWidgets.QVBoxLayout(hologram)
        hologram_lay.setContentsMargins(0, 0, 0, 0)
        hologram_lay.setSpacing(gap_px)
        hologram_lay.addWidget(self.generator_root)
        hologram_lay.addWidget(self.files_root)
        hologram_lay.addStretch(1)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(gap_px)
        lay.addWidget(framed_panel(self.root), round(100 * ARBITRARY_ARRAYS_WIDTH_FRACTION))
        lay.addWidget(framed_panel(hologram), round(100 * HOLOGRAM_PANEL_WIDTH_FRACTION))

    def _build(self) -> None:
        """Build the ARBITRARY ARRAYS column (`self.arbitrary_arrays_panel`): patterns, traps, lattice export, algorithms."""
        self.root = pg.LayoutWidget()
        self.arbitrary_arrays_panel = self.root

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
        self.saveArrayCoordinatesBtn = QtWidgets.QPushButton('Save array coordinates')
        self.generateTrapsBtn = QtWidgets.QPushButton('Generate Traps :')
        self.loadTrapPositionsEdt = MyQtLineEdit('rois')
        self.plotTrapsBtn = QtWidgets.QPushButton('Plot traps file')
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
        self.runBtn = QtWidgets.QPushButton('Run Algorithm')
        self.iterationsEdit = MyQtLineEdit(str(self.gui.GS_maxit))
        self.methodCB = QtWidgets.QComboBox()
        for key in getattr(self.gui, 'methodList', {}) or []:
            self.methodCB.addItem(key)

        self.equalize_with_atoms_Btn = QtWidgets.QPushButton('Equalize with atoms')
        self.computeHologram_Btn = QtWidgets.QPushButton('Compute hologram')
        self.devide_slm_area_Btn = QtWidgets.QPushButton('Partial Phase Correction')

        line_idx = 0
        self.arbitrary_arrays_panel.addWidget(self.label_Layout_Arbitrary_Array, row=line_idx, colspan=3)

        line_idx += 1
        self.arbitrary_arrays_panel.addWidget(self.coordsToJsonBtn, row=line_idx, col=0)
        self.arbitrary_arrays_panel.addWidget(self.pastedCoordsEdt, row=line_idx, col=1, colspan=2)
        self.arbitrary_arrays_panel.addWidget(self.pastedCoordsUnitsCB, row=line_idx, col=3)

        line_idx += 1
        self.arbitrary_arrays_panel.addWidget(self.latticeGridLabel, row=line_idx, col=0)

        line_idx += 1
        self.arbitrary_arrays_panel.addWidget(self.latticeRowsLabel, row=line_idx, col=0)
        self.arbitrary_arrays_panel.addWidget(self.latticeRowsSpin, row=line_idx, col=1)
        self.arbitrary_arrays_panel.addWidget(self.latticeColsLabel, row=line_idx, col=2)
        self.arbitrary_arrays_panel.addWidget(self.latticeColsSpin, row=line_idx, col=3)

        line_idx += 1
        self.arbitrary_arrays_panel.addWidget(self.latticeSepXLabel, row=line_idx, col=0)
        self.arbitrary_arrays_panel.addWidget(self.latticeSepXEdt, row=line_idx, col=1)
        self.arbitrary_arrays_panel.addWidget(self.latticeSepYLabel, row=line_idx, col=2)
        self.arbitrary_arrays_panel.addWidget(self.latticeSepYEdt, row=line_idx, col=3)

        line_idx += 1
        self.arbitrary_arrays_panel.addWidget(self.latticeSepZLabel, row=line_idx, col=0)
        self.arbitrary_arrays_panel.addWidget(self.latticeSepZEdt, row=line_idx, col=1)
        self.arbitrary_arrays_panel.addWidget(self.latticeRotLabel, row=line_idx, col=2)
        self.arbitrary_arrays_panel.addWidget(self.latticeRotEdt, row=line_idx, col=3)

        line_idx += 1
        self.arbitrary_arrays_panel.addWidget(self.latticeGapLabel, row=line_idx, col=0)
        self.arbitrary_arrays_panel.addWidget(self.latticeGapEdt, row=line_idx, col=1)

        line_idx += 1
        self.arbitrary_arrays_panel.addWidget(self.saveArrayCoordinatesBtn, row=line_idx, col=0)
        self.arbitrary_arrays_panel.addWidget(self.generateTrapsBtn, row=line_idx, col=1)
        self.arbitrary_arrays_panel.addWidget(self.loadTrapPositionsEdt, row=line_idx, col=2)
        self.arbitrary_arrays_panel.addWidget(self.plotTrapsBtn, row=line_idx, col=3)

        # Hologram generator (gui/hologram_generator.py) and hologram files (gui/hologram_handler.py): own panels below
        self.generator_root = pg.LayoutWidget()
        self.label_hologram_generator = QtWidgets.QLabel(""" -- HOLOGRAM GENERATOR -- """)
        self.label_hologram_generator.setAlignment(QtCore.Qt.AlignCenter)
        self.generator_root.addWidget(self.label_hologram_generator, row=0, col=0, colspan=3)
        self.generator_root.addWidget(self.runBtn, row=1, col=0)
        self.generator_root.addWidget(self.methodCB, row=1, col=1)
        self.generator_root.addWidget(self.iterationsEdit, row=1, col=2)

        self.files_root = pg.LayoutWidget()
        self.label_hologram_files = QtWidgets.QLabel(""" -- HOLOGRAM FILES -- """)
        self.label_hologram_files.setAlignment(QtCore.Qt.AlignCenter)
        self.files_root.addWidget(self.label_hologram_files, row=0, col=0, colspan=3)
        self.files_root.addWidget(self.generateBlankBtn, row=1, col=0)
        self.files_root.addWidget(self.loadBtn, row=1, col=1)
        self.files_root.addWidget(self.saveBtn, row=1, col=2)

        # line_idx += 1
        # self.arbitrary_arrays_panel.addWidget(self.equalize_with_atoms_Btn, row=line_idx, col=0)
        # self.arbitrary_arrays_panel.addWidget(self.computeHologram_Btn, row=line_idx, col=1)
        # self.arbitrary_arrays_panel.addWidget(self.devide_slm_area_Btn, row=line_idx, col=2)

        self.generateBlankBtn.clicked.connect(self.generateBlank)
        self.loadBtn.clicked.connect(self.loadPattern_clicked)
        self.saveBtn.clicked.connect(self.savePattern_clicked)
        self.generateTrapsBtn.clicked.connect(self.generateTraps)
        self.saveArrayCoordinatesBtn.clicked.connect(self.saveArrayCoordinates_clicked)
        self.plotTrapsBtn.clicked.connect(self.plotTrapsFile_clicked)
        self.coordsToJsonBtn.clicked.connect(self.pastedCoordsToJson_clicked)
        self.runBtn.clicked.connect(self.runAlgorithm_clicked)
        # self.equalize_with_atoms_Btn.clicked.connect(self.equalize_with_atoms)
        # self.devide_slm_area_Btn.clicked.connect(self.devide_slm_area)
        # self.computeHologram_Btn.clicked.connect(self.computeHologram_clicked)

    # ==============================================================================
    # Algorithms (Run Algorithm → gui.methodList → WGS)
    # ==============================================================================
    def runAlgorithm_clicked(self) -> None:
        method = str(self.methodCB.currentText())
        self.runAlgorithm(method)

    def runAlgorithm(self, method: str) -> None:
        '''
        This function chooses the method to calculate the phase mask for
        the desired trap positions
        '''
        # Each run starts from the current trap file, with fresh phases and weights
        if not self._load_traps_from_pattern_file(verbose=True):
            print('Run aborted: could not load trap coordinates.')
            return

        try:
            image = self.gui.methodList[method]()
        except:
            print('method not in list')
            raise

        if image is None:
            return

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
        """Load trap coordinates from PyCam ``rois.json`` (see ``_load_traps_from_pattern_file``)."""
        print('Generating traps')
        self._load_traps_from_pattern_file(verbose=True)

    def _traps_ready(self) -> bool:
        """True if trap arrays are initialized from a file or pickle (ready for algorithms)."""
        tp = getattr(self.gui, 'traps_pos', None)
        if not isinstance(tp, np.ndarray) or tp.ndim != 2 or tp.shape[1] != 4:
            return False
        if tp.shape[0] < 1:
            return False
        tk = getattr(self.gui, 'traps_k', None)
        if not isinstance(tk, np.ndarray) or tk.shape != (tp.shape[0], 3):
            return False
        return True

    def _load_traps_from_pattern_file(self, verbose: bool = False) -> bool:
        """
        Load trap coordinates from PyCam ``rois.json`` (path from `loadTrapPositionsEdt`).
        Pixel centers are converted to focal-plane positions using ``ROI_JSON_UM_PER_CAMERA_PIXEL``
        and the same origin as the array generator. Axial ``z`` is set to 0 (not stored in JSON).

        Returns True on success. Does nothing if the file cannot be resolved or has no centers.
        """
        path = rois.resolveRoisPath(self.loadTrapPositionsEdt.text())
        if path is None or not os.path.isfile(path):
            if verbose:
                print('rois.json not found (Tweezer_ROIs folder or full path to .json).')
            return False
        if verbose:
            print('Loading traps from', path)
        coords_m = rois.readTrapCoordsM(path, verbose=verbose)
        if coords_m is None:
            return False
        self.gui.traps_pos = coords_npy_to_traps_pos(coords_m)
        self._finalize_trap_generation()
        return True

    def _ensure_traps_loaded(self) -> bool:
        """
        Ensure `generateTraps`-equivalent state: load from the file field if not already valid
        (e.g. after loading a .pkl with traps, this is a no-op).
        """
        if self._traps_ready():
            return True
        print('Trap coordinates not in memory; loading from trap file field…')
        ok = self._load_traps_from_pattern_file(verbose=False)
        if not ok:
            print('Could not load trap coordinates. Set the rois basename ' '(Tweezer_ROIs or full path to .json).')
        return ok

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
    def saveArrayCoordinates_clicked(self) -> None:
        """Save the rectangular lattice as PyCam / ROIfinder `rois.json`."""
        os.makedirs(TWEEZER_ROIS_DIR, exist_ok=True)
        default_path = os.path.join(TWEEZER_ROIS_DIR, 'rois.json')
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save trap coordinates (rois.json)', default_path, 'PyCam ROIs (*.json)')
        if not path:
            return
        try:
            spacing_cal = float(SYSTEM_MAGNIFICATION)
            sx = float(self.latticeSepXEdt.text()) * spacing_cal
            sy = float(self.latticeSepYEdt.text()) * spacing_cal
            sz = float(self.latticeSepZEdt.text())
            rot = float(self.latticeRotEdt.text())
            gap = float(self.latticeGapEdt.text()) * spacing_cal
        except ValueError:
            print('Invalid lattice parameter (need numbers for Δx, Δy, Δz, rotation, center gap).')
            return
        arr = rectangular_lattice_coords(self.latticeRowsSpin.value(), self.latticeColsSpin.value(), sx, sy, sep_z_um=sz, rotation_deg_z=rot, center_gap_um=gap)

        path = path.strip()
        low = path.lower()
        _parent = os.path.dirname(os.path.abspath(path))
        if _parent:
            os.makedirs(_parent, exist_ok=True)

        if not low.endswith('.json'):
            root, ext = os.path.splitext(path)
            path = root + '.json' if ext else path + '.json'
        centers, side = rois.latticeCenters(arr, sx, sy)
        rois.writeRoisJson(path, centers, side, rot)
        self.loadTrapPositionsEdt.setText(os.path.splitext(os.path.basename(path))[0])
        print('Saved PyCam ROIs:', path, '(%d centers)' % arr.shape[0])

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
        rois.writeRoisJson(path, centers, side, 0.0)
        self.loadTrapPositionsEdt.setText(os.path.splitext(os.path.basename(path))[0])
        print('Saved PyCam ROIs:', path, '(%d centers, roi size %d px)' % (len(centers), side))

    def plotTrapsFile_clicked(self) -> None:
        """Pick one or more ``rois.json`` files and overlay their trap centers (camera pixels) in one window."""
        start = rois.resolveRoisPath(self.loadTrapPositionsEdt.text()) or TWEEZER_ROIS_DIR
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

        self.pack_traps_dict()
        writeTrapsDict(self.gui.file, self.gui.traps_dict)

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
            except Exception as e:
                print('Could not load traps_dict')
                raise e
        # obsolete
        elif name[-3:] in ('npy', 'bmp'):
            self.gui.phasePattern = readLegacyPattern(name)
        elif name == None:
            pass
        else:
            print('Incorrect file type')
            print('You are too stupid!!!!!!')  ### Chew and Sylvain.

        self.gui.updatePhase()

    def pack_traps_dict(self) -> None:
        self.gui.traps_dict['traps_pos'] = self.gui.traps_pos
        self.gui.traps_dict['traps_k'] = self.gui.traps_k
        self.gui.traps_dict['theta_traps'] = self.gui.theta_traps
        self.gui.traps_dict['weight_traps'] = self.gui.weight_traps
        self.gui.traps_dict['pupil_traps'] = self.gui.pupil_traps
        self.gui.traps_dict['phasePattern'] = self.gui.phasePattern

    def unpack_traps_dict(self) -> None:
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
