"""Arbitrary arrays panel (patterns, traps, algorithms, equalization)."""
import json
import os
import pickle
import random as rd
import sys
import time
from importlib import reload

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import pyqtSignal
from pyqtgraph.Qt import QtCore, QtWidgets
from PIL import Image

import Meadowlark_SLM.parameters as param
from MyQtLineEdit.myqtLineEdit import MyQtLineEdit
from gpu_module_v7_3 import slm_to_trap_amp, trap_phase_to_slm
from gui.constants import (
    DIR_SHARED_DETEKTOR,
    LOCAL_DIR,
    MESSAGE_FILE,
    ROI_JSON_UM_PER_CAMERA_PIXEL,
    SLM_DIFFRACTION_EFFICIENCY_HWHM,
    SLM_PATTERNS_DIR,
    SYSTEM_MAGNIFICATION,
    TWEEZER_ROIS_DIR,
    position_to_step_per_pixel,
    position_to_step_per_pixel_for_z,
)
from gui.helpers import parse_pasted_coordinates
from tweezer_array import coords_npy_to_traps_pos, rectangular_lattice_coords

try:
    import trap_equalization_ramsey as trap_equalization
except ImportError:
    trap_equalization = None


def _pump_gui():
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


class Equalization_intensity_thread(QtCore.QThread):
    '''
    This thread continuously look at the file 'message.txt' (MESSAGE_FILE)
    to communicate with the trap camera computer

    When we want to equalize the trap intensities, this computer calculates
    the phase pattern, and the other computer measures the trap intensities

    Two messages can be written on MESSAGE_FILE:

        1) 'run GS' : written by the trap camera computer to indicate that
        the intensities have been measured and written in FILE_INTENSITY
        this computer can now do a new iteration of the GS algorithm

        2) 'measure intensity i' : written by this computer where i is the
        iteration we are doing. when the other computer reads this message,
        it will measure the intensity and then write the message 1)

    In this thread we launch the runNextIter signal as soon as we read 'run GS'
    If the thread instance self.last is True, we launch the finishIter signal

    These two signal are connected to the function iterate_GS and finish_equalization.
    -    The thread is created when the button Equalize traps is pressed
    The thread is destroyed at the end of the function finish_equalization
    '''

    sigNext = pyqtSignal()
    sigLast = pyqtSignal()

    def __init__(self):
        QtCore.QThread.__init__(self)
        self.go_on_reading_message = False
        self.timer = QtCore.QTimer()
        self.last = False

    def __del__(self):
        self.wait()

    def run(self):
        self.keepCheckingFile()

    def keepCheckingFile(self):

        if self.go_on_reading_message:
            #Check the file for instruction
            with open(MESSAGE_FILE, 'r') as message_read:
                text = [x.strip().split('\t') for x in message_read]
            message_read.close()

            #Check instruction
            try:
                instruction = text[0][0]
            except Exception:
                instruction = ''

            if instruction == 'run GS':
                self.go_on_reading_message = False

                if self.last:
                    ## send signal to main thread to finish
                    self.sigLast.emit()
                else:
                    ## send signal to main thread to run the GSI
                    self.sigNext.emit()

        # repeat the loop.
        self.timer.singleShot(0, self.keepCheckingFile)



class ArbitraryArraysPanel(QtWidgets.QWidget):
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
        self.arbitrary_arrays_panel.addWidget(self.generateBlankBtn, row=line_idx, col=0)
        self.arbitrary_arrays_panel.addWidget(self.loadBtn, row=line_idx, col=1)
        self.arbitrary_arrays_panel.addWidget(self.saveBtn, row=line_idx, col=2)

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

        line_idx += 1
        self.arbitrary_arrays_panel.addWidget(self.runBtn, row=line_idx, col=0)
        self.arbitrary_arrays_panel.addWidget(self.methodCB, row=line_idx, col=1)
        self.arbitrary_arrays_panel.addWidget(self.iterationsEdit, row=line_idx, col=2)

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

    def runAlgorithm_clicked(self):
        method = str(self.methodCB.currentText())
        self.runAlgorithm(method)

    def runAlgorithm(self, method):
        '''
        This function chooses the method to calculate the phase mask for
        the desired trap positions
        '''
        try :
            image = self.gui.methodList[method]()
        except :
            print('method not in list')
            raise

        if image is None:
            return

        # Put the phase pattern on the GUI and on the SLM
        self.gui.phasePattern = image
        self.gui.updatePhase()

    def savePattern_clicked(self):
        '''
        Opens the dialog and saves the selected file
        '''
        os.makedirs(SLM_PATTERNS_DIR, exist_ok=True)
        self.gui.file = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Hologram', SLM_PATTERNS_DIR)[0]
        if not self.gui.file:
            return
        self.savePattern(name = self.gui.file)

    def savePattern(self, name = ''):
        '''
        Opens the dialog and saves the selected file
        '''
        if not self._ensure_traps_loaded():
            print('Save pattern aborted: trap coordinates are not available.')
            return

        self.gui.file = name
        if not(self.gui.file[-4:] == '.pkl'):
            self.gui.file += '.pkl'

        self.pack_traps_dict()

        a_file = open(self.gui.file, "wb")
        pickle.dump(self.gui.traps_dict, a_file)
        a_file.close()


    def loadPattern_clicked(self):
        '''
        Open the dialog and gets the selected file
        '''
        os.makedirs(SLM_PATTERNS_DIR, exist_ok=True)
        self.gui.file = QtWidgets.QFileDialog.getOpenFileName(self, 'Load Hologram', SLM_PATTERNS_DIR)[0]
        if not self.gui.file:
            return
        self.loadPattern(name = self.gui.file)

    def loadPattern(self, name = ''):
        print('SLM file')
        print(name)
        self.gui.file = name

        #obsolete
        if name[-3:] == 'npy':
            self.gui.phasePattern = np.load(name)
            self.gui.phasePattern = self.gui.phasePattern[:,:792] #in case that the pattern size is 600x800
        #obsolete
        elif name[-3:] == 'bmp':
            self.gui.phasePattern = np.asarray(Image.open(name), dtype=np.float64)
            self.gui.phasePattern *= 2*np.pi/256
        #current
        elif name[-3:] == 'pkl':
            try:
                a_file = open(self.gui.file, "rb")
                self.gui.traps_dict = pickle.load(a_file)
                self.unpack_traps_dict()
                a_file.close()
            except Exception as e:
                print('Could not load traps_dict')
                raise e
        elif name == None:
            pass
        else:
            print('Incorrect file type')
            print('You are too stupid!!!!!!') ### Chew and Sylvain.

        self.gui.updatePhase()

    def pack_traps_dict(self):
        self.gui.traps_dict['traps_pos'] = self.gui.traps_pos
        self.gui.traps_dict['traps_k'] = self.gui.traps_k
        self.gui.traps_dict['theta_traps'] = self.gui.theta_traps
        self.gui.traps_dict['weight_traps'] = self.gui.weight_traps
        self.gui.traps_dict['pupil_traps'] = self.gui.pupil_traps
        self.gui.traps_dict['phasePattern'] = self.gui.phasePattern

    def unpack_traps_dict(self):
        self.gui.traps_pos      = self.gui.traps_dict['traps_pos']
        self.gui.traps_k        = self.gui.traps_dict['traps_k']
        self.gui.theta_traps    = self.gui.traps_dict['theta_traps']
        self.gui.weight_traps   = self.gui.traps_dict['weight_traps']
        self.gui.pupil_traps    = self.gui.traps_dict['pupil_traps']
        self.gui.phasePattern   = self.gui.traps_dict['phasePattern']

        #Added in v29 to allow recompute the hologram based on updated traps_dict parameters
        if 'computeHologram' in self.gui.traps_dict.keys():
            print(self.gui.traps_dict['computeHologram'])

            if self.gui.traps_dict['computeHologram'] :
                ph = self.computeHologram()
                if ph is not None:
                    self.gui.phasePattern = ph
                    self.gui.updatePhase()



    #==============================================================================
    # Elementary patterns
    #==============================================================================
    def generateBlank(self):
        '''Generates a blank phase mask'''
        self.gui.phasePattern = np.zeros([param.ImgResY, param.ImgResX])
        self.gui.updatePhase()
    def _resolve_rois_json_path(self, base):
        """Resolve path to PyCam ``rois.json`` (``Tweezer_ROIs/`` or an existing file path)."""
        tweezer_dir = os.path.join(LOCAL_DIR, 'Tweezer_ROIs')
        base = (base or '').strip()
        if not base:
            return os.path.join(tweezer_dir, 'rois.json')
        if os.path.isfile(base):
            return base
        cand = os.path.join(tweezer_dir, base)
        if os.path.isfile(cand):
            return cand
        if not base.lower().endswith('.json'):
            p_json = os.path.join(tweezer_dir, base + '.json')
            if os.path.isfile(p_json):
                return p_json
        return None

    def _traps_ready(self):
        """True if trap arrays are initialized from a file or pickle (ready for algorithms)."""
        tp = getattr(self, 'traps_pos', None)
        if not isinstance(tp, np.ndarray) or tp.ndim != 2 or tp.shape[1] != 4:
            return False
        if tp.shape[0] < 1:
            return False
        tk = getattr(self, 'traps_k', None)
        if not isinstance(tk, np.ndarray) or tk.shape != (tp.shape[0], 3):
            return False
        return True

    def _load_traps_from_pattern_file(self, verbose=False):
        """
        Load trap coordinates from PyCam ``rois.json`` (path from `loadTrapPositionsEdt`).
        Pixel centers are converted to focal-plane positions using ``ROI_JSON_UM_PER_CAMERA_PIXEL``
        and the same origin as ``_save_array_rois_to_json``. Axial ``z`` is set to 0 (not stored in JSON).

        Returns True on success. Does nothing if the file cannot be resolved or has no centers.
        """
        path = self._resolve_rois_json_path(self.loadTrapPositionsEdt.text())
        if path is None or not os.path.isfile(path):
            if verbose:
                print('rois.json not found (Tweezer_ROIs folder or full path to .json).')
            return False
        if verbose:
            print('Loading traps from', path)

        with open(path, 'r', encoding='utf-8') as f:
            doc = json.load(f)
        centers = doc.get('centers')
        if not isinstance(centers, list) or len(centers) < 1:
            if verbose:
                print('rois.json has no valid "centers" array.')
            return False

        um_per_px = float(ROI_JSON_UM_PER_CAMERA_PIXEL)
        if um_per_px <= 0:
            um_per_px = 1.0
        cx = 0.5 * float(param.ImgResX)
        cy = 0.5 * float(param.ImgResY)

        rows = []
        for c in centers:
            if not isinstance(c, (list, tuple)) or len(c) < 2:
                continue
            px_x, px_y = float(c[0]), float(c[1])
            x_um = (px_x - cx) * um_per_px
            y_um = (px_y - cy) * um_per_px
            rows.append((x_um * 1e-6, y_um * 1e-6, 0.0))
        if not rows:
            if verbose:
                print('rois.json "centers" produced no valid points.')
            return False

        coords_m = np.asarray(rows, dtype=np.float64)
        self.gui.traps_pos = coords_npy_to_traps_pos(coords_m)
        self._finalize_trap_generation()
        return True

    def _ensure_traps_loaded(self):
        """
        Ensure `generateTraps`-equivalent state: load from the file field if not already valid
        (e.g. after loading a .pkl with traps, this is a no-op).
        """
        if self._traps_ready():
            return True
        print('Trap coordinates not in memory; loading from trap file field…')
        ok = self._load_traps_from_pattern_file(verbose=False)
        if not ok:
            print(
                'Could not load trap coordinates. Set the rois basename '
                '(Tweezer_ROIs or full path to .json).'
            )
        return ok

    def _finalize_trap_generation(self):
        """Build k-vectors, random phases, pupils, and traps_dict from self.gui.traps_pos (meters)."""
        self.gui.N_traps = int(self.gui.traps_pos.shape[0])
        print('Traps = ', self.gui.N_traps)
        self.gui.traps_k = np.copy(self.gui.traps_pos[:, 1:4])
        self.gui.traps_k[:, 0:2] *= position_to_step_per_pixel
        self.gui.traps_k[:, 2] *= position_to_step_per_pixel_for_z

        self.gui.theta_traps = np.random.rand(self.gui.N_traps) * 2 * np.pi
        self.gui.weight_traps = np.ones(self.gui.N_traps)

        # pupilEdt is one number for circular/square, two for rectangular, three for rotated / traps
        txt = (self.gui.pupilEdt.text() or '').strip()
        parts = [p.strip() for p in txt.split(',') if p.strip()]
        if not parts:
            shortaxis, longaxis, θ = 100.0, 100.0, 100.0
        elif len(parts) == 1:
            v = float(parts[0])
            shortaxis, longaxis, θ = v, v, 0.0
        elif len(parts) == 2:
            shortaxis, longaxis, θ = float(parts[0]), float(parts[1]), 0.0
        else:
            shortaxis, longaxis, θ = float(parts[0]), float(parts[1]), float(parts[2])

        longaxis /= param.pslm * 1e3
        shortaxis /= param.pslm * 1e3

        θ *= np.pi / 180

        self.gui.pupil_traps = {
            'pupil_longaxis': np.array([longaxis] * self.gui.N_traps),
            'pupil_shortaxis': np.array([shortaxis] * self.gui.N_traps),
            'pupil_angle': np.array([θ] * self.gui.N_traps),
        }

        self.gui.traps_dict = {
            'traps_pos': self.gui.traps_pos,
            'traps_k': self.gui.traps_k,
            'theta_traps': self.gui.theta_traps,
            'weight_traps': self.gui.weight_traps,
            'pupil_traps': self.gui.pupil_traps,
        }

    def saveArrayCoordinates_clicked(self):
        """Save the rectangular lattice as PyCam / ROIfinder `rois.json`."""
        os.makedirs(TWEEZER_ROIS_DIR, exist_ok=True)
        default_path = os.path.join(TWEEZER_ROIS_DIR, 'rois.json')
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            'Save trap coordinates (rois.json)',
            default_path,
            'PyCam ROIs (*.json)',
        )
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
        arr = rectangular_lattice_coords(
            self.latticeRowsSpin.value(),
            self.latticeColsSpin.value(),
            sx,
            sy,
            sep_z_um=sz,
            rotation_deg_z=rot,
            center_gap_um=gap,
        )

        path = path.strip()
        low = path.lower()
        _parent = os.path.dirname(os.path.abspath(path))
        if _parent:
            os.makedirs(_parent, exist_ok=True)

        if not low.endswith('.json'):
            root, ext = os.path.splitext(path)
            path = root + '.json' if ext else path + '.json'
        self._save_array_rois_to_json(path, arr, sx, sy, rot)
        self.loadTrapPositionsEdt.setText(os.path.splitext(os.path.basename(path))[0])
        print('Saved PyCam ROIs:', path, '(%d centers)' % arr.shape[0])

    def _save_array_rois_to_json(self, path, arr_m, sx_um, sy_um, angle_deg):
        """
        UTF-8 JSON, indent=2. Schema version 2: centers, roi_sizes, x_com, y_com, angle_deg, rescale.

        Array coordinates (m) are mapped to integer camera pixels using ROI_JSON_UM_PER_CAMERA_PIXEL
        and origin at (ImgResX/2, ImgResY/2). Tune the constant for your PyCam calibration.
        """
        um_per_px = float(ROI_JSON_UM_PER_CAMERA_PIXEL)
        if um_per_px <= 0:
            um_per_px = 1.0
        cx = 0.5 * float(param.ImgResX)
        cy = 0.5 * float(param.ImgResY)
        x_um = arr_m[:, 0] * 1e6
        y_um = arr_m[:, 1] * 1e6
        centers = [
            [int(round(x_um[i] / um_per_px + cx)), int(round(y_um[i] / um_per_px + cy))]
            for i in range(x_um.shape[0])
        ]
        side = max(3, int(round(min(float(sx_um), float(sy_um)) / um_per_px)))
        self._write_rois_json(path, centers, side, angle_deg)

    @staticmethod
    def _write_rois_json(path, centers, roi_side_px, angle_deg=0.0):
        """Write a PyCam ``rois.json`` (schema version 2) with pixel ``centers`` (int or float)."""
        doc = {
            'format_version': 2,
            'centers': centers,
            'roi_sizes': [int(roi_side_px)],
            'x_com': 0.0,
            'y_com': 0.0,
            'angle_deg': float(angle_deg),
            'rescale': 1.0,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
            f.write('\n')

    def plotTrapsFile_clicked(self):
        """Pick one or more ``rois.json`` files and overlay their trap centers (camera pixels) in one window."""
        start = self._resolve_rois_json_path(self.loadTrapPositionsEdt.text()) or TWEEZER_ROIS_DIR
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            'Plot trap coordinates',
            start,
            'PyCam ROIs (*.json);;All files (*.*)',
        )
        for path in paths:
            self._plot_traps_file(path)

    def _plot_traps_file(self, path):
        """Add the trap centers of one ``rois.json`` to the trap plot window (created if not open)."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                doc = json.load(f)
            centers = np.asarray(
                [c[:2] for c in doc.get('centers', []) if isinstance(c, (list, tuple)) and len(c) >= 2],
                dtype=np.float64,
            )
        except (OSError, ValueError, TypeError) as e:
            print('Could not read trap file:', path, '-', e)
            return
        if centers.size == 0:
            print('No valid "centers" in', path)
            return

        win = getattr(self, '_trap_plot_window', None)
        if win is None or not win.isVisible():
            win = pg.PlotWidget()
            win.resize(600, 600)
            win.setAspectLocked(True)
            win.invertY(True)
            win.showGrid(x=True, y=True, alpha=0.3)
            win.setLabel('bottom', 'x (px)')
            win.setLabel('left', 'y (px)')
            win.addLegend()
            self.gui._trap_plot_window = win
            self.gui._trap_plot_count = 0

        color = pg.intColor(self.gui._trap_plot_count, hues=8)
        self.gui._trap_plot_count += 1
        name = '%s (%d)' % (os.path.basename(path), centers.shape[0])
        win.plot(
            centers[:, 0], centers[:, 1],
            pen=None, symbol='o', symbolSize=8, symbolBrush=color, symbolPen=None, name=name,
        )
        if centers.shape[0] <= 200:
            for i, (x, y) in enumerate(centers):
                txt = pg.TextItem(str(i), color=color, anchor=(0, 1))
                txt.setPos(x, y)
                win.addItem(txt)
        win.setWindowTitle(
            'Traps: %s' % name if self.gui._trap_plot_count == 1 else 'Traps: %d files' % self.gui._trap_plot_count
        )
        win.show()
        win.raise_()
        win.activateWindow()
        print('Plotted %d traps from %s' % (centers.shape[0], path))

    def pastedCoordsToJson_clicked(self):
        """Convert the pasted coordinate list / np.array into a PyCam ``rois.json`` chosen via a save dialog."""
        try:
            coords = parse_pasted_coordinates(self.pastedCoordsEdt.text())
        except ValueError as e:
            print('Invalid pasted coordinates:', e)
            return
        if coords.shape[1] == 3 and np.any(coords[:, 2] != 0):
            print('Note: z column ignored (rois.json stores only x, y).')

        xy = coords[:, :2]
        if self.pastedCoordsUnitsCB.currentIndex() == 1:
            um_per_px = float(ROI_JSON_UM_PER_CAMERA_PIXEL)
            if um_per_px <= 0:
                um_per_px = 1.0
            center = np.array([0.5 * float(param.ImgResX), 0.5 * float(param.ImgResY)])
            xy = xy * float(SYSTEM_MAGNIFICATION) / um_per_px + center
        centers = [[round(float(x), 4), round(float(y), 4)] for x, y in xy]

        if xy.shape[0] > 1:
            d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=-1))
            np.fill_diagonal(d, np.inf)
            side = max(3, int(round(d.min())))
        else:
            side = 3

        os.makedirs(TWEEZER_ROIS_DIR, exist_ok=True)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            'Save pasted coordinates (rois.json)',
            os.path.join(TWEEZER_ROIS_DIR, 'rois_custom.json'),
            'PyCam ROIs (*.json)',
        )
        if not path:
            return
        path = path.strip()
        if not path.lower().endswith('.json'):
            path = os.path.splitext(path)[0] + '.json'
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._write_rois_json(path, centers, side, 0.0)
        self.loadTrapPositionsEdt.setText(os.path.splitext(os.path.basename(path))[0])
        print('Saved PyCam ROIs:', path, '(%d centers, roi size %d px)' % (len(centers), side))

    def generateTraps(self):
        """Load trap coordinates from PyCam ``rois.json`` (see ``_load_traps_from_pattern_file``)."""
        print('Generating traps')
        self._load_traps_from_pattern_file(verbose=True)

    # =============================================================================
    # Trap equalization using atom signal
    # ============================================================================
    def equalize_with_atoms(self):
        print('equalize with atoms')
        signal = np.load(DIR_SHARED_DETEKTOR+'/map.npy')
        # signal += 1-np.mean(signal)
        print('current stdev: %.1f'%(np.std(signal)*100))

        self.gui.equalize_gain = 1.#float(self.equalize_gain_le.text())

        #Update hologram
        self.gui.phasePattern = self.GS_intensity(signal,self.gui.equalize_gain)
        if self.gui.phasePattern is None:
            return

        self.gui.updatePhase()

        #save hologram information
        timestr = time.strftime("%Y%m%d_%H%M%S")
        self.savePattern(name = LOCAL_DIR+'\\holograms\\equalized_with_atoms_'+timestr+'.pkl')

    def devide_slm_area(self, change = True, signal = None ):

        """
        With Martin we change only part of the hologram to gain more precission in equalizing the trap intensity.
        """


        if signal== None:
            signal = np.load(DIR_SHARED_DETEKTOR+'/map.npy')

        else:
            signal = self.gui.map


        # signal += 1-np.mean(signal)
        if change :
            print('current stdev: %.1f'%(np.std(signal)*100))
        else :
            print('no change of initial hologram')

        self.gui.equalize_gain = 1.#float(self.equalize_gain_le.text())

        #Update hologram
        phasePattern_orig = self.gui.phasePattern
        phasePattern_cal = self.GS_intensity(signal,self.gui.equalize_gain)
        if phasePattern_cal is None:
            return

        if np.std(signal) < 0.6 :
            k = rd.randint(0,2)
            l = rd.randint(0,2)

            y_len = 792 #param.ImgResX
            x_len = 600 #param.ImgResY

        # type(param.ImgResX)
            d_phi = phasePattern_cal - phasePattern_orig

            phasePattern_cal += 3*d_phi
            phasePattern_orig[k*(x_len//2):(k+1)*(x_len//2), l*(y_len//2): (l+1)*(y_len//2)] = phasePattern_cal[k*(x_len//2):(k+1)*(x_len//2), l*(y_len//2): (l+1)*(y_len//2)]
        else :
            phasePattern_orig = phasePattern_cal

        self.gui.phasePattern = phasePattern_orig

        self.gui.updatePhase()

        #save hologram information
        timestr = time.strftime("%Y%m%d_%H%M%S")

        hologram_save_file = LOCAL_DIR+'\\holograms\\equalized_with_atoms_'+timestr+'.pkl'
        self.savePattern(name = LOCAL_DIR+'\\holograms\\equalized_with_atoms_'+timestr+'.pkl')

        print('done partial phase correction, saved at : ' + hologram_save_file)

    def devide_slm_area_loop(self, loop = None):

        global trap_equalization
        trap_equalization = reload(trap_equalization)
        #best_hologram =
        folder_dir = self.gui.parent.experiment_directory

        change = True
        print('loop num', loop)

        if loop == 0:
            pass
        else:
            loop = int(loop)
            loop = int(loop-1)
            if loop !=0.0 :
                current_map = self.gui.map

            self.gui.map = trap_equalization.cal(folder_dir, loop = loop)

            std = np.std(self.gui.map) #martin
            print(std)
            if std > self.gui.std and (self.gui.std * 100) < 0.5 : #martin
                change = False
                self.gui.map = current_map
                self.gui.phasePattern = self.gui.best_hologram

            else : #martin
                self.gui.std = std #martin
                self.gui.best_hologram = self.gui.phasePattern #martin
                print('new point') #martin

            self.devide_slm_area(change, signal = 1)



    ###########################################################################
    # Algorithms for phase calculation
    ###########################################################################
    def computeHologram_clicked(self):
        print('Computed hologram')
        phi = self.computeHologram()
        if phi is None:
            return
        self.gui.phasePattern = phi
        self.gui.updatePhase()

    def computeHologram(self):
        if not self._ensure_traps_loaded():
            return None
        # Calculate phase pattern with homemade gpu function
        phi = trap_phase_to_slm(
                self.gui.traps_k,
                self.gui.theta_traps,
                self.gui.weight_traps,
                self.gui.pupil_traps,
                shape = self.gui.SLM_shape,
                )

        return phi

    def randomMask(self):
        '''
        random mask (RM)
        '''
        if not self._ensure_traps_loaded():
            return None
        randmat = np.ceil(np.random.rand(param.ImgResY,param.ImgResX)*self.gui.N_traps).astype(int)

        phi = np.zeros((param.ImgResY,param.ImgResX))

        for i in range(param.ImgResY):
            for j in range(param.ImgResX):
                random_trap = randmat[i,j]-1

                k_x, k_y = self.gui.traps_k[random_trap]
                phi[i,j] = self.gui.i_slm*k_x + self.gui.y_slm*k_y

        return phi

    def superposition(self):
        '''
        superposition of gratings and lenses (S)
        '''
        if not self._ensure_traps_loaded():
            return None
        self.gui.theta_traps = np.zeros(self.gui.N_traps)
        self.gui.weight_traps = np.ones(self.gui.N_traps)

        phi = self.computeHologram()

        return phi

    def randomSuperposition(self):
        '''
        Superposition of gratings with random phases.
        '''
        if not self._ensure_traps_loaded():
            return None
        self.gui.theta_traps = np.random.rand(self.gui.N_traps)*2*np.pi
        self.gui.weight_traps = np.ones(self.gui.N_traps)

        phi = self.computeHologram()

        return phi

    def GS(self):
        '''
        Gerchberg−Saxton (GS). Optimize the phase of each grating.
        '''
        print('Starting Gerchberg-Saxton calculation')
        if not self._ensure_traps_loaded():
            return None
        GS_maxit = int(self.iterationsEdit.text())

        #Get initial random phases
        self.gui.theta_traps = np.random.rand(self.gui.N_traps)*2*np.pi
        self.gui.weight_traps = np.ones(self.gui.N_traps)

        #Calculate phase pattern with homemade gpu function
        phi = self.computeHologram()
        if phi is None:
            return None

        #(a) Calculate the intensity at trap position
        traps_amplitude = 1./param.ImgResY/param.ImgResX*slm_to_trap_amp(phi, self.gui.traps_k)

        trap_intensities = np.real(traps_amplitude*np.conj(traps_amplitude))
        efficiency = np.sum(trap_intensities)
        uniformity = np.std(trap_intensities)/np.mean(trap_intensities)

        print('1st calculation:')
        print('Efficiency std = %.1f'%(efficiency*100))
        print('Intensity std = %.1f'%(uniformity*100))
        _pump_gui()

        #Start the iterations5
        for k in range(GS_maxit):
            print ('    GS ' + str(k+1) +'/' + str(GS_maxit))

            #(a) Calculate the phase of each trap
            if k == 0: #(b0) if first time: start with random phases
                self.gui.theta_traps = np.random.rand(self.gui.N_traps)*2*np.pi
                self.gui.weight_traps = np.ones(self.gui.N_traps)
            else: #(b0)else  apply the GS technique
                self.gui.theta_traps = np.angle(traps_amplitude)

            #(c) Calculate the new slm phase
            phi = self.computeHologram()
            if phi is None:
                return None

            #(a) Calculate the intensity at trap position
            traps_amplitude = 1./param.ImgResY/param.ImgResX*slm_to_trap_amp(phi, self.gui.traps_k)

            trap_intensities = np.real(traps_amplitude*np.conj(traps_amplitude))
            efficiency = np.sum(trap_intensities)
            uniformity = np.std(trap_intensities)/np.mean(trap_intensities)
            print('Efficiency std = %.1f'%(efficiency*100))
            print('Intensity std = %.1f'%(uniformity*100))
            _pump_gui()

        return phi

    def WGS(self, correct_diffraction_efficiency = False):
        '''
        Weighted Gerchberg−Saxton (GS). Otpimize the phase and weigth of each grating.
        '''
        print('Starting weighted Gerchberg-Saxton calculation')
        _pump_gui()
        if not self._ensure_traps_loaded():
            return None

        if correct_diffraction_efficiency:
            # Calculate the radial displacement of each traps
            traps_displacements = ( (self.gui.traps_pos[:,1]+self.gui.dx*1e-6)**2 + (self.gui.traps_pos[:,2]+self.gui.dy*1e-6)**2 )**0.5

            # The SLM diffraction efficiency depends on the displacement of the traps
            SLM_diffraction_efficiency = 1. - 0.5*traps_displacements/SLM_DIFFRACTION_EFFICIENCY_HWHM

            # Correct for the limited diffraction efficiency
            target_intensities = 1./SLM_diffraction_efficiency
            target_amplitudes = target_intensities**0.5
            target_amplitudes_norm = target_amplitudes/np.mean(target_amplitudes)

        else:
            target_intensities = np.ones(self.gui.N_traps)
            target_amplitudes_norm = np.ones(self.gui.N_traps)

        # Read the number of iterations
        GSW_maxit = int(self.iterationsEdit.text())

        # Calculate phase pattern with homemade gpu function
        phi = self.computeHologram()
        if phi is None:
            return None

        # Measure the trap to trap deviation of intensity
        traps_amplitude         = 1./param.ImgResY/param.ImgResX*slm_to_trap_amp(phi, self.gui.traps_k)
        trap_intensities        = np.real(traps_amplitude*np.conj(traps_amplitude))
        deviation_from_target   = trap_intensities/target_intensities
        rel_dev                 = deviation_from_target**0.5/np.mean(deviation_from_target**0.5)
        efficiency              = np.sum(trap_intensities)
        uniformity              = np.std(deviation_from_target)/np.mean(deviation_from_target)

        print('Initial calculation:')
        print('Efficiency std = %.1f'%(efficiency*100))
        print('Intensity std = %.1f'%(uniformity*100))
        _pump_gui()

        # Start the iterations
        for k in range(GSW_maxit):
            print('    WGS %d/%d' % (k + 1, GSW_maxit))
            # Calculate new phases and weights
            self.gui.theta_traps  = np.angle(traps_amplitude)
            self.gui.weight_traps *= 1. / rel_dev

            # Calculate new hologram
            phi = self.computeHologram()
            if phi is None:
                return None

            # Calculate new trap intensities
            traps_amplitude         = 1./param.ImgResY/param.ImgResX*slm_to_trap_amp(phi, self.gui.traps_k)
            trap_intensities        = np.real(traps_amplitude*np.conj(traps_amplitude))
            deviation_from_target   = trap_intensities/target_intensities
            rel_dev                 = deviation_from_target**0.5/np.mean(deviation_from_target**0.5)
            efficiency              = np.sum(trap_intensities)
            uniformity              = np.std(deviation_from_target)/np.mean(deviation_from_target)
            print('Efficiency std = %.1f'%(efficiency*100))
            print('Intensity std = %.1f'%(uniformity*100))
            _pump_gui()

        return phi

    def GS_intensity(self,measured_intensities,equalize_gain = 1.):
        '''
        Adjust the weight of each trap according to the measured intensities
        Keep the same phase for each trap
        '''
        if not self._ensure_traps_loaded():
            return None
        self.gui.weight_traps *= measured_intensities**-0.5

        phi = self.computeHologram()
        if phi is None:
            return None
        print(phi)

        return phi


    def GAA(self):
        '''
        Generalized adaptive additive
        '''
        if not self._ensure_traps_loaded():
            return None
        GAA_maxit = int(self.iterationsEdit.text())
        Vtraps_it = np.zeros(self.gui.N_traps,dtype=complex)

        phi = self.randomSuperposition()
        if phi is None:
            return None
        print('SR guess created')

        #Measure the trap to trap deviation of intensity
        for k in range(self.gui.N_traps):
            Vtraps_it[k] = 1/param.ImgResY/param.ImgResX*np.sum(np.sum(np.exp(1j *(phi-self.gui.delta_traps[:,:,k])), axis=0), axis=0)
        trap_intensities = np.real(Vtraps_it*np.conj(Vtraps_it))
        efficiency = np.sum(trap_intensities)
        uniformity = np.std(trap_intensities)/np.mean(trap_intensities)
        print('Efficiency std = %.1f'%(efficiency*100))
        print('Intensity std = %.1f'%(uniformity*100))

        for k in range(GAA_maxit):

            print ('Iteration ' + str(k+1) +'/' + str(GAA_maxit))
            Vtraps_it_mod = np.sqrt(Vtraps_it * np.conj(Vtraps_it))
            phi = np.angle(np.sum(
                    np.exp(1j*self.gui.delta_traps)*Vtraps_it/Vtraps_it_mod *(1-self.gui.GAA_xi+self.gui.GAA_xi/Vtraps_it_mod),
                    axis=2
                    ))

            for j in range(self.gui.N_traps):
                Vtraps_it[j] = 1/param.ImgResY/param.ImgResX*np.sum(np.sum(np.exp(1j * (phi-self.gui.delta_traps[:,:,j])),axis=0),axis=0)
            trap_intensities = Vtraps_it*np.conj(Vtraps_it)
            efficiency = np.real(np.sum(trap_intensities))
            uniformity = np.std(trap_intensities)/np.mean(trap_intensities)
            print('Efficiency std = %.1f'%(efficiency*100))
            print('Intensity std = %.1f'%(uniformity*100))

        return phi

