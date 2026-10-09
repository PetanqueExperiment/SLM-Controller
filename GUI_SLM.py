import sys
import os

# Allow `import slmpy` / `import zernike` when running this file as a script (not as a package).
_SLM_SOFTWARE_DIR = os.path.dirname(os.path.abspath(__file__))
if _SLM_SOFTWARE_DIR not in sys.path:
    sys.path.insert(0, _SLM_SOFTWARE_DIR)

import numpy as np
import time
import time
import json
import pickle
import random as rd

# GUI (pyqtgraph 0.14+ may otherwise pick PyQt6; this project uses PyQt5 throughout)
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")
from PyQt5.QtCore import pyqtSignal
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph.dockarea as dockarea


class _StreamToPlainText(QtCore.QObject):
    """Mirror writes to a QPlainTextEdit while keeping the original stream (e.g. terminal). Thread-safe."""

    _append = pyqtSignal(str)

    def __init__(self, original_stream, plain_widget, parent=None):
        super().__init__(parent)
        self._orig = original_stream
        self._plain = plain_widget
        self._append.connect(self._append_slot)
        self.encoding = getattr(original_stream, "encoding", None) or "utf-8"

    def _append_slot(self, text):
        self._plain.moveCursor(QtGui.QTextCursor.End)
        self._plain.insertPlainText(text)
        sb = self._plain.verticalScrollBar()
        sb.setValue(sb.maximum())

    def write(self, s):
        if not isinstance(s, str):
            s = str(s)
        try:
            self._orig.write(s)
        except Exception:
            pass
        try:
            self._orig.flush()
        except Exception:
            pass
        if s:
            self._append.emit(s)
        return len(s)

    def flush(self):
        try:
            self._orig.flush()
        except Exception:
            pass

    def isatty(self):
        return False

import matplotlib.pyplot as plt

from MyQtLineEdit.myqtLineEdit import MyQtLineEdit
from SLM_device import SLMDevice

from PIL import Image

try:
    from zernike import __zernikepolar__, __zernikepolynomials__
except ImportError:
    if not __package__:
        raise
    from .zernike import __zernikepolar__, __zernikepolynomials__

try:
    import trap_equalization_ramsey as trap_equalization
except ImportError:
    if not __package__:
        raise
    from . import trap_equalization_ramsey as trap_equalization
from importlib import reload

import Meadowlark_SLM.parameters as param
from gpu_module_v7_3 import slm_to_trap_amp, trap_phase_to_slm
from trap_lattice import coords_npy_to_traps_pos, rectangular_lattice_coords
from temperature_log import DailyTemperatureLog

PUPIL_VERTICAL_OFFSET = 0

N_Zernike = 15
SLM_DIFFRACTION_EFFICIENCY_HWHM = 90e-6
position_to_step_per_pixel = 2*np.pi/param.lamb*(param.M/param.f)*param.pslm
position_to_step_per_pixel_for_z = (np.pi)/ param.lamb  *  (param.M/param.f)**2*(param.pslm**2)

LOCAL_DIR = _SLM_SOFTWARE_DIR
SLM_PATTERNS_DIR = os.path.join(LOCAL_DIR, "SLM_patterns")
ZERNIKE_CORRECTIONS_DIR = os.path.join(LOCAL_DIR, "Zernike_corrections")
TWEEZER_ROIS_DIR = os.path.join(LOCAL_DIR, "Tweezer_ROIs")
ZERNIKE_DEFAULT_COEFFS_JSON = "zernike_coefficients_default.json"

# Optional measured Shack–Hartmann phase correction (radians), shape (ImgResY, ImgResX). If missing, SH map is zeros.
SH_CORRECTION_NPY = os.path.join(LOCAL_DIR, 'SH_correction.npy')

DIR_SHARED_DETEKTOR = 'Y:\\Public'
MESSAGE_FILE = DIR_SHARED_DETEKTOR+'\\SYNC_SLM_CAM.txt'
FILE_INTENSITY = DIR_SHARED_DETEKTOR + '\\file_intensity.dat'

DEFAULT_SLM = "Meadowlark SLM"

# PyCam / ROIfinder rois.json: lattice positions (µm in focal plane) → camera pixels [x, y].
# Centers are placed with the lattice origin at (ImgResX/2, ImgResY/2) in pixel space.
# Tune this to match your camera calibration (micrometers per pixel at the sample plane).
ROI_JSON_UM_PER_CAMERA_PIXEL = 1.0

# Calibration for the system units (µm to phase slope)
SYSTEM_MAGNIFICATION = 1.5

def _load_sh_correction_phase_file(ny, nx):
    """Load optional SH phase map in radians; shape must match SLM. Otherwise return zeros."""
    out = np.zeros((ny, nx), dtype=np.float64)
    if not os.path.isfile(SH_CORRECTION_NPY):
        return out
    arr = np.load(SH_CORRECTION_NPY)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.shape == (ny, nx):
        return arr
    print(
        'SH_correction.npy shape',
        arr.shape,
        'does not match SLM (%d, %d); Shack–Hartmann correction disabled until fixed.' % (ny, nx),
    )
    return out


def framed_panel(inner_widget):
    """Wrap a `pg.LayoutWidget` (or any QWidget) in a bordered frame so regions are visually distinct."""
    frame = QtWidgets.QFrame()
    frame.setObjectName("slm_panel_frame")
    frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
    frame.setFrameShadow(QtWidgets.QFrame.Plain)
    frame.setLineWidth(1)
    frame.setStyleSheet(
        "#slm_panel_frame { border: 1px solid #8a8a8a; border-radius: 4px; background: transparent; }"
    )
    lay = QtWidgets.QVBoxLayout(frame)
    lay.setContentsMargins(6, 6, 6, 6)
    lay.setSpacing(0)
    lay.addWidget(inner_widget)
    return frame


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
            try :
                instruction = text[0][0]
            except :
                instruction = ''

            if instruction == 'run GS':
                self.go_on_reading_message = False

                if self.last :
                    ## send signal to main thread to finish
                    self.sigLast.emit()
                else :
                    ## send signal to main thread to run the GSI
                    self.sigNext.emit()

        # repeat the loop.
        self.timer.singleShot(0, self.keepCheckingFile)

class SLMGUI(QtWidgets.QMainWindow):

    def __init__(self, parent=None, selected_SLM=DEFAULT_SLM):
        """ In the constructor we're doing everything to get our application
            started, which is basically constructing a basic QApplication by
            its __init__ method, then adding our widgets and finally starting
            the exec_loop."""
        super(SLMGUI, self).__init__()

        if parent:
            self.parent = parent

        self.selected_SLM = selected_SLM
        self.remote_control = {
                'SLM1 x': self.updateHorizontalGrating,
                'SLM1 y': self.updateVerticalGrating,
                'SLM1 pupil 1st axis': self.apply_pupil_longaxis_byatom_loop,
                'SLM1 pupil 2st axis': self.apply_pupil_shortaxis_byatom_loop,
                'SLM1: defocus': self.updateAberration_defocus,
                'SLM1: asti0': self.updateAberration_asti0,
                'SLM1: asti45': self.updateAberration_asti45,
                'SLM1: coma0': self.updateAberration_coma0,
                'SLM1: coma90': self.updateAberration_coma90,
                'SLM1: spherical': self.updateAberration_spherical,
                'SLM1: TM01 spacing': self.createTEM01trap,
                'SLM1: TM01 angle(degree)': self.createTEM01trap_angle,
                'SLM1: trefoil0': self.updateAberration_trefoil0,
                'SLM1: trefoil90': self.updateAberration_trefoil90,
                'SLM1: asti2 0': self.updateAberration_asti2_0,
                # 'SLM1: asti2 45': self.updateAberration_asti2_45,
                # 'SLM1: coma2 0': self.updateAberration_coma2_0,
                # 'SLM1: coma2 90': self.updateAberration_coma2_90,
                # 'SLM1: second sph': self.updateAberration_second_sph,
                # 'SLM1: intens. equa.': self.devide_slm_area_loop,
                }


        #a dictionnary that contains the state of the device (frequencies, trigger, power ...)
        self.state_dict = {}

        self.initUI()
        self.initSLM()

        self.last_sent_plot.installEventFilter(self)
        self._start_slm_temperature_monitor()

        self.std = 10
        self.best_hologram = np.zeros((600,792))

    def state_to_dict(self):
        save_list = [
            'dx',
            'dy',
            'aberration_values',
            'file',
            'spacing',
            'theta_deg',
            'r_pi'
            ]

        for instance_name in save_list:
            self.state_dict[instance_name] = self.__dict__[instance_name]

        self.state_dict['TEM01_check'] = self.tem01ChkBx.isChecked()
        self.state_dict['SO_check'] = self.SOChkBx.isChecked()
        self.state_dict['LG_check'] = self.LGChkBx.isChecked()

    def dict_to_state(self, memory_dict):
        try:
            self.createGratingPattern(dx = memory_dict['dx'], dy = memory_dict['dy'])
            self.updateAberration(aber_list = memory_dict['aberration_values'])
            self.loadPattern(name = memory_dict['file'])

            self.tem01spacingEdt.setText(str(memory_dict['spacing']))
            self.tem01ThetaEdt.setText(str(memory_dict['theta_deg']))
            self.tem01ChkBx.setChecked(memory_dict['TEM01_check'])
            self.SOrpiEdt.setText(str(memory_dict['r_pi']))
            self.SOChkBx.setChecked(memory_dict['SO_check'])
            self.LGChkBx.setChecked(memory_dict['LG_check'])
        except Exception as e:
            print(e)

    def initSLM(self):
        '''
        It initializes the SLM device and pixel grids for the selected hardware.
        To run it on a different computer or with a different arrangement,
        adjust paths and parameters as needed.
        '''

        self.SLM = SLMDevice(self.selected_SLM)

        Nx, Ny = self.SLM.getSize()

        self.SLM_shape = (Ny, Nx)

        # Initialize SLM pixel index arrays
        self.X, self.Y = np.meshgrid(
            (np.arange(0, Nx, 1) - Nx / 2) * param.pslm,
            (np.arange(0, Ny, 1) - Ny / 2) * param.pslm,
            )

        self.i_slm, self.j_slm = np.meshgrid(
            np.arange(Nx),
            np.arange(Ny)
            )


        # SLM parameters
        self.deltax = 1
        self.deltay = 1
        self.epsilon = 0
        self.param_P = self.SLM.parameters.P # SLM 2 pi phase shift corresponds to self.param_P 8 bits value


        # Initialize grating, Fresnel and arbitrary phase masks
        self.phasePattern = np.zeros([param.ImgResY, param.ImgResX])
        self.im_grating = np.zeros([param.ImgResY, param.ImgResX])
        self.fresnelPhase = np.zeros([param.ImgResY, param.ImgResX])

        self.zernike_r = (self.X**2+self.Y**2)**0.5
        self.zernike_theta = np.arctan2(self.X,self.Y)

        self.zernike_pupil_radius = param.max_radius
        self.zernike_r = (self.X**2+self.Y**2)**0.5/self.zernike_pupil_radius

        self.zernike_coefficients = np.zeros(32).tolist()

        # Zernike / aberration control panel sum
        self.im_zernike = np.zeros([param.ImgResY, param.ImgResX], dtype=np.float64)

        self.zernike_polynomials = np.asarray(__zernikepolynomials__(self.zernike_r,self.zernike_theta))
        self.zernike_polynomials[:,self.zernike_r>1] = 0 #Zernike polynom not correct beyond a unit pupil
        self.zernike_polynomials *= 2*np.pi

        # Display phase mask
        # self.SLM = slmpy.SLMdisplay(self._name)
        self.createGratingPattern(self.dx,self.dy)
        self.updatePhase()

    def initUI(self):

        #Position
        self.dx = 0
        self.dy = 0

        #Aberrations
        self.aberration_label_list = [
                'tip',
                'tilt',
                'defocus',
                'asti 0',
                'asti 45',
                'coma 0',
                'coma 90',
                'primary spherical',
                'trefoil 0',
                'trefoil 90',
                'asti 2nd 0',
                'asti 2nd 45',
                'coma 2nd 0',
                'coma 2nd 90',
                'secondary spherical'
                ]
        self.aberration_values = np.zeros(len(self.aberration_label_list))

        #Traps
        self.N_traps = 30
        self.traps_dict = {}
        self.traps_pos = [[]]
        self.traps_k = [[]]
        self.theta_traps = []
        self.weigth_traps = []
        self.k_dump = np.array([40,0])*1e-6*position_to_step_per_pixel

        #TEM01
        self.spacing = 5.
        self.theta_deg = -4.

        #Superoscillation (SO)
        self.r_pi = 0.5

        #Trap equalizations
        self.theta_traps_list = []
        self.weight_traps_list = []

        #saved hologram
        self.file = ''

        #Methods only the GS and GSI method are done on the GPU for now
        #We don't really use the other methods
        self.methodList = {
            # 'GS' : self.GS,
            # 'WGS' : lambda: self.WGS(correct_diffraction_efficiency = False),
            'WGS SLMcorr': lambda: self.WGS(correct_diffraction_efficiency = True),
            # 'GAA' : self.GAA,
            # 'RM' : self.randomMask,
            # 'S' : self.superposition,
            # 'SR': self.randomSuperposition,
            # 'LG': self.generateLGMask,
            }

        self.equalize_iter = 5
        self.equalize_gain = 1
        self.GS_maxit = 5 #maximum number of iterations for GS algorithm
        self.GAA_xi = 0.5 #parameter xi for GAA algorithm

        #GUI
        self.area = dockarea.DockArea()
        self.setCentralWidget(self.area)
        self.resize(1200, 720)
        self.setWindowTitle('SLM Controller Petanque')
        self.createDocks()
        self._install_gui_stdio()

    def eventFilter(self, watched, event):
        """Keep the last-sent SLM preview scaled to the full frame when the plot is resized."""
        if watched is getattr(self, "last_sent_plot", None) and event.type() == QtCore.QEvent.Resize:
            QtCore.QTimer.singleShot(0, self._fit_last_sent_view_to_image)
        return super().eventFilter(watched, event)

    def createDocks(self):
        self.d1 = dockarea.Dock("All", size=(700, 700))
        self.area.addDock(self.d1, 'left')
        self.d1.hideTitleBar()

        self.global_layout = pg.LayoutWidget()

        self._build_elementary_control_panel()
        self.global_layout.addWidget(framed_panel(self.w6), row=0, col=0)

        self._build_advanced_control_panel()
        self.global_layout.addWidget(framed_panel(self.Advanced_control_layout), row=1, col=0)

        self._build_aberration_control_panel()
        self.global_layout.addWidget(
            framed_panel(self.Abberation_control_layout), row=0, col=1, rowspan=2
        )

        self._build_arbitrary_arrays_panel()
        self.global_layout.addWidget(framed_panel(self.arbitrary_arrays_panel), row=0, col=2)

        self._build_last_sent_preview_panel()
        self.global_layout.addWidget(framed_panel(self.last_sent_preview_layout), row=1, col=2)

        self.d1.addWidget(self.global_layout, row=0, col=0)

        self._build_output_dock()

    def _build_output_dock(self):
        """Dock showing print() / stdout / stderr (see _install_gui_stdio)."""
        self.d_output = dockarea.Dock("Output", size=(800, 160))
        out_wrap = pg.LayoutWidget()
        self.btn_clear_output_log = QtWidgets.QPushButton("Clear log")
        self.output_log = QtWidgets.QPlainTextEdit()
        self.output_log.setReadOnly(True)
        try:
            self.output_log.setMaximumBlockCount(8000)
        except Exception:
            pass
        _mono = QtGui.QFont("Consolas")
        if not _mono.exactMatch():
            _mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.output_log.setFont(_mono)
        self.btn_clear_output_log.clicked.connect(self.output_log.clear)
        out_wrap.addWidget(self.btn_clear_output_log, row=0, col=0)
        out_wrap.addWidget(self.output_log, row=1, col=0)
        self.d_output.addWidget(out_wrap)
        self.area.addDock(self.d_output, "bottom")

    def _install_gui_stdio(self):
        """Show print() and traceback text in the Output dock as well as the terminal."""
        self._stdout_prev = sys.stdout
        self._stderr_prev = sys.stderr
        self._stdout_tee = _StreamToPlainText(self._stdout_prev, self.output_log, parent=self)
        self._stderr_tee = _StreamToPlainText(self._stderr_prev, self.output_log, parent=self)
        sys.stdout = self._stdout_tee
        sys.stderr = self._stderr_tee

    def _restore_stdio(self):
        if getattr(self, "_stdout_prev", None) is not None:
            sys.stdout = self._stdout_prev
        if getattr(self, "_stderr_prev", None) is not None:
            sys.stderr = self._stderr_prev

    def _build_elementary_control_panel(self):
        """Grating / Fresnel / wavelength column (`self.w6`)."""
        self.w6 = pg.LayoutWidget()

        self.label_w6 = QtWidgets.QLabel(""" -- ELEMENTARY CONTROL -- """)
        self.label_w6.setAlignment(QtCore.Qt.AlignCenter)
        self.moveLabel = QtWidgets.QLabel('GRATING')
        self.moveLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.upBtn = QtWidgets.QPushButton('UP')
        self.downBtn = QtWidgets.QPushButton('DOWN')
        self.leftBtn = QtWidgets.QPushButton('LEFT')
        self.rightBtn = QtWidgets.QPushButton('RIGHT')
        self.stepGratingEdit = MyQtLineEdit('1')
        self.stepGratingEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.positionLabel = QtWidgets.QLabel('Grating shift (in um)')
        self.position_xEdt = MyQtLineEdit(str(self.dx))
        self.position_yEdt = MyQtLineEdit(str(self.dy))
        self.fresnelLabel = QtWidgets.QLabel('FRESNEL LENS')
        self.fresnelLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.fresnelEpsilonLabel = QtWidgets.QLabel('Defocus')
        self.fresnelEpsilonEdt = MyQtLineEdit('0')
        self.paramPLabel = QtWidgets.QLabel('P parameter')
        self.paramPEdt = MyQtLineEdit(str(param.P))
        self.paramWavelengthLabel = QtWidgets.QLabel('Wavelength (nm)')
        self.paramWavelengthEdit = MyQtLineEdit('%.0f'%(param.lamb*1e9))

        self.w6.addWidget(self.label_w6, row=0, colspan = 4)
        self.w6.addWidget(self.moveLabel, row=10, col=0)
        self.w6.addWidget(self.upBtn, row = 9, col=2)
        self.w6.addWidget(self.downBtn, row=11, col=2)
        self.w6.addWidget(self.leftBtn, row=10, col=1)
        self.w6.addWidget(self.rightBtn, row=10, col=3)
        self.w6.addWidget(self.stepGratingEdit, row=10, col=2)
        self.w6.addWidget(self.positionLabel, row=12, col=1)
        self.w6.addWidget(self.position_xEdt, row=12, col=2)
        self.w6.addWidget(self.position_yEdt, row=12, col=3)
        self.w6.addWidget(self.fresnelLabel, row=17, col=0)
        self.w6.addWidget(self.fresnelEpsilonLabel, row=17, col=1)
        self.w6.addWidget(self.fresnelEpsilonEdt, row=17, col=2)
        self.w6.addWidget(self.paramPLabel, row=21, col=1)
        self.w6.addWidget(self.paramPEdt, row=21, col=2)
        self.w6.addWidget(self.paramWavelengthLabel, row=22, col=1)
        self.w6.addWidget(self.paramWavelengthEdit, row=22, col=2)

        self.upBtn.clicked.connect(self.moveUp)
        self.downBtn.clicked.connect(self.moveDown)
        self.leftBtn.clicked.connect(self.moveLeft)
        self.rightBtn.clicked.connect(self.moveRight)
        self.stepGratingEdit.returnPressed.connect(self.updateStepGrating)
        self.position_xEdt.returnPressed.connect(self.updatePositionGrating)
        self.position_yEdt.returnPressed.connect(self.updatePositionGrating)
        self.fresnelEpsilonEdt.returnPressed.connect(self.createFresnelPattern)
        self.paramPEdt.returnPressed.connect(self.updateParamP) #2020/01/27
        self.paramWavelengthEdit.returnPressed.connect(self.updateParamWavelength)

    def _build_advanced_control_panel(self):
        """Corrections, pupil, traps, SO, LG (`self.w7`)."""
        self.Advanced_control_layout = pg.LayoutWidget()

        self.label_Advanced_control_layout = QtWidgets.QLabel(""" -- ADVANCED CONTROL -- """)
        self.label_Advanced_control_layout.setAlignment(QtCore.Qt.AlignCenter)
        self.correctionLabel = QtWidgets.QLabel('CORRECTION')
        self.correctionLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.ManufacturerCorrectionChkBx = QtWidgets.QCheckBox('Manufacturer correction')
        self.ManufacturerCorrectionChkBx.setChecked(True)
        self.zernikeCorrectionChkBx = QtWidgets.QCheckBox('Aberration control')
        self.zernikeCorrectionChkBx.setChecked(True)
        self.pupilLabel = QtWidgets.QLabel('PUPIL')
        self.pupilLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.pupilChkBx = QtWidgets.QCheckBox('radius (mm):')
        self.pupilChkBx.setChecked(False)
        self.pupilEdt = MyQtLineEdit('1')
        self.pupilEdt.setFixedWidth(100)
        self.pupilCenterLabel = QtWidgets.QLabel('PUPIL CENTER (mm)[x0, y0]:')
        self.pupilCenterEdt_x0 = MyQtLineEdit(str(0))
        self.pupilCenterEdt_y0 = MyQtLineEdit(str(0))
        self.pupilCenterEdt_x0.setFixedWidth(50)
        self.pupilCenterEdt_y0.setFixedWidth(50)
        self.pupilOutLabel = QtWidgets.QLabel('Move out')
        self.pupilOutLineEdit = MyQtLineEdit('40')
        self.pupilOutLineEdit.setFixedWidth(50)
        # self.pupilTypeCB = QtWidgets.QComboBox()
        # self.pupilTypeCB.addItems(['circular','square','rectangular','rectangular_rotated', 'rectangular_rotated_byAtom'])
        # self.ovulaLabel = QtWidgets.QLabel('OVULA TRAP')
        # self.ovulaLabel.setAlignment(QtCore.Qt.AlignCenter)
        # self.ovulaTrapChkBx = QtWidgets.QCheckBox('Disk radius (mm) :')
        # self.ovulaTrapChkBx.setChecked(False)
        # self.sizeRingEdt = MyQtLineEdit('2.5')
        # self.sizeRingEdt.setFixedWidth(50)
        # self.tem01Label = QtWidgets.QLabel('TEM01')
        # self.tem01Label.setAlignment(QtCore.Qt.AlignCenter)
        # self.tem01ChkBx = QtWidgets.QCheckBox('Spacing (in um) :')
        # self.tem01ChkBx.setChecked(False)
        # self.tem01spacingEdt = MyQtLineEdit(str(self.spacing))
        # self.tem01spacingEdt.setFixedWidth(50)
        # self.tem01PfactorLabel = QtWidgets.QLabel('P factor (in pi, default=1) :')
        # self.tem01PfactorEdt = MyQtLineEdit('1')
        # self.tem01PfactorEdt.setFixedWidth(50)
        # self.tem01ThetaLabel = QtWidgets.QLabel('Angle (in degree, default=0) :')
        # self.tem01ThetaEdt = MyQtLineEdit(str(self.theta_deg) )
        # self.tem01ThetaEdt.setFixedWidth(50)
        # self.SOLabel = QtWidgets.QLabel('Superoscillation')
        # self.SOLabel.setAlignment(QtCore.Qt.AlignCenter)
        # self.SOChkBx = QtWidgets.QCheckBox('r_pi [from 0 to 1]:')
        # self.SOChkBx.setChecked(False)
        # self.SOrpiEdt = MyQtLineEdit(str(self.r_pi))
        # self.SOrpiEdt.setFixedWidth(50)
        # self.SOPfactorLabel = QtWidgets.QLabel('P factor (in pi, default=1) :')
        # self.SOPfactorEdt = QtWidgets.QLineEdit('1')
        # self.SOPfactorEdt.setFixedWidth(50)
        # self.SOCenterLabel = QtWidgets.QLabel('SO CENTER (mm)[x0, y0]:')
        # self.SOCenterEdt_x0 = QtWidgets.QLineEdit(str(0))
        # self.SOCenterEdt_y0 = QtWidgets.QLineEdit(str(0))
        # self.SOCenterEdt_x0.setFixedWidth(50)
        # self.SOCenterEdt_y0.setFixedWidth(50)
        # self.LGLabel = QtWidgets.QLabel('Laguerre-Gaussian')
        # self.LGLabel.setAlignment(QtCore.Qt.AlignCenter)
        # self.LGChkBx = QtWidgets.QCheckBox('LG01')
        # self.LGChkBx.setChecked(False)
        # self.LGPfactorLabel = QtWidgets.QLabel('P factor (in pi, default=1) :')
        # self.LGPfactorEdt = QtWidgets.QLineEdit('1')
        # self.LGPfactorEdt.setFixedWidth(50)
        # self.LGCenterLabel = QtWidgets.QLabel('LG CENTER (mm)[x0, y0]:')
        # self.LGCenterEdt_x0 = QtWidgets.QLineEdit(str(0))
        # self.LGCenterEdt_y0 = QtWidgets.QLineEdit(str(0))
        # self.LGCenterEdt_x0.setFixedWidth(50)
        # self.LGCenterEdt_y0.setFixedWidth(50)

        self.Advanced_control_layout.addWidget(self.label_Advanced_control_layout, row=0, colspan=3)
        self.Advanced_control_layout.addWidget(self.correctionLabel, row=23, col=0)
        self.Advanced_control_layout.addWidget(self.ManufacturerCorrectionChkBx, row=23, col=1)
        self.Advanced_control_layout.addWidget(self.zernikeCorrectionChkBx, row=23, col=2)
        self.Advanced_control_layout.addWidget(self.pupilLabel, row=25, col=0)
        self.Advanced_control_layout.addWidget(self.pupilChkBx, row=25, col=1)
        self.Advanced_control_layout.addWidget(self.pupilEdt, row=25, col=2)
        # self.Advanced_control_layout.addWidget(self.pupilTypeCB, row=25, col=3)
        self.Advanced_control_layout.addWidget(self.pupilCenterLabel, row=26, col=0)
        self.Advanced_control_layout.addWidget(self.pupilCenterEdt_x0, row=26, col=1)
        self.Advanced_control_layout.addWidget(self.pupilCenterEdt_y0, row=26, col=2)
        self.Advanced_control_layout.addWidget(self.pupilOutLabel, row=28, col=1)
        self.Advanced_control_layout.addWidget(self.pupilOutLineEdit, row=28, col=2)
        # self.Advanced_control_layout.addWidget(self.ovulaLabel, row=35, col=0)
        # self.Advanced_control_layout.addWidget(self.ovulaTrapChkBx, row=35, col=1)
        # self.Advanced_control_layout.addWidget(self.sizeRingEdt, row=35, col=2)
        # self.Advanced_control_layout.addWidget(self.tem01Label, row=45, col=0)
        # self.Advanced_control_layout.addWidget(self.tem01ChkBx, row=45, col=1)
        # self.Advanced_control_layout.addWidget(self.tem01spacingEdt, row=45, col=2)
        # self.Advanced_control_layout.addWidget(self.tem01PfactorLabel, row=45, col=3)
        # self.Advanced_control_layout.addWidget(self.tem01PfactorEdt, row=45, col=4)
        # self.Advanced_control_layout.addWidget(self.tem01ThetaLabel, row=47, col=1)
        # self.Advanced_control_layout.addWidget(self.tem01ThetaEdt, row=47, col=2)
        # self.Advanced_control_layout.addWidget(self.SOLabel, row=55, col=0)
        # self.Advanced_control_layout.addWidget(self.SOChkBx, row=55, col=1)
        # self.Advanced_control_layout.addWidget(self.SOrpiEdt, row=55, col=2)
        # self.Advanced_control_layout.addWidget(self.SOPfactorLabel, row=55, col=3)
        # self.Advanced_control_layout.addWidget(self.SOPfactorEdt, row=55, col=4)
        # self.Advanced_control_layout.addWidget(self.SOCenterLabel, row=56, col=1)
        # self.Advanced_control_layout.addWidget(self.SOCenterEdt_x0, row=56, col=2)
        # self.Advanced_control_layout.addWidget(self.SOCenterEdt_y0, row=56, col=3)
        # self.Advanced_control_layout.addWidget(self.LGLabel, row=65, col=0)
        # self.Advanced_control_layout.addWidget(self.LGChkBx, row=65, col=1)
        # self.Advanced_control_layout.addWidget(self.LGPfactorLabel, row=65, col=3)
        # self.Advanced_control_layout.addWidget(self.LGPfactorEdt, row=65, col=4)
        # self.Advanced_control_layout.addWidget(self.LGCenterLabel, row=66, col=1)
        # self.Advanced_control_layout.addWidget(self.LGCenterEdt_x0, row=66, col=2)
        # self.Advanced_control_layout.addWidget(self.LGCenterEdt_y0, row=66, col=3)

        self.ManufacturerCorrectionChkBx.stateChanged.connect(self.updatePhase)
        self.zernikeCorrectionChkBx.stateChanged.connect(self.updatePhase)
        self.pupilChkBx.stateChanged.connect(self.calculatePupil)
        self.pupilEdt.returnPressed.connect(self.calculatePupil)
        self.pupilCenterEdt_x0.returnPressed.connect(self.calculatePupil)
        self.pupilCenterEdt_y0.returnPressed.connect(self.calculatePupil)
        self.pupilOutLineEdit.returnPressed.connect(self.update_k_dump)
        # self.ovulaTrapChkBx.stateChanged.connect(self.createOvulaDisk)
        # self.sizeRingEdt.returnPressed.connect(self.createOvulaDisk)
        # self.tem01ChkBx.stateChanged.connect(lambda : self.createTEM01trap())
        # self.tem01spacingEdt.returnPressed.connect(self.createTEM01trap)
        # self.tem01PfactorEdt.returnPressed.connect(self.createTEM01trap)
        # self.tem01ThetaEdt.returnPressed.connect(self.createTEM01trap_angle)
        # self.SOChkBx.stateChanged.connect(lambda : self.createSOtrap())
        # self.SOrpiEdt.returnPressed.connect(self.createSOtrap)
        # self.SOPfactorEdt.returnPressed.connect(self.createSOtrap)
        # self.SOCenterEdt_x0.returnPressed.connect(self.createSOtrap)
        # self.SOCenterEdt_y0.returnPressed.connect(self.createSOtrap)
        # self.LGChkBx.stateChanged.connect(lambda : self.createLGtrap())
        # self.LGPfactorEdt.returnPressed.connect(self.createLGtrap)
        # self.LGCenterEdt_x0.returnPressed.connect(self.createLGtrap)
        # self.LGCenterEdt_y0.returnPressed.connect(self.createLGtrap)

    def _build_aberration_control_panel(self):
        """Zernike / aberration sliders and save (`self.Abberation_control_layout`)."""
        self.Abberation_control_layout = pg.LayoutWidget()

        self.label_Abberation_control_layout = QtWidgets.QLabel(""" -- ABERRATION CONTROL -- """)
        self.label_Abberation_control_layout.setAlignment(QtCore.Qt.AlignCenter)
        self.Abberation_control_layout.addWidget(self.label_Abberation_control_layout, row=0, colspan = 2)

        self.aber_lineEdit_list = []
        for i, aberration_name in enumerate(self.aberration_label_list):
            label = QtWidgets.QLabel(aberration_name)
            lineEdit = MyQtLineEdit('0')
            lineEdit.setFixedWidth(50)
            self.Abberation_control_layout.addWidget(label, row=1+i, col = 0)
            self.Abberation_control_layout.addWidget(lineEdit, row=1+i, col = 1)

            lineEdit.returnPressed.connect(self.updateAberration)

            self.aber_lineEdit_list.append(lineEdit)

        self.loadDefaultZernikeCoeffsBtn = QtWidgets.QPushButton('Load default coefficients')
        _zernike_save_row = 1 + len(self.aberration_label_list)
        self.Abberation_control_layout.addWidget(self.loadDefaultZernikeCoeffsBtn, row=_zernike_save_row, col=0, colspan=2)
        self.loadDefaultZernikeCoeffsBtn.clicked.connect(self._load_default_zernike_coefficients_json)

        _coef_row = _zernike_save_row + 1
        self.loadZernikeCoeffsBtn = QtWidgets.QPushButton('Load coefficients…')
        self.saveZernikeCoeffsBtn = QtWidgets.QPushButton('Save coefficients…')
        self.Abberation_control_layout.addWidget(self.loadZernikeCoeffsBtn, row=_coef_row, col=0)
        self.Abberation_control_layout.addWidget(self.saveZernikeCoeffsBtn, row=_coef_row, col=1)
        self.loadZernikeCoeffsBtn.clicked.connect(self._load_zernike_coefficients_json)
        self.saveZernikeCoeffsBtn.clicked.connect(self._save_zernike_coefficients_json)

    def _build_last_sent_preview_panel(self):
        """Preview of last uint8 frame sent to the SLM (`self.w_last_sent`)."""
        self.last_sent_preview_layout = pg.LayoutWidget()
        self.label_last_sent_preview = QtWidgets.QLabel("Last pattern sent to SLM")
        self.label_last_sent_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.last_sent_plot = pg.PlotWidget()
        self.last_sent_plot.setAspectLocked(True)
        self.last_sent_plot.hideButtons()
        self.last_sent_plot.setMenuEnabled(False)
        for _ax in ("left", "bottom", "right", "top"):
            self.last_sent_plot.plotItem.hideAxis(_ax)
        self.last_sent_img_item = pg.ImageItem()
        self.last_sent_plot.addItem(self.last_sent_img_item)
        _vb = self.last_sent_plot.getViewBox()
        _vb.setMouseEnabled(False, False)
        # Resize must not change the visible *data* range (avoids apparent cropping).
        if hasattr(_vb, "disableAutoRange"):
            _vb.disableAutoRange()
        else:
            _vb.enableAutoRange(enable=False)
        self.last_sent_plot.plotItem.setClipToView(False)
        try:
            _grey_cmap = pg.colormap.get("grey", source="matplotlib", skipCache=True)
        except Exception:
            _grey_cmap = pg.colormap.get("grey")
        _cb_kw = dict(
            values=(0, 255),
            colorMap=_grey_cmap,
            interactive=False,
            limits=(0, 255),
        )
        try:
            self.last_sent_colorbar = pg.ColorBarItem(colorMapMenu=False, **_cb_kw)
        except TypeError:
            self.last_sent_colorbar = pg.ColorBarItem(**_cb_kw)
        self.last_sent_colorbar.setImageItem(
            self.last_sent_img_item,
            insert_in=self.last_sent_plot.plotItem,
        )
        self.last_sent_plot.setMinimumSize(280, 220)
        self.last_sent_plot.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.label_slm_temperature = QtWidgets.QLabel("SLM temperature: —")
        self.label_slm_temperature.setAlignment(QtCore.Qt.AlignCenter)
        _temp_font = self.label_slm_temperature.font()
        _temp_font.setBold(True)
        self.label_slm_temperature.setFont(_temp_font)
        self.last_sent_preview_layout.addWidget(self.label_last_sent_preview, row=0, col=0)
        self.last_sent_preview_layout.addWidget(self.label_slm_temperature, row=1, col=0)
        self.last_sent_preview_layout.addWidget(self.last_sent_plot, row=2, col=0)

    def _build_arbitrary_arrays_panel(self):
        """Build the ARBITRARY ARRAYS column (`self.arbitrary_arrays_panel`): patterns, traps, lattice export, algorithms."""
        self.arbitrary_arrays_panel = pg.LayoutWidget()

        self.label_Layout_Arbitrary_Array = QtWidgets.QLabel(""" -- ARBITRARY ARRAYS -- """)
        self.label_Layout_Arbitrary_Array.setAlignment(QtCore.Qt.AlignCenter)
        self.generateBlankBtn = QtWidgets.QPushButton('Blank')
        self.loadBtn = QtWidgets.QPushButton('Load Pattern')
        self.saveBtn = QtWidgets.QPushButton('Save Pattern')
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
        self.runBtn = QtWidgets.QPushButton('Run Algorithm')
        self.iterationsEdit = MyQtLineEdit(str(self.GS_maxit))
        self.methodCB = QtWidgets.QComboBox()
        for key in self.methodList:
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

        line_idx += 1
        self.arbitrary_arrays_panel.addWidget(self.runBtn, row=line_idx, col=0)
        self.arbitrary_arrays_panel.addWidget(self.methodCB, row=line_idx, col=1)
        self.arbitrary_arrays_panel.addWidget(self.iterationsEdit, row=line_idx, col=2)

        line_idx += 1
        self.arbitrary_arrays_panel.addWidget(self.equalize_with_atoms_Btn, row=line_idx, col=0)
        self.arbitrary_arrays_panel.addWidget(self.computeHologram_Btn, row=line_idx, col=1)
        self.arbitrary_arrays_panel.addWidget(self.devide_slm_area_Btn, row=line_idx, col=2)

        self.generateBlankBtn.clicked.connect(self.generateBlank)
        self.loadBtn.clicked.connect(self.loadPattern_clicked)
        self.saveBtn.clicked.connect(self.savePattern_clicked)
        self.generateTrapsBtn.clicked.connect(self.generateTraps)
        self.saveArrayCoordinatesBtn.clicked.connect(self.saveArrayCoordinates_clicked)
        self.runBtn.clicked.connect(self.runAlgorithm_clicked)
        self.equalize_with_atoms_Btn.clicked.connect(self.equalize_with_atoms)
        self.devide_slm_area_Btn.clicked.connect(self.devide_slm_area)
        self.computeHologram_Btn.clicked.connect(self.computeHologram_clicked)

    #==============================================================================
    # GUI functions
    #==============================================================================
    def moveLeft(self):
        self.dx = self.dx - self.deltax
        self.createGratingPattern(self.dx, self.dy)

    def moveRight(self):
        self.dx = self.dx + self.deltax
        self.createGratingPattern(self.dx, self.dy)

    def moveUp(self):
        self.dy = self.dy + self.deltay
        self.createGratingPattern(self.dx, self.dy)

    def moveDown(self):
        self.dy = self.dy - self.deltay
        self.createGratingPattern(self.dx, self.dy)

    def updateHorizontalGrating(self,x):
        self.dx = x
        self.position_xEdt.setText(str(x))
        self.createGratingPattern(self.dx, self.dy)

    def updateVerticalGrating(self,y):
        self.dy = y
        self.position_yEdt.setText(str(y))
        self.createGratingPattern(self.dx, self.dy)

    def updatePositionGrating(self):
        '''
        Updates the grating with the values in the interface
        '''
        self.dx = float(self.position_xEdt.text())
        self.dy = float(self.position_yEdt.text())
        self.createGratingPattern(self.dx, self.dy)

    def updateStepGrating(self):
        '''Updates the grating step when a return is pressed in the field'''
        self.deltax = float(self.stepGratingEdit.text())
        self.deltay = self.deltax

    def updateParamP(self):
        '''Updates the P parameter when a return is pressed in the field'''
        self.param_P = float(self.paramPEdt.text())
        self.updatePhase()

    def updateParamWavelength(self):
        param.lamb = float(self.paramWavelengthEdit.text())*1e-9
        position_to_step_per_pixel = 2*np.pi/param.lamb*(param.M/param.f)*param.pslm

    def runAlgorithm_clicked(self):
        method = str(self.methodCB.currentText())
        self.runAlgorithm(method)

    def runAlgorithm(self, method):
        '''
        This function chooses the method to calculate the phase mask for
        the desired trap positions
        '''
        try :
            image = self.methodList[method]()
        except :
            print('method not in list')
            raise

        if image is None:
            return

        # Put the phase pattern on the GUI and on the SLM
        self.phasePattern = image
        self.updatePhase()

    def savePattern_clicked(self):
        '''
        Opens the dialog and saves the selected file
        '''
        os.makedirs(SLM_PATTERNS_DIR, exist_ok=True)
        self.file = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Pattern', SLM_PATTERNS_DIR)[0]
        if not self.file:
            return
        self.savePattern(name = self.file)

    def savePattern(self, name = ''):
        '''
        Opens the dialog and saves the selected file
        '''
        if not self._ensure_traps_loaded():
            print('Save pattern aborted: trap coordinates are not available.')
            return

        self.file = name
        if not(self.file[-4:] == '.pkl'):
            self.file += '.pkl'

        self.pack_traps_dict()

        a_file = open(self.file, "wb")
        pickle.dump(self.traps_dict, a_file)
        a_file.close()


    def loadPattern_clicked(self):
        '''
        Open the dialog and gets the selected file
        '''
        os.makedirs(SLM_PATTERNS_DIR, exist_ok=True)
        self.file = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', SLM_PATTERNS_DIR)[0]
        if not self.file:
            return
        self.loadPattern(name = self.file)

    def loadPattern(self, name = ''):
        print('SLM file')
        print(name)
        self.file = name

        #obsolete
        if name[-3:] == 'npy':
            self.phasePattern = np.load(name)
            self.phasePattern = self.phasePattern[:,:792] #in case that the pattern size is 600x800
        #obsolete
        elif name[-3:] == 'bmp':
            self.phasePattern = np.asarray(Image.open(name), dtype=np.float64)
            self.phasePattern *= 2*np.pi/256
        #current
        elif name[-3:] == 'pkl':
            try:
                a_file = open(self.file, "rb")
                self.traps_dict = pickle.load(a_file)
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

        self.updatePhase()

    def pack_traps_dict(self):
        self.traps_dict['traps_pos'] = self.traps_pos
        self.traps_dict['traps_k'] = self.traps_k
        self.traps_dict['theta_traps'] = self.theta_traps
        self.traps_dict['weight_traps'] = self.weight_traps
        self.traps_dict['pupil_traps'] = self.pupil_traps
        self.traps_dict['phasePattern'] = self.phasePattern

    def unpack_traps_dict(self):
        self.traps_pos      = self.traps_dict['traps_pos']
        self.traps_k        = self.traps_dict['traps_k']
        self.theta_traps    = self.traps_dict['theta_traps']
        self.weight_traps   = self.traps_dict['weight_traps']
        self.pupil_traps    = self.traps_dict['pupil_traps']
        self.phasePattern   = self.traps_dict['phasePattern']

        #Added in v29 to allow recompute the hologram based on updated traps_dict parameters
        if 'computeHologram' in self.traps_dict.keys():
            print(self.traps_dict['computeHologram'])

            if self.traps_dict['computeHologram'] :
                ph = self.computeHologram()
                if ph is not None:
                    self.phasePattern = ph
                    self.updatePhase()



    #==============================================================================
    # Elementary patterns
    #==============================================================================
    def generateBlank(self):
        '''Generates a blank phase mask'''
        self.phasePattern = np.zeros([param.ImgResY, param.ImgResX])
        self.updatePhase()

    def createGratingPattern(self, dx, dy):
        '''
        Create the grating phase pattern with user-defined wavevectors kx and ky

        (1) To shift the trap by a distance d, we need to deflect the beam by an angle
        alpha = M * d / f

        with f the effective focal length of the objective
        and M the telescope magnification between SLM and objective

        (2) A grating with a phase shift of phi between two pixels gives a deflection of
        alpha = (phi / 2pi * wavelength) / pixel separation

        This gives us the change of phase from pixel to pixel to shift the trap by a distance d:

        phi = 2pi * M/f * d * pixel separation / wavelength
        '''
        self.position_xEdt.setText(str(dx))
        self.position_yEdt.setText(str(dy))

        dphi_x = 2 * np.pi / param.lamb * param.M / param.f * dx * 1e-6
        dphi_y = 2 * np.pi / param.lamb * param.M / param.f * dy * 1e-6

        self.im_grating  = self.X * dphi_x + self.Y * dphi_y
        self.updatePhase()

    def createFresnelPattern(self, dz = None):
        '''
        Creates a Fresnel lens pattern with a user-defined defocus
        '''
        if dz == None:
            dz = float(self.fresnelEpsilonEdt.text())
        else:
            self.fresnelEpsilonEdt.setText(str(dz))

        self.fresnelPhase = np.pi/param.lamb * (param.M/param.f)**2 * dz*1e-6 * (self.X**2+self.Y**2)
        self.updatePhase()

    def update_k_dump(self):
        d_out = float(self.pupilOutLineEdit.text())
        self.k_dump = np.array([d_out,0])*1e-6*position_to_step_per_pixel

    def calculatePupil(self):
        d_out = float(self.pupilOutLineEdit.text())
        dphi_x = 2*np.pi/param.lamb * param.M/param.f * d_out*1e-6
        dphi_y = 2*np.pi/param.lamb * param.M/param.f * d_out*1e-6
        self.grating_out = self.X*dphi_x + self.Y*dphi_y

        pupil_x0 = float(self.pupilCenterEdt_x0.text())*1e-3
        pupil_y0 = float(self.pupilCenterEdt_y0.text())*1e-3
        # self.pupil_type = str(self.pupilTypeCB.currentText())
        self.pupil_type = 'circular'
        self.pupil = np.zeros((param.ImgResY,param.ImgResX)).astype('float64')

        if self.pupil_type == 'circular':
            r_pupil = float(self.pupilEdt.text())*1e-3

            self.pupil[((self.X - pupil_x0)**2 + (self.Y - pupil_y0)**2)< r_pupil**2] = 1.
            self.grating_out[((self.X - pupil_x0)**2 + (self.Y - pupil_y0)**2)< r_pupil**2] = 0.

        # elif self.pupil_type == 'rectangular':
        #     txt = self.pupilEdt.text()
        #     width, height = np.array(txt.split(','),dtype = 'float')*1e-3

        #     cut_select = np.logical_and(np.abs(self.X)< width,np.abs(self.Y)<height)
        #     self.pupil[cut_select] = 1.
        #     self.grating_out[cut_select] = 0.

        # elif self.pupil_type == 'rectangular_rotated':
        #     txt = self.pupilEdt.text()
        #     width, height, θ = np.array(txt.split(','),dtype = 'float')
        #     width *= 1e-3
        #     height *=1e-3
        #     θ *= np.pi/180
        #     x_rot = np.cos(θ)*self.X + np.sin(θ)*self.Y
        #     y_rot = np.sin(θ)*self.X - np.cos(θ)*self.Y

        #     cut_select = np.logical_and(np.abs(x_rot)< width,np.abs(y_rot)<height)
        #     self.pupil[cut_select] = 1.
        #     self.grating_out[cut_select] = 0.

        # elif self.pupil_type == 'square':
        #     r_pupil = float(self.pupilEdt.text())*1e-3

        #     self.pupil[np.logical_and(np.abs(self.X)< r_pupil,np.abs(self.Y)<r_pupil)] = 1.
        #     self.grating_out[np.logical_and(np.abs(self.X)< r_pupil, np.abs(self.Y)<r_pupil)] = 0.

        # elif self.pupil_type == 'rectangular_rotated_byAtom':
        #     self.pupil = np.ones((param.ImgResY,param.ImgResX)).astype('float64')

        self.updatePhase()


    def applyPupil(self):
        '''
        Create an artificial pupil limiting the NA of the trap.

        Inside a radius defined by the user in self.pupilEdt
        the phase pattern is not modified

        Outside this radius, the phase pattern is replaced by a grating with
        a large (user defined) wavevector to displace the light in this region
        '''

        # self.pupil_type = str(self.pupilTypeCB.currentText())

        # if self.pupil_type == 'rectangular_rotated_byAtom':
        #     """
        #     Apply pupil by atoms.
        #     """
        #     if not self._ensure_traps_loaded():
        #         print('applyPupil (by atom): falling back to standard pupil; trap load failed.')
        #         self.totalPhase *= self.pupil
        #         self.totalPhase += self.grating_out
        #         return

        #     #Chew
        #     txt = self.pupilEdt.text()
        #     width, height, θ = np.array(txt.split(','),dtype = 'float')
        #     print('original long axis')
        #     print(self.pupil_traps['pupil_longaxis'])

        #     long_axis = self.pupil_traps['pupil_longaxis'] * width
        #     short_axis = self.pupil_traps['pupil_shortaxis'] * height
        #     pupil_angle = self.pupil_traps['pupil_angle']

        #     print('after long axis')
        #     print(self.pupil_traps['pupil_longaxis']* width)

        #     pupil_traps = {
        #         'pupil_longaxis': long_axis,
        #         'pupil_shortaxis': short_axis,
        #         'pupil_angle': pupil_angle,
        #         }

        #     phi = trap_phase_to_slm(
        #         self.traps_k,
        #         self.theta_traps,
        #         self.weight_traps,
        #         pupil = pupil_traps,
        #         shape = self.SLM_shape,
        #         k_dump = self.k_dump
        #         )
        #     self.totalPhase = self.totalPhase - self.phasePattern + phi

        # else:
        self.totalPhase *= self.pupil
        self.totalPhase += self.grating_out


    def apply_pupil_longaxis_byatom_loop(self, longaxis_ratio = 1.0):
        txt = self.pupilEdt.text()
        width, height, θ = np.array(txt.split(','),dtype = 'float')

        width = longaxis_ratio

        new_pupil = str(width) + ',' + str(height) + ',' + str(θ)
        self.pupilEdt.setText(new_pupil)

        self.updatePhase()

    def apply_pupil_shortaxis_byatom_loop(self, shortaxis_ratio = 1.0):
        txt = self.pupilEdt.text()
        width, height, θ = np.array(txt.split(','),dtype = 'float')

        height = shortaxis_ratio

        new_pupil = str(width) + ',' + str(height) + ',' + str(θ)
        self.pupilEdt.setText(new_pupil)

        self.updatePhase()

    def createOvulaDisk(self):
        '''
        Make the so-called ovula trap

        Shift the phase by pi in an inner disk of radius defined by the user
        '''
        r = float(self.sizeRingEdt.text())/1000

        phi_ovula = np.zeros((param.ImgResY,param.ImgResX))
        for i in range(param.ImgResY):
            for j in range(param.ImgResX):
                if ((self.X[i,j]**2 + self.Y[i,j]**2)< r**2):
                    phi_ovula[i,j+PUPIL_VERTICAL_OFFSET] = np.pi
                else:
                    phi_ovula[i,j+PUPIL_VERTICAL_OFFSET] = 0

        self.phi_ovula = phi_ovula
        self.updatePhase()

    def createTEM01trap(self, value=None):
        '''
        Make the TEM01-like traps
        Make a patten with a 0-pi phase difference between upper and lower side
        '''
        if type(value) == float or type(value) == int:
            self.spacing = value
            self.tem01spacingEdt.setText(f'{self.spacing: .3}')
        else:
            self.spacing = float(self.tem01spacingEdt.text())

        phi_tem01 = np.zeros((param.ImgResY,param.ImgResX))
        P_factor = float(self.tem01PfactorEdt.text())

        self.theta_deg = float(self.tem01ThetaEdt.text())
        theta_rad = self.theta_deg/180*np.pi
        x = np.arange(-param.ImgResX/2,param.ImgResX/2)*param.pslm
        y = np.arange(-param.ImgResY/2,param.ImgResY/2)*param.pslm
        X,Y = np.meshgrid(x,y)
        amp_list = np.sin(-2*np.pi*0.5*self.spacing*1e-6*(np.cos(theta_rad)*Y+np.sin(theta_rad)*X)/(param.lamb*param.f/param.M))
        phase_list = np.array([amp_list >= 0])*np.pi*P_factor
        phi_tem01 = phase_list[0,:,:]

        self.phi_tem01 = phi_tem01
        self.updatePhase()

    def createTEM01trap_angle(self, value=None):
        '''
        Make the TEM01-like traps(change the angle)

        '''

        if type(value) == float or type(value) == int:
            self.theta_deg = value
            self.tem01ThetaEdt.setText(f'{self.theta_deg: .3}')

        else:
            self.theta_deg = float(self.tem01ThetaEdt.text()) #[m]

        self.spacing = float(self.tem01spacingEdt.text())
        phi_tem01 = np.zeros((param.ImgResY,param.ImgResX))
        #P_factor = 1
        P_factor = float(self.tem01PfactorEdt.text())
        #theta_deg = -45
        theta_rad = self.theta_deg/180*np.pi
        x = np.arange(-param.ImgResX/2,param.ImgResX/2)*param.pslm
        y = np.arange(-param.ImgResY/2,param.ImgResY/2)*param.pslm
        X,Y = np.meshgrid(x,y)
        amp_list = np.sin(-2*np.pi*0.5*self.spacing*1e-6*(np.cos(theta_rad)*Y+np.sin(theta_rad)*X)/(param.lamb*param.f/param.M))
        phase_list = np.array([amp_list >= 0])*np.pi*P_factor
        phi_tem01 = phase_list[0,:,:]

        self.phi_tem01 = phi_tem01
        self.updatePhase()

    def createSOtrap(self, value=None):
        '''
        Make the superoscillation traps
        Make a circular patten with a 0-pi phase difference between center and outer region
        '''
        if type(value) == float or type(value) == int:
            self.r_pi = value
            self.SOrpiEdt.setText(f'{self.r_pi:.2f}')

        else:
            self.r_pi = float(self.SOrpiEdt.text()) #[unitless]

        SO_x0 = float(self.SOCenterEdt_x0.text())*1e-3 #[m]
        SO_y0 = float(self.SOCenterEdt_y0.text())*1e-3 #[m]
        r_pi = self.r_pi
        phi_SO = np.zeros((param.ImgResY,param.ImgResX))
        P_factor = float(self.SOPfactorEdt.text())
        x = np.arange(-param.ImgResX/2,param.ImgResX/2)*param.pslm
        y = np.arange(-param.ImgResY/2,param.ImgResY/2)*param.pslm
        X,Y = np.meshgrid(x,y)
        Rho = ((X-SO_x0)**2 + (Y-SO_y0)**2)**0.5
        phase_list = np.array([Rho <= r_pi*param.ImgResY/2*param.pslm])*np.pi*P_factor
        phi_SO = phase_list[0,:,:]

        self.phi_SO = phi_SO
        self.updatePhase()

    def createLGtrap(self):
        '''
        Make the Laguerre-Gaussian 01 mode traps.
        Make a spiral phase patten from 0 to 2 pi corresponding to the polar angle.
        '''

        LG_x0 = float(self.LGCenterEdt_x0.text())*1e-3 #[m]
        LG_y0 = float(self.LGCenterEdt_y0.text())*1e-3 #[m]

        phi_LG = np.zeros((param.ImgResY,param.ImgResX))
        P_factor = float(self.LGPfactorEdt.text())
        x = np.arange(-param.ImgResX/2,param.ImgResX/2)*param.pslm
        y = np.arange(-param.ImgResY/2,param.ImgResY/2)*param.pslm
        X,Y = np.meshgrid(x,y)
        Theta = np.arctan2((Y-LG_y0),(X-LG_x0)) + np.pi
        phase_list = Theta*P_factor
        phi_LG = phase_list#[0,:,:]

        self.phi_LG = phi_LG
        self.updatePhase()

    def generateLGMask(self):
        '''
        Generates the phase mask for Laguerre-Gauss beams

        CURRENTLY NOT ACCESSIBLE FROM THE GUI
        '''
        self.angularM = 1
        self.ImgCenterX = 400
        self.ImgCenterY = 300

        if self.angularM == 0:
            image = np.zeros([param.ImgResY, param.ImgResX])
        else:
            image = np.angle(np.exp((self.angularM*np.angle((self.X - self.ImgCenterX) + (self.Y - self.ImgCenterY)*1j))*1j))

        return image

    def updateAberration(self, aber_list= []):
        zernike_fit = np.zeros_like(self.zernike_polynomials[0])
        zernike_coeff = []
        N_poly = len(self.aber_lineEdit_list)

        for i in range(N_poly):
            if aber_list is not None and len(aber_list) > i:
                value = aber_list[i]
                self.aber_lineEdit_list[i].setText(str(value))
            else:
                value = float(self.aber_lineEdit_list[i].text())

            self.aberration_values[i] = value
            zernike_coeff.append(value)
            zernike_fit += value * self.zernike_polynomials[i]

        self.im_zernike = zernike_fit

        self.updatePhase()

    def updateAberration_ext(self,aberration_index, aberration_value):

        zernike_fit = np.zeros_like(self.zernike_polynomials[0])
        zernike_coeff = []
        N_poly = len(self.aber_lineEdit_list)


        for i in range(N_poly):
            if i != aberration_index:
                value = float(self.aber_lineEdit_list[i].text())
            else:
                value = aberration_value
                self.aber_lineEdit_list[i].setText(str(value))
                self.aberration_values[i] = value
            zernike_coeff.append(value)
            zernike_fit += value * self.zernike_polynomials[i]

        self.im_zernike = zernike_fit

        self.updatePhase()

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

    def adjustImageForS1028(self,array):
        '''
        array size adjustment for the SLM s1028, whose size is (1024,1272), 2021/11/08 Tomita
        '''
        v_s,h_s = 1024,1272 #size of the SLM s1028
        v,h = array.shape
        converted_array = np.zeros([v_s,h_s])
        converted_array[(v_s-v)//2:(v_s+v)//2,(h_s-h)//2:(h_s+h)//2] = np.fliplr(array)
        return converted_array

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
            for edt in self.aber_lineEdit_list:
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
        for label, c in zip(self.aberration_label_list, coeffs):
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
            if isinstance(labels, list) and labels != list(self.aberration_label_list):
                print(
                    "Warning: JSON labels differ from current GUI order; applying values by index only."
                )
            N = len(self.aber_lineEdit_list)
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
        for label in self.aberration_label_list:
            if label not in doc:
                vals.append(0.0)
            else:
                try:
                    vals.append(float(doc[label]))
                except (TypeError, ValueError):
                    vals.append(0.0)
        for k in doc:
            if k in _reserved or k in self.aberration_label_list:
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

    def updatePhase(self):
        '''
        Combines: grating, Fresnel lens and arbitrary arrays phase masks

        Also add (if checked by the user in the GUI):
            wavefront correction
            pupil
            ovula mask

        The final array has to be a uint8 type (8 bits integer) since the SLM
        has 2**8 level of phase shift.

        The (uint8) self.param_P value corresponds to the level where a phase shift
        is 2 pi (calibrated by the manufacturer or the user)
        '''

        self.totalPhase = self.im_grating + self.fresnelPhase + self.phasePattern

        if self.ManufacturerCorrectionChkBx.isChecked():
            self.totalPhase = self.totalPhase + self.SLM.correction_pattern
        if self.zernikeCorrectionChkBx.isChecked():
            self.totalPhase = self.totalPhase + self.im_zernike
        # if self.ovulaTrapChkBx.isChecked() :
        #     self.totalPhase = self.totalPhase + self.phi_ovula
        if self.pupilChkBx.isChecked():
            self.applyPupil()
        # if self.tem01ChkBx.isChecked():
        #     self.totalPhase = self.totalPhase + self.phi_tem01
        # if self.SOChkBx.isChecked():
        #     self.totalPhase = self.totalPhase + self.phi_SO
        # if self.LGChkBx.isChecked():
        #     self.totalPhase = self.totalPhase + self.phi_LG

        self.totalPhase = np.mod(self.totalPhase, 2*np.pi)

        self.totalPhase = np.round(self.param_P*self.totalPhase / (2*np.pi)).astype('uint8')
        self.SLM.updateArray(self.totalPhase)
        self._update_last_sent_preview()

    def _start_slm_temperature_monitor(self):
        """Poll Read_SLM_temperature once per second and log each sample."""
        self._temperature_log = DailyTemperatureLog()
        self._temp_log_error_reported = False
        self._slm_temp_timer = QtCore.QTimer(self)
        self._slm_temp_timer.setInterval(1000)
        self._slm_temp_timer.timeout.connect(self._refresh_slm_temperature)
        self._refresh_slm_temperature()
        self._slm_temp_timer.start()

    def _refresh_slm_temperature(self):
        comm = getattr(getattr(self, "SLM", None), "communication", None)
        if comm is None:
            self.label_slm_temperature.setText("SLM temperature: —")
            return
        try:
            temp_c = comm.read_temperature()
        except Exception:
            self.label_slm_temperature.setText("SLM temperature: unavailable")
            return
        self.label_slm_temperature.setText(f"SLM temperature: {temp_c:.1f} °C")
        try:
            self._temperature_log.append(temp_c)
        except Exception as exc:
            if not self._temp_log_error_reported:
                self._temp_log_error_reported = True
                print(f"Temperature log write failed: {exc}")

    def _fit_last_sent_view_to_image(self):
        """Lock the plot view to the full SLM frame so resize does not crop the preview."""
        if not hasattr(self, "totalPhase"):
            return
        arr = np.asarray(self.totalPhase)
        if arr.ndim != 2:
            return
        ny, nx = arr.shape
        vb = self.last_sent_plot.getViewBox()
        vb.setRange(xRange=(0, nx), yRange=(0, ny), padding=0)

    def _update_last_sent_preview(self):
        """Show the same buffer last passed to the SLM (device cannot return the live frame)."""
        arr = np.asarray(self.totalPhase)
        self.last_sent_img_item.setAutoLevels(False)
        self.last_sent_img_item.setImage(arr, autoLevels=False, levels=[0, 255])
        self.last_sent_colorbar.setLevels((0, 255), update_items=True)
        self._fit_last_sent_view_to_image()

    #==============================================================================
    # Functions for arbitrary arrays
    #==============================================================================
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
        self.traps_pos = coords_npy_to_traps_pos(coords_m)
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
        """Build k-vectors, random phases, pupils, and traps_dict from self.traps_pos (meters)."""
        self.N_traps = int(self.traps_pos.shape[0])
        print('Traps = ', self.N_traps)
        self.traps_k = np.copy(self.traps_pos[:, 1:4])
        self.traps_k[:, 0:2] *= position_to_step_per_pixel
        self.traps_k[:, 2] *= position_to_step_per_pixel_for_z

        self.theta_traps = np.random.rand(self.N_traps) * 2 * np.pi
        self.weight_traps = np.ones(self.N_traps)

        # pupilEdt is one number for circular/square, two for rectangular, three for rotated / traps
        txt = (self.pupilEdt.text() or '').strip()
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

        self.pupil_traps = {
            'pupil_longaxis': np.array([longaxis] * self.N_traps),
            'pupil_shortaxis': np.array([shortaxis] * self.N_traps),
            'pupil_angle': np.array([θ] * self.N_traps),
        }

        self.traps_dict = {
            'traps_pos': self.traps_pos,
            'traps_k': self.traps_k,
            'theta_traps': self.theta_traps,
            'weight_traps': self.weight_traps,
            'pupil_traps': self.pupil_traps,
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
        roi_sizes = [side]
        doc = {
            'format_version': 2,
            'centers': centers,
            'roi_sizes': roi_sizes,
            'x_com': 0.0,
            'y_com': 0.0,
            'angle_deg': float(angle_deg),
            'rescale': 1.0,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
            f.write('\n')

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

        self.equalize_gain = 1.#float(self.equalize_gain_le.text())

        #Update hologram
        self.phasePattern = self.GS_intensity(signal,self.equalize_gain)
        if self.phasePattern is None:
            return

        self.updatePhase()

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
            signal = self.map


        # signal += 1-np.mean(signal)
        if change :
            print('current stdev: %.1f'%(np.std(signal)*100))
        else :
            print('no change of initial hologram')

        self.equalize_gain = 1.#float(self.equalize_gain_le.text())

        #Update hologram
        phasePattern_orig = self.phasePattern
        phasePattern_cal = self.GS_intensity(signal,self.equalize_gain)
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

        self.phasePattern = phasePattern_orig

        self.updatePhase()

        #save hologram information
        timestr = time.strftime("%Y%m%d_%H%M%S")

        hologram_save_file = LOCAL_DIR+'\\holograms\\equalized_with_atoms_'+timestr+'.pkl'
        self.savePattern(name = LOCAL_DIR+'\\holograms\\equalized_with_atoms_'+timestr+'.pkl')

        print('done partial phase correction, saved at : ' + hologram_save_file)

    def devide_slm_area_loop(self, loop = None):

        global trap_equalization
        trap_equalization = reload(trap_equalization)
        #best_hologram =
        folder_dir = self.parent.experiment_directory

        change = True
        print('loop num', loop)

        if loop == 0:
            pass
        else:
            loop = int(loop)
            loop = int(loop-1)
            if loop !=0.0 :
                current_map = self.map

            self.map = trap_equalization.cal(folder_dir, loop = loop)

            std = np.std(self.map) #martin
            print(std)
            if std > self.std and (self.std * 100) < 0.5 : #martin
                change = False
                self.map = current_map
                self.phasePattern = self.best_hologram

            else : #martin
                self.std = std #martin
                self.best_hologram = self.phasePattern #martin
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
        self.phasePattern = phi
        self.updatePhase()

    def computeHologram(self):
        if not self._ensure_traps_loaded():
            return None
        # Calculate phase pattern with homemade gpu function
        phi = trap_phase_to_slm(
                self.traps_k,
                self.theta_traps,
                self.weight_traps,
                self.pupil_traps,
                shape = self.SLM_shape,
                )

        return phi

    def randomMask(self):
        '''
        random mask (RM)
        '''
        if not self._ensure_traps_loaded():
            return None
        randmat = np.ceil(np.random.rand(param.ImgResY,param.ImgResX)*self.N_traps).astype(int)

        phi = np.zeros((param.ImgResY,param.ImgResX))

        for i in range(param.ImgResY):
            for j in range(param.ImgResX):
                random_trap = randmat[i,j]-1

                k_x, k_y = self.traps_k[random_trap]
                phi[i,j] = self.i_slm*k_x + self.y_slm*k_y

        return phi

    def superposition(self):
        '''
        superposition of gratings and lenses (S)
        '''
        if not self._ensure_traps_loaded():
            return None
        self.theta_traps = np.zeros(self.N_traps)
        self.weight_traps = np.ones(self.N_traps)

        phi = self.computeHologram()

        return phi

    def randomSuperposition(self):
        '''
        Superposition of gratings with random phases.
        '''
        if not self._ensure_traps_loaded():
            return None
        self.theta_traps = np.random.rand(self.N_traps)*2*np.pi
        self.weight_traps = np.ones(self.N_traps)

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
        self.theta_traps = np.random.rand(self.N_traps)*2*np.pi
        self.weight_traps = np.ones(self.N_traps)

        #Calculate phase pattern with homemade gpu function
        phi = self.computeHologram()
        if phi is None:
            return None

        #(a) Calculate the intensity at trap position
        traps_amplitude = 1./param.ImgResY/param.ImgResX*slm_to_trap_amp(phi, self.traps_k)

        trap_intensities = np.real(traps_amplitude*np.conj(traps_amplitude))
        efficiency = np.sum(trap_intensities)
        uniformity = np.std(trap_intensities)/np.mean(trap_intensities)

        print('1st calculation:')
        print('Efficiency std = %.1f'%(efficiency*100))
        print('Intensity std = %.1f'%(uniformity*100))

        #Start the iterations5
        for k in range(GS_maxit):
            print ('    GS ' + str(k+1) +'/' + str(GS_maxit))

            #(a) Calculate the phase of each trap
            if k == 0: #(b0) if first time: start with random phases
                self.theta_traps = np.random.rand(self.N_traps)*2*np.pi
                self.weight_traps = np.ones(self.N_traps)
            else: #(b0)else  apply the GS technique
                self.theta_traps = np.angle(traps_amplitude)

            #(c) Calculate the new slm phase
            phi = self.computeHologram()
            if phi is None:
                return None

            #(a) Calculate the intensity at trap position
            traps_amplitude = 1./param.ImgResY/param.ImgResX*slm_to_trap_amp(phi, self.traps_k)

            trap_intensities = np.real(traps_amplitude*np.conj(traps_amplitude))
            efficiency = np.sum(trap_intensities)
            uniformity = np.std(trap_intensities)/np.mean(trap_intensities)
            print('Efficiency std = %.1f'%(efficiency*100))
            print('Intensity std = %.1f'%(uniformity*100))

        return phi

    def WGS(self, correct_diffraction_efficiency = False):
        '''
        Weighted Gerchberg−Saxton (GS). Otpimize the phase and weigth of each grating.
        '''
        print('Starting weighted Gerchberg-Saxton calculation')
        if not self._ensure_traps_loaded():
            return None

        if correct_diffraction_efficiency:
            # Calculate the radial displacement of each traps
            traps_displacements = ( (self.traps_pos[:,1]+self.dx*1e-6)**2 + (self.traps_pos[:,2]+self.dy*1e-6)**2 )**0.5

            # The SLM diffraction efficiency depends on the displacement of the traps
            SLM_diffraction_efficiency = 1. - 0.5*traps_displacements/SLM_DIFFRACTION_EFFICIENCY_HWHM

            # Correct for the limited diffraction efficiency
            target_intensities = 1./SLM_diffraction_efficiency
            target_amplitudes = target_intensities**0.5
            target_amplitudes_norm = target_amplitudes/np.mean(target_amplitudes)

        else:
            target_intensities = np.ones(self.N_traps)
            target_amplitudes_norm = np.ones(self.N_traps)

        # Read the number of iterations
        GSW_maxit = int(self.iterationsEdit.text())

        # Calculate phase pattern with homemade gpu function
        phi = self.computeHologram()
        if phi is None:
            return None

        # Measure the trap to trap deviation of intensity
        traps_amplitude         = 1./param.ImgResY/param.ImgResX*slm_to_trap_amp(phi, self.traps_k)
        trap_intensities        = np.real(traps_amplitude*np.conj(traps_amplitude))
        deviation_from_target   = trap_intensities/target_intensities
        rel_dev                 = deviation_from_target**0.5/np.mean(deviation_from_target**0.5)
        efficiency              = np.sum(trap_intensities)
        uniformity              = np.std(deviation_from_target)/np.mean(deviation_from_target)

        print('Initial calculation:')
        print('Efficiency std = %.1f'%(efficiency*100))
        print('Intensity std = %.1f'%(uniformity*100))

        # Start the iterations
        for k in range(GSW_maxit):
            # Calculate new phases and weights
            self.theta_traps  = np.angle(traps_amplitude)
            self.weight_traps *= 1. / rel_dev

            # Calculate new hologram
            phi = self.computeHologram()
            if phi is None:
                return None

            # Calculate new trap intensities
            traps_amplitude         = 1./param.ImgResY/param.ImgResX*slm_to_trap_amp(phi, self.traps_k)
            trap_intensities        = np.real(traps_amplitude*np.conj(traps_amplitude))
            deviation_from_target   = trap_intensities/target_intensities
            rel_dev                 = deviation_from_target**0.5/np.mean(deviation_from_target**0.5)
            efficiency              = np.sum(trap_intensities)
            uniformity              = np.std(deviation_from_target)/np.mean(deviation_from_target)
            print('Efficiency std = %.1f'%(efficiency*100))
            print('Intensity std = %.1f'%(uniformity*100))

        return phi

    def GS_intensity(self,measured_intensities,equalize_gain = 1.):
        '''
        Adjust the weight of each trap according to the measured intensities
        Keep the same phase for each trap
        '''
        if not self._ensure_traps_loaded():
            return None
        self.weight_traps *= measured_intensities**-0.5

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
        Vtraps_it = np.zeros(self.N_traps,dtype=complex)

        phi = self.randomSuperposition()
        if phi is None:
            return None
        print('SR guess created')

        #Measure the trap to trap deviation of intensity
        for k in range(self.N_traps):
            Vtraps_it[k] = 1/param.ImgResY/param.ImgResX*np.sum(np.sum(np.exp(1j *(phi-self.delta_traps[:,:,k])), axis=0), axis=0)
        trap_intensities = np.real(Vtraps_it*np.conj(Vtraps_it))
        efficiency = np.sum(trap_intensities)
        uniformity = np.std(trap_intensities)/np.mean(trap_intensities)
        print('Efficiency std = %.1f'%(efficiency*100))
        print('Intensity std = %.1f'%(uniformity*100))

        for k in range(GAA_maxit):

            print ('Iteration ' + str(k+1) +'/' + str(GAA_maxit))
            Vtraps_it_mod = np.sqrt(Vtraps_it * np.conj(Vtraps_it))
            phi = np.angle(np.sum(
                    np.exp(1j*self.delta_traps)*Vtraps_it/Vtraps_it_mod *(1-self.GAA_xi+self.GAA_xi/Vtraps_it_mod),
                    axis=2
                    ))

            for j in range(self.N_traps):
                Vtraps_it[j] = 1/param.ImgResY/param.ImgResX*np.sum(np.sum(np.exp(1j * (phi-self.delta_traps[:,:,j])),axis=0),axis=0)
            trap_intensities = Vtraps_it*np.conj(Vtraps_it)
            efficiency = np.real(np.sum(trap_intensities))
            uniformity = np.std(trap_intensities)/np.mean(trap_intensities)
            print('Efficiency std = %.1f'%(efficiency*100))
            print('Intensity std = %.1f'%(uniformity*100))

        return phi

    def __del__(self):
        plt.close('all')
        print('closing all')
        self.SLM.close()
        del self.SLM
        del self

    def closeEvent(self, err):
        timer = getattr(self, "_slm_temp_timer", None)
        if timer is not None:
            timer.stop()
        log = getattr(self, "_temperature_log", None)
        if log is not None:
            log.close()
        self._restore_stdio()
        self.__del__()


def register_windows_taskbar_app_id():
    """Set AppUserModelID so the Windows taskbar uses our icon. Call before QApplication()."""
    if sys.platform == 'win32':
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            'Petanque.SLMController.GUI.1'
        )

def application_icon(base_dir=None):
    """Load the GUI icon. Requires QApplication (QIcon may use QPixmap internally)."""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    return QtGui.QIcon(os.path.join(base_dir, 'Icon', 'Petanque array.ico'))


if __name__ == '__main__':
    register_windows_taskbar_app_id()

    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance()

    _app_icon = application_icon()
    app.setWindowIcon(_app_icon)

    slmGui = SLMGUI()
    slmGui.setWindowIcon(_app_icon)
    slmGui.show()
    try :
        ret = app.exec_()
    except Exception as err:
        print('Close event has been raised ',type(err))
    finally :
        del slmGui
