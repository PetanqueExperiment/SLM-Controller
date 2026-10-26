"""SLMGUI main window: shared phase state and dock layout."""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph.dockarea as dockarea

from SLM_device import SLMDevice
from gui.constants import DEFAULT_SLM, position_to_step_per_pixel
from gui.helpers import framed_panel
from gui.stdio_tee import StreamToPlainText
from gui.panels.elementary import ElementaryControlPanel
from gui.panels.advanced import AdvancedControlPanel
from gui.panels.aberration import AberrationControlPanel
from gui.panels.arbitrary_arrays import ArbitraryArraysPanel
from gui.panels.preview import LastSentPreviewPanel
from gui.panels.output import OutputDock

try:
    from zernike import __zernikepolar__, __zernikepolynomials__
except ImportError:
    from zernike import __zernikepolar__, __zernikepolynomials__

import Meadowlark_SLM.parameters as param


class SLMGUI(QtWidgets.QMainWindow):

    def __init__(self, parent=None, selected_SLM=DEFAULT_SLM, simulate=False):
        super(SLMGUI, self).__init__()

        if parent:
            self.parent = parent

        self.selected_SLM = selected_SLM
        self.simulate_slm = bool(simulate)
        self.remote_control = {}

        self.state_dict = {}

        self.initUI()
        self.initSLM()

        self.last_sent_plot.installEventFilter(self.preview)
        self.preview.ft_plot.installEventFilter(self.preview)
        self.preview._start_slm_temperature_monitor()

        self.std = 10
        self.best_hologram = np.zeros((600, 792))

        self._wire_remote_control()

    def _wire_remote_control(self):
        self.remote_control = {
            "SLM1 x": self.elementary.updateHorizontalGrating,
            "SLM1 y": self.elementary.updateVerticalGrating,
            "SLM1 pupil 1st axis": self.advanced.apply_pupil_longaxis_byatom_loop,
            "SLM1 pupil 2st axis": self.advanced.apply_pupil_shortaxis_byatom_loop,
            "SLM1: defocus": self.aberration.updateAberration_defocus,
            "SLM1: asti0": self.aberration.updateAberration_asti0,
            "SLM1: asti45": self.aberration.updateAberration_asti45,
            "SLM1: coma0": self.aberration.updateAberration_coma0,
            "SLM1: coma90": self.aberration.updateAberration_coma90,
            "SLM1: spherical": self.aberration.updateAberration_spherical,
            "SLM1: TM01 spacing": self.advanced.createTEM01trap,
            "SLM1: TM01 angle(degree)": self.advanced.createTEM01trap_angle,
            "SLM1: trefoil0": self.aberration.updateAberration_trefoil0,
            "SLM1: trefoil90": self.aberration.updateAberration_trefoil90,
            "SLM1: asti2 0": self.aberration.updateAberration_asti2_0,
            "SLM1: trefoil2 0": self.aberration.updateAberration_trefoil2_0,
            "SLM1: trefoil2 90": self.aberration.updateAberration_trefoil2_90,
        }

    def state_to_dict(self):
        save_list = [
            "dx",
            "dy",
            "aberration_values",
            "file",
            "spacing",
            "theta_deg",
            "r_pi",
        ]

        for instance_name in save_list:
            self.state_dict[instance_name] = self.__dict__[instance_name]

        self.state_dict["TEM01_check"] = self.tem01ChkBx.isChecked()
        self.state_dict["SO_check"] = self.SOChkBx.isChecked()
        self.state_dict["LG_check"] = self.LGChkBx.isChecked()

    def dict_to_state(self, memory_dict):
        try:
            self.elementary.createGratingPattern(dx=memory_dict["dx"], dy=memory_dict["dy"])
            self.aberration.updateAberration(aber_list=memory_dict["aberration_values"])
            self.arbitrary.loadPattern(name=memory_dict["file"])

            self.tem01spacingEdt.setText(str(memory_dict["spacing"]))
            self.tem01ThetaEdt.setText(str(memory_dict["theta_deg"]))
            self.tem01ChkBx.setChecked(memory_dict["TEM01_check"])
            self.SOrpiEdt.setText(str(memory_dict["r_pi"]))
            self.SOChkBx.setChecked(memory_dict["SO_check"])
            self.LGChkBx.setChecked(memory_dict["LG_check"])
        except Exception as e:
            print(e)

    def initSLM(self):
        self.SLM = SLMDevice(self.selected_SLM, simulate=self.simulate_slm)

        if getattr(self.SLM, "simulated", False):
            self.setWindowTitle("SLM Controller Petanque [SIMULATE — no hardware]")

        Nx, Ny = self.SLM.getSize()

        self.SLM_shape = (Ny, Nx)

        self.X, self.Y = np.meshgrid(
            (np.arange(0, Nx, 1) - Nx / 2) * param.pslm,
            (np.arange(0, Ny, 1) - Ny / 2) * param.pslm,
        )

        self.i_slm, self.j_slm = np.meshgrid(
            np.arange(Nx),
            np.arange(Ny),
        )

        self.deltax = 1
        self.deltay = 1
        self.epsilon = 0
        self.param_P = self.SLM.parameters.P

        self.phasePattern = np.zeros([param.ImgResY, param.ImgResX])
        self.im_grating = np.zeros([param.ImgResY, param.ImgResX])
        self.fresnelPhase = np.zeros([param.ImgResY, param.ImgResX])

        self.zernike_r = (self.X ** 2 + self.Y ** 2) ** 0.5
        self.zernike_theta = np.arctan2(self.X, self.Y)

        self.zernike_pupil_radius = param.max_radius
        self.zernike_r = (self.X ** 2 + self.Y ** 2) ** 0.5 / self.zernike_pupil_radius

        self.zernike_coefficients = np.zeros(32).tolist()

        self.im_zernike = np.zeros([param.ImgResY, param.ImgResX], dtype=np.float64)

        self.zernike_polynomials = np.asarray(
            __zernikepolynomials__(self.zernike_r, self.zernike_theta)
        )
        self.zernike_polynomials[:, self.zernike_r > 1] = 0
        self.zernike_polynomials *= 2 * np.pi

        self.elementary.createGratingPattern(self.dx, self.dy)
        self.updatePhase()

    def initUI(self):
        self.dx = 0
        self.dy = 0

        # Tip/tilt (Zernike indices 0–1) omitted from the UI.
        # Poly indices skip tetrafoil (15–16) so trefoil 2nd maps to Z19/Z20.
        self.aberration_label_list = [
            "defocus",
            "asti 0",
            "asti 45",
            "coma 0",
            "coma 90",
            "primary spherical",
            "trefoil 0",
            "trefoil 90",
            "asti 2nd 0",
            "asti 2nd 45",
            "coma 2nd 0",
            "coma 2nd 90",
            "2nd spherical",
            "trefoil 2nd 0",
            "trefoil 2nd 90",
        ]
        self.aberration_poly_indices = [
            2, 3, 4, 5, 6, 7, 8, 9,  # defocus … trefoil
            10, 11, 12, 13, 14,  # asti/coma 2nd, 2nd spherical
            17, 18,  # trefoil 2nd (Z19/Z20); skip tetrafoil 15–16
        ]
        # First N labels go in the left column; the rest are 2nd-order (right column).
        self.aberration_n_primary = 8
        self.aberration_poly_offset = 2  # fallback if poly_indices missing
        self.aberration_values = np.zeros(len(self.aberration_label_list))

        self.N_traps = 30
        self.traps_dict = {}
        self.traps_pos = [[]]
        self.traps_k = [[]]
        self.theta_traps = []
        self.weigth_traps = []
        self.k_dump = np.array([40, 0]) * 1e-6 * position_to_step_per_pixel

        self.spacing = 5.0
        self.theta_deg = -4.0
        self.r_pi = 0.5

        self.theta_traps_list = []
        self.weight_traps_list = []

        self.file = ""

        self.equalize_iter = 5
        self.equalize_gain = 1
        self.GS_maxit = 5
        self.GAA_xi = 0.5

        # Placeholder until panels exist; filled after createDocks.
        self.methodList = {}

        self.area = dockarea.DockArea()
        self.setCentralWidget(self.area)
        self.resize(1200, 720)
        self.setWindowTitle("SLM Controller Petanque")
        self.createDocks()
        self._install_gui_stdio()

        # Algorithms registered after the arbitrary-arrays panel exists
        self.methodList = {
            "WGS SLMcorr": lambda: self.arbitrary.WGS(correct_diffraction_efficiency=True),
        }
        self.arbitrary.methodCB.clear()
        for key in self.methodList:
            self.arbitrary.methodCB.addItem(key)

    def createDocks(self):
        self.d1 = dockarea.Dock("All", size=(700, 700))
        self.area.addDock(self.d1, "left")
        self.d1.hideTitleBar()

        self.global_layout = pg.LayoutWidget()

        # Left column in its own VBox so Elementary is not stretched to the
        # Arbitrary Arrays row height in the outer grid.
        left_col = QtWidgets.QWidget()
        left_lay = QtWidgets.QVBoxLayout(left_col)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(4)

        self.elementary = ElementaryControlPanel(self)
        elem_frame = framed_panel(self.elementary)
        elem_frame.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum
        )
        left_lay.addWidget(elem_frame)

        self.advanced = AdvancedControlPanel(self)
        adv_frame = framed_panel(self.advanced)
        adv_frame.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum
        )
        left_lay.addWidget(adv_frame)
        # Aliases for updatePhase / cross-panel reads
        self.ManufacturerCorrectionChkBx = self.advanced.ManufacturerCorrectionChkBx
        self.zernikeCorrectionChkBx = self.advanced.zernikeCorrectionChkBx
        self.pupilChkBx = self.advanced.pupilChkBx
        self.pupilEdt = self.advanced.pupilEdt
        self.pupilCenterEdt_x0 = self.advanced.pupilCenterEdt_x0
        self.pupilCenterEdt_y0 = self.advanced.pupilCenterEdt_y0
        self.pupilOutLineEdit = self.advanced.pupilOutLineEdit

        self.aberration = AberrationControlPanel(self)
        aber_frame = framed_panel(self.aberration)
        aber_frame.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum
        )
        left_lay.addWidget(aber_frame)
        left_lay.addStretch(1)

        self.aber_lineEdit_list = self.aberration.aber_lineEdit_list

        self.global_layout.addWidget(left_col, row=0, col=0, rowspan=2)

        self.arbitrary = ArbitraryArraysPanel(self)
        self.global_layout.addWidget(framed_panel(self.arbitrary), row=0, col=1)

        self.preview = LastSentPreviewPanel(self)
        self.global_layout.addWidget(framed_panel(self.preview), row=1, col=1)
        self.last_sent_plot = self.preview.last_sent_plot
        self.last_sent_img_item = self.preview.last_sent_img_item
        self.last_sent_colorbar = self.preview.last_sent_colorbar
        self.label_slm_temperature = self.preview.label_slm_temperature

        self.d1.addWidget(self.global_layout, row=0, col=0)

        self.output = OutputDock(self)

    def _install_gui_stdio(self):
        self._stdout_prev = sys.stdout
        self._stderr_prev = sys.stderr
        self._stdout_tee = StreamToPlainText(self._stdout_prev, self.output_log, parent=self)
        self._stderr_tee = StreamToPlainText(self._stderr_prev, self.output_log, parent=self)
        sys.stdout = self._stdout_tee
        sys.stderr = self._stderr_tee

    def _restore_stdio(self):
        if getattr(self, "_stdout_prev", None) is not None:
            sys.stdout = self._stdout_prev
        if getattr(self, "_stderr_prev", None) is not None:
            sys.stderr = self._stderr_prev

    def adjustImageForS1028(self, array):
        v_s, h_s = 1024, 1272
        v, h = array.shape
        converted_array = np.zeros([v_s, h_s])
        converted_array[(v_s - v) // 2 : (v_s + v) // 2, (h_s - h) // 2 : (h_s + h) // 2] = np.fliplr(
            array
        )
        return converted_array

    def updatePhase(self):
        self.totalPhase = self.im_grating + self.fresnelPhase + self.phasePattern

        if self.ManufacturerCorrectionChkBx.isChecked():
            self.totalPhase = self.totalPhase + self.SLM.correction_pattern
        if self.zernikeCorrectionChkBx.isChecked():
            self.totalPhase = self.totalPhase + self.im_zernike
        if self.pupilChkBx.isChecked():
            self.advanced.applyPupil()

        self.totalPhase = np.mod(self.totalPhase, 2 * np.pi)

        self.totalPhase = np.round(self.param_P * self.totalPhase / (2 * np.pi)).astype("uint8")
        self.SLM.updateArray(self.totalPhase)
        self.preview._update_last_sent_preview()

    def __del__(self):
        plt.close("all")
        print("closing all")
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
