"""SLMGUI main window: shared phase state and dock layout."""
from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any, Callable, Optional

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

if TYPE_CHECKING:
    from gui.hero import SLMHero
    from temperature_log import DailyTemperatureLog


# Fraction of the window height given to the bottom Output (log) dock;
# the remaining height goes to the main "All" dock.
OUTPUT_HEIGHT_FRACTION = 0.10


class SLMMode:
    """Base class for an operating regime of the SLM; decides which panels are shown."""
    name: str = ""


class Static(SLMMode):
    name = "Static"


class Dynamic(SLMMode):
    name = "Dynamic"


class Rearranger(SLMMode):
    name = "Rearranger"


class SLMGUI(QtWidgets.QMainWindow):
    # --- Construction / device ---
    parent: Any                       # optional owner passed to __init__ (only set if given)
    _selected_SLM: str
    _simulate_slm: bool
    SLM: SLMDevice
    SLM_shape: tuple[int, int]        # (Ny, Nx)
    param_P: int                      # gray levels per 2π (from SLM parameters)

    # --- Remote control / state saving ---
    _remote_control: dict[str, Callable[..., Any]]
    _state_dict: dict[str, Any]
    _hero: Optional[SLMHero]

    # --- SLM coordinate grids ---
    X: np.ndarray                     # (Ny, Nx) x coordinate on the SLM [m]
    Y: np.ndarray                     # (Ny, Nx) y coordinate on the SLM [m]
    i_slm: np.ndarray                 # (Ny, Nx) column index
    _j_slm: np.ndarray                # (Ny, Nx) row index
    deltax: float
    deltay: float
    _epsilon: float

    # --- Phase terms summed in updatePhase ---
    phasePattern: np.ndarray          # (Ny, Nx) pattern from Arbitrary Arrays / Blank [rad]
    im_grating: np.ndarray            # (Ny, Nx) steering grating from Elementary [rad]
    fresnelPhase: np.ndarray          # (Ny, Nx) Fresnel lens from Elementary [rad]
    im_zernike: np.ndarray            # (Ny, Nx) Zernike correction from Aberration [rad]
    totalPhase: np.ndarray            # (Ny, Nx) uint8 frame last sent to the SLM

    # --- Grating ---
    dx: float
    dy: float

    # --- Zernike / aberrations ---
    _zernike_r: np.ndarray            # (Ny, Nx) normalized pupil radius
    _zernike_theta: np.ndarray        # (Ny, Nx) polar angle
    _zernike_pupil_radius: float      # [m]
    _zernike_coefficients: list[float]
    zernike_polynomials: np.ndarray   # (n_poly, Ny, Nx), scaled by 2π
    aberration_label_list: list[str]
    aberration_poly_indices: list[int]
    aberration_n_primary: int
    aberration_poly_offset: int
    aberration_values: np.ndarray     # one value per aberration label

    # --- Traps / arbitrary arrays ---
    N_traps: int
    traps_dict: dict[str, Any]        # what Save/Load Hologram pickles
    traps_pos: np.ndarray             # (N, 4): index, y, x, z [m]; [[]] before traps are loaded
    traps_k: np.ndarray               # (N, 3) k-vectors; [[]] before traps are loaded
    theta_traps: np.ndarray           # (N,) trap phases; [] before traps are loaded
    weight_traps: np.ndarray          # (N,) WGS weights (set by Arbitrary Arrays)
    pupil_traps: dict[str, np.ndarray]  # per-trap pupil axes / angle (set by Arbitrary Arrays)
    _weigth_traps: list               # unused (misspelled; weight_traps is the real one)
    _theta_traps_list: list
    _weight_traps_list: list
    k_dump: np.ndarray                # (2,)
    file: str                         # last hologram file loaded / saved

    # --- TEM01 / SO parameters ---
    spacing: float
    theta_deg: float
    r_pi: float

    # --- Algorithm settings ---
    _equalize_iter: int
    equalize_gain: float
    GS_maxit: int
    GAA_xi: float
    methodList: dict[str, Callable[[], Optional[np.ndarray]]]
    std: float
    best_hologram: np.ndarray

    # --- Layout / panels ---
    area: dockarea.DockArea
    _d1: dockarea.Dock
    _global_layout: pg.LayoutWidget
    elementary: ElementaryControlPanel
    _advanced: AdvancedControlPanel
    aberration: AberrationControlPanel
    _arbitrary: ArbitraryArraysPanel
    _preview: LastSentPreviewPanel
    _output: OutputDock
    _dock_ratio_user_set: bool
    _dock_splitter_connected: bool

    # --- Aliases of panel widgets ---
    _ManufacturerCorrectionChkBx: QtWidgets.QCheckBox
    _zernikeCorrectionChkBx: QtWidgets.QCheckBox
    _pupilChkBx: QtWidgets.QCheckBox
    pupilEdt: QtWidgets.QLineEdit
    pupilCenterEdt_x0: QtWidgets.QLineEdit
    pupilCenterEdt_y0: QtWidgets.QLineEdit
    pupilOutLineEdit: QtWidgets.QLineEdit
    aber_lineEdit_list: list[QtWidgets.QLineEdit]
    _last_sent_plot: pg.PlotWidget
    _last_sent_img_item: pg.ImageItem
    _last_sent_colorbar: pg.ColorBarItem
    _label_slm_temperature: QtWidgets.QLabel

    # --- stdout / stderr redirection to the Output log ---
    _stdout_prev: Any
    _stderr_prev: Any
    _stdout_tee: StreamToPlainText
    _stderr_tee: StreamToPlainText

    # --- Set on this object by the panels (not in __init__) ---
    # Output dock (gui/panels/output.py)
    d_output: dockarea.Dock
    output_log: QtWidgets.QPlainTextEdit
    btn_clear_output_log: QtWidgets.QPushButton
    # Advanced panel (gui/panels/advanced.py)
    grating_out: np.ndarray
    pupil: np.ndarray
    pupil_type: str
    phi_tem01: np.ndarray
    phi_ovula: np.ndarray
    phi_SO: np.ndarray
    phi_LG: np.ndarray
    angularM: int
    ImgCenterX: int
    ImgCenterY: int
    # Arbitrary Arrays panel (gui/panels/arbitrary_arrays.py)
    map: Any
    _trap_plot_window: pg.PlotWidget
    _trap_plot_count: int
    # Preview panel (gui/panels/preview.py)
    _temperature_log: DailyTemperatureLog
    _slm_temp_timer: QtCore.QTimer
    _temp_log_error_reported: bool

    def __init__(self, parent=None, selected_SLM=DEFAULT_SLM, simulate=False, start_hero=False, hero_name=None):
        super(SLMGUI, self).__init__()

        if parent:
            self.parent = parent

        self._selected_SLM = selected_SLM
        self._simulate_slm = bool(simulate)
        self._remote_control = {}

        self._state_dict = {}

        self.initUI()
        self.initSLM()

        self._last_sent_plot.installEventFilter(self._preview)
        self._preview.ft_plot.installEventFilter(self._preview)
        self._preview._start_slm_temperature_monitor()

        self.std = 10
        self.best_hologram = np.zeros((600, 792))

        self._wire_remote_control()

        self._hero = None
        if start_hero:
            self._start_hero(hero_name)

    def _start_hero(self, name=None):
        """Expose grating shift + Zernike coefficients over HEROS (optional dependency)."""
        try:
            from gui.hero import HERO_NAME, SLMHero
        except ImportError as e:
            print("HERO disabled (heros not installed):", e)
            return
        if name is None:
            name = HERO_NAME + ("_sim" if self._simulate_slm else "")
        try:
            self._hero = SLMHero(self, name)
            print("HERO started:", name)
        except Exception as e:
            print("HERO could not be started:", e)

    def _stop_hero(self):
        hero, self._hero = getattr(self, "_hero", None), None
        if hero is not None:
            hero.__exit__(None, None, None)

    def _publish_static_changes(self):
        """Tell HERO listeners (the sequencer) about grating / Zernike changes."""
        hero = getattr(self, "_hero", None)
        if hero is None:
            return
        try:
            hero._publish_static_changes()
        except Exception as e:
            print("HERO static_changed publish failed:", e)

    def _wire_remote_control(self):
        self._remote_control = {
            "SLM1 x": self.elementary.updateHorizontalGrating,
            "SLM1 y": self.elementary.updateVerticalGrating,
            "SLM1 pupil 1st axis": self._advanced.apply_pupil_longaxis_byatom_loop,
            "SLM1 pupil 2st axis": self._advanced.apply_pupil_shortaxis_byatom_loop,
            "SLM1: defocus": self.aberration.updateAberration_defocus,
            "SLM1: asti0": self.aberration.updateAberration_asti0,
            "SLM1: asti45": self.aberration.updateAberration_asti45,
            "SLM1: coma0": self.aberration.updateAberration_coma0,
            "SLM1: coma90": self.aberration.updateAberration_coma90,
            "SLM1: spherical": self.aberration.updateAberration_spherical,
            "SLM1: TM01 spacing": self._advanced.createTEM01trap,
            "SLM1: TM01 angle(degree)": self._advanced.createTEM01trap_angle,
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
            self._state_dict[instance_name] = self.__dict__[instance_name]

        self._state_dict["TEM01_check"] = self.tem01ChkBx.isChecked()
        self._state_dict["SO_check"] = self.SOChkBx.isChecked()
        self._state_dict["LG_check"] = self.LGChkBx.isChecked()

    def dict_to_state(self, memory_dict):
        try:
            self.elementary.createGratingPattern(dx=memory_dict["dx"], dy=memory_dict["dy"])
            self.aberration.updateAberration(aber_list=memory_dict["aberration_values"])
            self._arbitrary.loadPattern(name=memory_dict["file"])

            self.tem01spacingEdt.setText(str(memory_dict["spacing"]))
            self.tem01ThetaEdt.setText(str(memory_dict["theta_deg"]))
            self.tem01ChkBx.setChecked(memory_dict["TEM01_check"])
            self.SOrpiEdt.setText(str(memory_dict["r_pi"]))
            self.SOChkBx.setChecked(memory_dict["SO_check"])
            self.LGChkBx.setChecked(memory_dict["LG_check"])
        except Exception as e:
            print(e)

    def initSLM(self):
        self.SLM = SLMDevice(self._selected_SLM, simulate=self._simulate_slm)

        if getattr(self.SLM, "simulated", False):
            self.setWindowTitle("SLM Controller Petanque [SIMULATE — no hardware]")

        Nx, Ny = self.SLM.getSize()

        self.SLM_shape = (Ny, Nx)

        self.X, self.Y = np.meshgrid(
            (np.arange(0, Nx, 1) - Nx / 2) * param.pslm,
            (np.arange(0, Ny, 1) - Ny / 2) * param.pslm,
        )

        self.i_slm, self._j_slm = np.meshgrid(
            np.arange(Nx),
            np.arange(Ny),
        )

        self.deltax = 1
        self.deltay = 1
        self._epsilon = 0
        self.param_P = self.SLM.parameters.P

        self.phasePattern = np.zeros([param.ImgResY, param.ImgResX])
        self.im_grating = np.zeros([param.ImgResY, param.ImgResX])
        self.fresnelPhase = np.zeros([param.ImgResY, param.ImgResX])

        self._zernike_r = (self.X ** 2 + self.Y ** 2) ** 0.5
        self._zernike_theta = np.arctan2(self.X, self.Y)

        self._zernike_pupil_radius = param.max_radius
        self._zernike_r = (self.X ** 2 + self.Y ** 2) ** 0.5 / self._zernike_pupil_radius

        self._zernike_coefficients = np.zeros(32).tolist()

        self.im_zernike = np.zeros([param.ImgResY, param.ImgResX], dtype=np.float64)

        self.zernike_polynomials = np.asarray(
            __zernikepolynomials__(self._zernike_r, self._zernike_theta)
        )
        self.zernike_polynomials[:, self._zernike_r > 1] = 0
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
        self._weigth_traps = []
        self.k_dump = np.array([40, 0]) * 1e-6 * position_to_step_per_pixel

        self.spacing = 5.0
        self.theta_deg = -4.0
        self.r_pi = 0.5

        self._theta_traps_list = []
        self._weight_traps_list = []

        self.file = ""

        self._equalize_iter = 5
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
        # The Output/main height split is applied in resizeEvent (see there).
        self._dock_ratio_user_set = False
        self._dock_splitter_connected = False

        # Algorithms registered after the arbitrary-arrays panel exists
        self.methodList = {
            "WGS SLMcorr": lambda: self._arbitrary.WGS(correct_diffraction_efficiency=True),
            "WGS": lambda: self._arbitrary.WGS(correct_diffraction_efficiency=False),
        }
        self._arbitrary.methodCB.clear()
        for key in self.methodList:
            self._arbitrary.methodCB.addItem(key)

    def createDocks(self):
        self._d1 = dockarea.Dock("All", size=(700, 700))
        self.area.addDock(self._d1, "left")
        self._d1.hideTitleBar()

        self._global_layout = pg.LayoutWidget()

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

        self._advanced = AdvancedControlPanel(self)
        adv_frame = framed_panel(self._advanced)
        adv_frame.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum
        )
        left_lay.addWidget(adv_frame)
        # Aliases for updatePhase / cross-panel reads
        self._ManufacturerCorrectionChkBx = self._advanced.ManufacturerCorrectionChkBx
        self._zernikeCorrectionChkBx = self._advanced.zernikeCorrectionChkBx
        self._pupilChkBx = self._advanced.pupilChkBx
        self.pupilEdt = self._advanced.pupilEdt
        self.pupilCenterEdt_x0 = self._advanced.pupilCenterEdt_x0
        self.pupilCenterEdt_y0 = self._advanced.pupilCenterEdt_y0
        self.pupilOutLineEdit = self._advanced.pupilOutLineEdit

        self.aberration = AberrationControlPanel(self)
        aber_frame = framed_panel(self.aberration)
        aber_frame.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum
        )
        left_lay.addWidget(aber_frame)
        left_lay.addStretch(1)

        self.aber_lineEdit_list = self.aberration.aber_lineEdit_list

        self._global_layout.addWidget(left_col, row=0, col=0, rowspan=2)

        self._arbitrary = ArbitraryArraysPanel(self)
        self._global_layout.addWidget(framed_panel(self._arbitrary), row=0, col=1)

        self._preview = LastSentPreviewPanel(self)
        self._global_layout.addWidget(framed_panel(self._preview), row=1, col=1)
        self._last_sent_plot = self._preview.last_sent_plot
        self._last_sent_img_item = self._preview.last_sent_img_item
        self._last_sent_colorbar = self._preview.last_sent_colorbar
        self._label_slm_temperature = self._preview.label_slm_temperature

        self._d1.addWidget(self._global_layout, row=0, col=0)

        self._output = OutputDock(self)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Re-apply the Output/main height split on every window resize until
        # the user drags the splitter themselves. Deferred so the dock
        # splitter has already been resized to the new window height.
        if hasattr(self, "d_output") and not self._dock_ratio_user_set:
            QtCore.QTimer.singleShot(0, self._apply_dock_ratio)

    def _on_dock_splitter_moved(self, *_):
        self._dock_ratio_user_set = True

    def _apply_dock_ratio(self):
        """Give the Output dock OUTPUT_HEIGHT_FRACTION of the dock area height
        and the rest to the docks above it."""
        splitter = self.d_output.container()
        if not isinstance(splitter, QtWidgets.QSplitter):
            return
        if not self._dock_splitter_connected:
            splitter.splitterMoved.connect(self._on_dock_splitter_moved)
            self._dock_splitter_connected = True
        i_out = splitter.indexOf(self.d_output)
        sizes = splitter.sizes()
        total = sum(sizes)
        if i_out < 0 or len(sizes) < 2 or total <= 0:
            return
        out_h = int(round(total * OUTPUT_HEIGHT_FRACTION))
        # The main dock may be wrapped in its own container, so give the
        # remaining height to all other splitter entries proportionally.
        others = [i for i in range(len(sizes)) if i != i_out]
        other_total = sum(sizes[i] for i in others) or len(others)
        rest = total - out_h
        new_sizes = list(sizes)
        new_sizes[i_out] = out_h
        for i in others:
            new_sizes[i] = int(rest * (sizes[i] or 1) / other_total)
        splitter.setSizes(new_sizes)

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

        if self._ManufacturerCorrectionChkBx.isChecked():
            self.totalPhase = self.totalPhase + self.SLM.correction_pattern
        if self._zernikeCorrectionChkBx.isChecked():
            self.totalPhase = self.totalPhase + self.im_zernike
        if self._pupilChkBx.isChecked():
            self._advanced.applyPupil()

        self.totalPhase = np.mod(self.totalPhase, 2 * np.pi)

        self.totalPhase = np.round(self.param_P * self.totalPhase / (2 * np.pi)).astype("uint8")
        self.SLM.updateArray(self.totalPhase)
        self._publish_static_changes()
        self._preview._update_last_sent_preview()

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
        self._stop_hero()
        self._restore_stdio()
        self.__del__()
