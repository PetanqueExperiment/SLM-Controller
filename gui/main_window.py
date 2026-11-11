"""SLMGUI main window: shared phase state and dock layout."""

from __future__ import annotations

import filecmp
import os
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph.dockarea as dockarea

from libslm.slm import MeadowlarkSLM, SerialMismatchError, loadCorrectionPattern
from gui.defaults import DEFAULT_PROFILE, DEVICES_DIR, DEVICES_FILE, PROFILE_TEMPLATE_FILE, STARTUP_MODES
from gui.settings import PROFILE_LOADED, PROFILE_MISSING, SIMULATED_BOARD, AppSettings, Device, deviceLabel, factoryLutPath, setDeviceFactoryLut, ProfileTemplateError, addDevice, findDevice, isValidProfileName, listProfiles, loadDevices, loadProfileParams, parseBoard, profilePath, restoreDefaultProfile, setDeviceSerial, writeProfile
from gui.helpers import framed_panel
from gui.stdio_tee import StreamToPlainText
from gui.panels.elementary import ElementaryControlPanel
from gui.panels.advanced import AdvancedControlPanel
from gui.panels.aberration import AberrationControlPanel
from gui.panels.arbitrary_arrays import ArbitraryArraysPanel
from gui.panels.preview import LastSentPreviewPanel
from gui.panels.output import OutputDock
from gui.panels.slm_info import SlmInfoDock
from gui.zernike import __zernikepolar__, __zernikepolynomials__

import libslm.parameters as param

if TYPE_CHECKING:
    from gui.hero import SLMHero
    from gui.temperature_log import DailyTemperatureLog


# Fraction of the window height given to the bottom Output (log) dock and to the top SLM info dock;
# the remaining height goes to the main "All" dock.
OUTPUT_HEIGHT_FRACTION: float = 0.10
SLM_INFO_HEIGHT_FRACTION: float = 0.02

# Gap between panels [px]: between the left-column panels, between the docks (splitter handles), to the window edges
# (left, right, bottom, below the top menu)
# and between the profile / SLM panels of the info strip.
PANEL_GAP_PX: int = 4

WINDOW_TITLE: str = "SLM Controller"


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
    parent: Any  # optional owner passed to __init__ (only set if given)
    SLM: Optional[MeadowlarkSLM]  # None when no SLM is connected
    factory_lut: Optional[np.ndarray]  # factory correction of the connected SLM [rad] (devices.json factory_lut); None = none
    board: Optional[int]  # Blink PCIe board number of the SLM (None = no SLM chosen); stored in the profile
    SLM_shape: tuple[int, int]  # (Ny, Nx)
    param_P: float  # gray levels per 2π (profile: grating.phase_levels)
    wavelength_m: float  # laser wavelength [m] (profile: grating.wavelength_nm); converts µm in the focal plane to SLM phase

    # --- Remote control / state saving ---
    _remote_control: dict[str, Callable[..., Any]]
    _state_dict: dict[str, Any]
    _hero: Optional[SLMHero]

    # --- SLM coordinate grids ---
    X: np.ndarray  # (Ny, Nx) x coordinate on the SLM [m]
    Y: np.ndarray  # (Ny, Nx) y coordinate on the SLM [m]
    i_slm: np.ndarray  # (Ny, Nx) column index
    _j_slm: np.ndarray  # (Ny, Nx) row index
    deltax: float
    deltay: float
    _epsilon: float

    # --- Phase terms summed in updatePhase ---
    phasePattern: np.ndarray  # (Ny, Nx) pattern from Arbitrary Arrays / Blank [rad]
    im_grating: np.ndarray  # (Ny, Nx) steering grating from Elementary [rad]
    fresnelPhase: np.ndarray  # (Ny, Nx) Fresnel lens from Elementary [rad]
    im_zernike: np.ndarray  # (Ny, Nx) Zernike correction from Aberration [rad]
    totalPhase: np.ndarray  # (Ny, Nx) uint8 frame last sent to the SLM

    # --- Grating ---
    dx: float
    dy: float

    # --- Zernike / aberrations ---
    _zernike_r: np.ndarray  # (Ny, Nx) normalized pupil radius
    _zernike_theta: np.ndarray  # (Ny, Nx) polar angle
    _zernike_pupil_radius: float  # [m]
    _zernike_coefficients: list[float]
    zernike_polynomials: np.ndarray  # (n_poly, Ny, Nx), scaled by 2π
    aberration_label_list: list[str]
    aberration_poly_indices: list[int]
    aberration_n_primary: int
    aberration_poly_offset: int
    aberration_values: np.ndarray  # one value per aberration label

    # --- Traps / arbitrary arrays ---
    N_traps: int
    traps_dict: dict[str, Any]  # what Save/Load Hologram pickles
    traps_pos: np.ndarray  # (N, 4): index, y, x, z [m]; [[]] before traps are loaded
    traps_k: np.ndarray  # (N, 3) k-vectors; [[]] before traps are loaded
    theta_traps: np.ndarray  # (N,) trap phases; [] before traps are loaded
    weight_traps: np.ndarray  # (N,) WGS weights (set by Arbitrary Arrays)
    pupil_traps: dict[str, np.ndarray]  # per-trap pupil axes / angle (set by Arbitrary Arrays)
    _weigth_traps: list  # unused (misspelled; weight_traps is the real one)
    _theta_traps_list: list
    _weight_traps_list: list
    k_dump: np.ndarray  # (2,)
    file: str  # last hologram file loaded / saved

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
    _slm_info: SlmInfoDock  # top dock: board, serial number, connection state
    _dock_ratio_user_set: bool
    _dock_splitter_connected: bool
    top_menu_bar: QtWidgets.QToolBar
    top_menu_buttons: dict[str, QtWidgets.QPushButton]
    _startup_mode_group: QtWidgets.QActionGroup
    _connect_action: QtWidgets.QAction  # Device ▸ Connect… (enabled only while no SLM is connected)
    _disconnect_action: QtWidgets.QAction  # Device ▸ Disconnect (enabled only while an SLM is connected)
    _set_factory_lut_action: QtWidgets.QAction  # Device ▸ Set factory LUT… (enabled only for a real, selected SLM)

    # --- Settings / profiles ---
    settings: AppSettings  # settings/settings.json
    current_profile: str  # name of the active profile (profiles/<name>.json)
    _suspend_phase_updates: bool  # True while a profile is applied: updatePhase() waits until all values are set

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
    # SLM info dock (gui/panels/slm_info.py)
    d_slm_info: dockarea.Dock
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
    # Arbitrary Arrays panel (gui/panels/arbitrary_arrays/)
    map: Any
    _trap_plot_window: pg.PlotWidget
    _trap_plot_count: int
    # Preview panel (gui/panels/preview.py)
    _temperature_log: DailyTemperatureLog
    _slm_temp_timer: QtCore.QTimer
    _temp_log_error_reported: bool

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, start_hero: bool = False, hero_name: Optional[str] = None) -> None:
        super(SLMGUI, self).__init__()

        if parent:
            self.parent = parent

        self._remote_control = {}

        self._state_dict = {}

        self.settings = AppSettings.load()
        self.current_profile = self.settings.startupProfile()
        self.board = None
        self.SLM = None
        self.factory_lut = None
        self._suspend_phase_updates = False

        self.initUI()
        self.initPhase()
        self.loadProfile(self.current_profile)
        self._updateWindowTitle()

        self._last_sent_plot.installEventFilter(self._preview)
        self._preview.ft_plot.installEventFilter(self._preview)
        self._preview._start_slm_temperature_monitor()

        self.std = 10
        self.best_hologram = np.zeros((600, 792))

        self._wire_remote_control()

        self._hero = None
        if start_hero:
            self._start_hero(hero_name)

        QtCore.QTimer.singleShot(0, self._connectAtStartup)

    def _connectAtStartup(self) -> None:
        """Last startup step, run once the window is up: connect to the SLM stored in the profile, if any."""
        if self.board is None:
            print("No SLM in the profile: running without an SLM (Device ▸ Connect… to connect)")
            return
        self.connectSLM(self.board)

    def _start_hero(self, name: Optional[str] = None) -> None:
        """Expose grating shift + Zernike coefficients over HEROS (optional dependency)."""
        try:
            from gui.hero import HERO_NAME, SLMHero
        except ImportError as e:
            print("HERO disabled (heros not installed):", e)
            return
        if name is None:
            name = HERO_NAME + ("_sim" if self.board == SIMULATED_BOARD else "")
        try:
            self._hero = SLMHero(self, name)
            print("HERO started:", name)
        except Exception as e:
            print("HERO could not be started:", e)

    def _stop_hero(self) -> None:
        hero, self._hero = getattr(self, "_hero", None), None
        if hero is not None:
            hero.__exit__(None, None, None)

    def _publish_static_changes(self) -> None:
        """Tell HERO listeners (the sequencer) about grating / Zernike changes."""
        hero = getattr(self, "_hero", None)
        if hero is None:
            return
        try:
            hero._publish_static_changes()
        except Exception as e:
            print("HERO static_changed publish failed:", e)

    def _wire_remote_control(self) -> None:
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

    def state_to_dict(self) -> None:
        save_list = ["dx", "dy", "aberration_values", "file", "spacing", "theta_deg", "r_pi"]

        for instance_name in save_list:
            self._state_dict[instance_name] = self.__dict__[instance_name]

        self._state_dict["TEM01_check"] = self.tem01ChkBx.isChecked()
        self._state_dict["SO_check"] = self.SOChkBx.isChecked()
        self._state_dict["LG_check"] = self.LGChkBx.isChecked()

    def dict_to_state(self, memory_dict: dict[str, Any]) -> None:
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

    def connectSLM(self, board: int) -> bool:
        """Connect to the SLM on ``board`` and upload the current hologram. On failure the GUI keeps running without an SLM."""
        if self.SLM is not None:
            print(f"Already connected: {self._boardLabel(self.board)}")
            return False
        name = self._boardLabel(board)
        device = findDevice(board)
        expected = None if device is None else device.serial
        try:
            slm = MeadowlarkSLM(board, simulate=board == SIMULATED_BOARD, wfc_path=None, expected_serial=expected, on_serial_mismatch=lambda exp, act: self._confirmSerialOverwrite(board, exp, act))
            slm.connect()
        except SerialMismatchError as e:
            print(f"Connection aborted: {e} Nothing was sent to the SLM.")
            return False
        except Exception as e:
            print(f"Could not connect to {name}: {e}")
            QtWidgets.QMessageBox.warning(self, "SLM not connected", f"Could not connect to {name}:\n{e}\n\nThe GUI runs without an SLM.")
            return False
        self.SLM = slm
        self.board = board
        self._updateWindowTitle()
        print(f"Connected to {name}")
        self._checkSerial(device, slm.serial)
        self._loadFactoryLut(device)
        self.updatePhase()
        return True

    def _loadFactoryLut(self, device: Optional[Device]) -> None:
        """Load the connected SLM's factory correction (devices.json factory_lut → devices/<name>.bmp) into self.factory_lut.

        The Simulated SLM gets none. A real SLM without a usable file gets none either, with a warning.
        """
        self.factory_lut = None
        if self.board == SIMULATED_BOARD:
            print("Simulated SLM: no factory LUT")
            return
        name = "" if device is None else device.factory_lut
        label = self._boardLabel(self.board)
        if not name:
            reason = "no factory LUT is set"
        else:
            path = factoryLutPath(name)
            try:
                self.factory_lut = loadCorrectionPattern(path)
                print(f"{label}: factory LUT {name!r} loaded from {path}")
                return
            except (OSError, ValueError) as e:
                reason = f"factory LUT {name!r} could not be loaded ({e})"
        print(f"{label}: {reason}; frames are sent without manufacturer correction")
        QtWidgets.QMessageBox.warning(self, "No factory LUT", f"{label}: {reason}.\n\nFrames are sent without manufacturer correction.\nSet it with Device ▸ Set factory LUT…")
    def _confirmSerialOverwrite(self, board: int, expected: int, actual: int) -> bool:
        """Serial number mismatch while connecting (before anything is sent to the SLM). Yes: overwrite it in devices.json and connect; No: abort."""
        text = f"The SLM on board {board} is not the one in the device list:\n\n    expected serial number {expected}\n    found serial number {actual}\n\nYes: overwrite the serial number in the device list with {actual} and connect.\nNo: abort the connection (nothing is sent to the SLM)."
        answer = QtWidgets.QMessageBox.warning(self, "Wrong SLM?", text, QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if answer != QtWidgets.QMessageBox.Yes:
            return False
        try:
            setDeviceSerial(board, actual)
            print(f"Serial number of board {board} overwritten in the device list: {expected} -> {actual}")
        except OSError as e:
            print(f"Could not write serial number {actual} to the device list (connecting anyway):", e)
        return True

    def _checkSerial(self, device: Optional[Device], actual: Optional[int]) -> None:
        """After connecting: log the serial check, and record the serial of a listed device the first time it is seen."""
        if self.SLM is not None and self.SLM.simulate:
            print("Simulated SLM: no serial number to check")
        elif actual is None:
            print(f"Could not read the serial number of the SLM on board {self.board}: it was not verified")
        elif device is None:
            print(f"SLM on board {self.board} has serial number {actual} (board not in the device list)")
        elif device.serial is None:
            try:
                setDeviceSerial(device.board, actual)
                print(f"Recorded serial number {actual} for board {device.board} in the device list")
            except OSError as e:
                print(f"Could not record serial number {actual} in the device list:", e)
        elif device.serial == actual:
            print(f"Serial number {actual} matches the device list")

    def _updateWindowTitle(self) -> None:
        """The title is fixed; the profile and SLM state are shown in the info strip at the top, refreshed here."""
        self.setWindowTitle(WINDOW_TITLE)
        slm_info = getattr(self, "_slm_info", None)
        if slm_info is not None:
            slm_info.update()

    def isSLMConnected(self) -> bool:
        """True while an SLM session is open (the Blink SDK has no separate cable/presence check)."""
        return self.SLM is not None and self.SLM.connected

    def initPhase(self) -> None:
        """SLM coordinate grids and phase buffers (independent of whether an SLM is connected)."""
        Nx, Ny = param.ImgResX, param.ImgResY

        self.SLM_shape = (Ny, Nx)

        self.X, self.Y = np.meshgrid(
            (np.arange(0, Nx, 1) - Nx / 2) * param.pslm, (np.arange(0, Ny, 1) - Ny / 2) * param.pslm
        )

        self.i_slm, self._j_slm = np.meshgrid(np.arange(Nx), np.arange(Ny))

        self.deltax = 1
        self.deltay = 1
        self._epsilon = 0

        self.phasePattern = np.zeros([param.ImgResY, param.ImgResX])
        self.im_grating = np.zeros([param.ImgResY, param.ImgResX])
        self.fresnelPhase = np.zeros([param.ImgResY, param.ImgResX])

        self._zernike_r = (self.X**2 + self.Y**2) ** 0.5
        self._zernike_theta = np.arctan2(self.X, self.Y)

        self._zernike_pupil_radius = param.max_radius
        self._zernike_r = (self.X**2 + self.Y**2) ** 0.5 / self._zernike_pupil_radius

        self._zernike_coefficients = np.zeros(32).tolist()

        self.im_zernike = np.zeros([param.ImgResY, param.ImgResX], dtype=np.float64)

        self.zernike_polynomials = np.asarray(__zernikepolynomials__(self._zernike_r, self._zernike_theta))
        self.zernike_polynomials[:, self._zernike_r > 1] = 0
        self.zernike_polynomials *= 2 * np.pi

    def initUI(self) -> None:
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
        # defocus … trefoil, asti/coma 2nd, 2nd spherical, trefoil 2nd (Z19/Z20); skip tetrafoil 15–16
        self.aberration_poly_indices = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18]
        # First N labels go in the left column; the rest are 2nd-order (right column).
        self.aberration_n_primary = 8
        self.aberration_poly_offset = 2  # fallback if poly_indices missing

        self.N_traps = 30
        self.traps_dict = {}
        self.traps_pos = [[]]
        self.traps_k = [[]]
        self.theta_traps = []
        self._weigth_traps = []

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
        self.setWindowTitle(WINDOW_TITLE)
        self._create_top_menu()
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

    TOP_MENU_HEIGHT: int = 32

    def _create_top_menu(self) -> None:
        """Fixed horizontal button bar at the top of the window (placeholder actions)."""
        bar = QtWidgets.QToolBar("Top menu", self)
        bar.setObjectName("topMenuBar")
        bar.setMovable(False)
        bar.setFloatable(False)
        bar.setFixedHeight(self.TOP_MENU_HEIGHT)
        bar.toggleViewAction().setVisible(False)  # no hide option in the context menu
        bar.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)

        self.top_menu_buttons = {}
        for label in ("Device", "Profiles", "Settings"):
            btn = QtWidgets.QPushButton(label)
            btn.setFixedSize(90, self.TOP_MENU_HEIGHT - 6)
            bar.addWidget(btn)
            self.top_menu_buttons[label] = btn

        self.top_menu_buttons["Device"].setMenu(self._create_devices_menu())
        self.top_menu_buttons["Profiles"].setMenu(self._create_profiles_menu())
        self.top_menu_buttons["Settings"].setMenu(self._create_settings_menu())

        self.addToolBar(QtCore.Qt.TopToolBarArea, bar)
        self.top_menu_bar = bar

    def _create_devices_menu(self) -> QtWidgets.QMenu:
        """Device drop-down: the SLM list stored in devices/devices.json."""
        menu = QtWidgets.QMenu(self)
        menu.addAction("View devices").triggered.connect(lambda _checked=False: self._showDevices())
        menu.addAction("Add device").triggered.connect(lambda _checked=False: self._addDeviceDialog())
        self._set_factory_lut_action = menu.addAction("Set factory LUT…")
        self._set_factory_lut_action.triggered.connect(lambda _checked=False: self._setFactoryLutDialog())
        menu.addSeparator()
        self._connect_action = menu.addAction("Connect…")
        self._connect_action.triggered.connect(lambda _checked=False: self._connectDialog())
        self._disconnect_action = menu.addAction("Disconnect")
        self._disconnect_action.triggered.connect(lambda _checked=False: self._disconnectClicked())
        menu.aboutToShow.connect(self._updateDeviceActions)
        return menu

    def _updateDeviceActions(self) -> None:
        self._connect_action.setEnabled(self.SLM is None)
        self._disconnect_action.setEnabled(self.SLM is not None)
        self._set_factory_lut_action.setEnabled(self.board is not None and self.board != SIMULATED_BOARD)

    def _setFactoryLutDialog(self) -> None:
        """Device ▸ Set factory LUT…: pick a .bmp for the selected (real) SLM, copy it to devices/ and store its name."""
        board = self.board
        if board is None or board == SIMULATED_BOARD:
            return
        label = self._boardLabel(board)
        if findDevice(board) is None:
            QtWidgets.QMessageBox.warning(self, "Set factory LUT", f"{label} is not in the device list.\nAdd it first with Device ▸ Add device.")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, f"Factory LUT for {label}", str(DEVICES_DIR), "Bitmap (*.bmp)")
        if not path:
            return
        source = Path(path)
        try:
            pattern = loadCorrectionPattern(source)
        except (OSError, ValueError) as e:
            QtWidgets.QMessageBox.warning(self, "Set factory LUT", f"{source.name} cannot be used:\n{e}")
            return
        name = source.stem
        target = factoryLutPath(name)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not (target.exists() and target.resolve() == source.resolve()):
                if target.exists() and not filecmp.cmp(source, target, shallow=False):
                    text = f"A different file named {target.name} is already in\n{target.parent}\n(it may be used by another SLM).\n\nOverwrite it?"
                    if QtWidgets.QMessageBox.question(self, "Set factory LUT", text) != QtWidgets.QMessageBox.Yes:
                        return
                shutil.copyfile(source, target)
            setDeviceFactoryLut(board, name)
        except OSError as e:
            print("Could not set the factory LUT:", e)
            QtWidgets.QMessageBox.warning(self, "Set factory LUT", f"Could not set the factory LUT:\n{e}")
            return
        print(f"{label}: factory LUT set to {name!r} ({target})")
        if self.isSLMConnected() and self.board == board:
            self.factory_lut = pattern
            self.updatePhase()
            print(f"{label}: new factory LUT applied")

    def _disconnectClicked(self) -> None:
        """Device ▸ Disconnect: close the SLM session after confirmation; the GUI keeps running without an SLM."""
        if self.SLM is None:
            return
        text = f"Disconnect {self._boardLabel(self.board)}?\n\nHolograms will no longer be sent until you connect again (Device ▸ Connect…)."
        if QtWidgets.QMessageBox.question(self, "Disconnect", text) != QtWidgets.QMessageBox.Yes:
            return
        self.disconnectSLM()

    def disconnectSLM(self) -> None:
        """Close the SLM session. ``self.board`` is kept, so the profile and Device ▸ Connect… still refer to it."""
        slm, self.SLM = self.SLM, None
        self.factory_lut = None
        if slm is not None:
            try:
                slm.disconnect()
            except Exception as e:
                print(f"Error while disconnecting {self._boardLabel(self.board)}: {e}")
            print(f"Disconnected {self._boardLabel(self.board)}")
        self._updateWindowTitle()

    @staticmethod
    def _boardLabel(board: int) -> str:
        return deviceLabel(Device(board))

    @staticmethod
    def _deviceLabel(device: Device) -> str:
        if device.simulated:
            return deviceLabel(device)
        return f"{deviceLabel(device)}  (SN {'not known yet' if device.serial is None else device.serial}, factory LUT: {device.factory_lut or 'none'})"

    def _showDevices(self) -> None:
        devices = loadDevices()
        boards = [d.board for d in devices]
        lines = [self._deviceLabel(d) + ("   ← connected" if self.SLM is not None and d.board == self.board else "") for d in devices]
        text = "\n".join(lines)
        if not any(not d.simulated for d in devices):
            text += "\n\nNo real SLM in the device list yet: use Device ▸ Add device."
        if self.SLM is not None and self.board not in boards:
            text += f"\n\nConnected: {self._boardLabel(self.board)} (not in the list)"
        QtWidgets.QMessageBox.information(self, "Devices", f"{text}\n\nThe serial number is recorded the first time a device is connected.\nList file: {DEVICES_FILE}")

    def _addDeviceDialog(self) -> Optional[int]:
        """Ask for a board number and add it to devices.json; return the board, or None if cancelled."""
        boards = [d.board for d in loadDevices()]
        suggestion = next(b for b in range(1, len(boards) + 2) if b not in boards)
        board, ok = QtWidgets.QInputDialog.getInt(self, "Add device", "Board number of the SLM:", suggestion, 1, 99)
        if not ok:
            return None
        try:
            added = addDevice(board)
        except OSError as e:
            print("Could not save the device list:", e)
            QtWidgets.QMessageBox.warning(self, "Add device", f"Could not save the device list:\n{e}")
            return None
        print(f"Added {self._boardLabel(board)} to the device list" if added else f"{self._boardLabel(board)} is already in the device list")
        return board

    def _connectDialog(self) -> None:
        """Device ▸ Connect…: choose an SLM from the device list and connect to it."""
        if self.SLM is not None:
            return
        devices = loadDevices()
        boards = [d.board for d in devices]
        labels = [self._deviceLabel(d) for d in devices]
        current = boards.index(self.board) if self.board in boards else 0
        label, ok = QtWidgets.QInputDialog.getItem(self, "Connect", "SLM to connect to (Simulated SLM = no hardware):", labels, current, False)
        board = boards[labels.index(label)] if ok else None
        if board is not None:
            self.connectSLM(board)

    def _create_settings_menu(self) -> QtWidgets.QMenu:
        """Settings drop-down: which profile is loaded when the app starts (stored in settings.json)."""
        menu = QtWidgets.QMenu(self)
        startup_menu = menu.addMenu("Startup profile")
        self._startup_mode_group = QtWidgets.QActionGroup(self)
        self._startup_mode_group.setExclusive(True)
        for mode, label in STARTUP_MODES.items():
            action = startup_menu.addAction(label)
            action.setCheckable(True)
            action.setData(mode)
            action.setChecked(mode == self.settings.startup_mode)
            self._startup_mode_group.addAction(action)
        self._startup_mode_group.triggered.connect(self._on_startup_mode_selected)
        return menu

    def _on_startup_mode_selected(self, action: QtWidgets.QAction) -> None:
        self.settings.startup_mode = str(action.data())
        try:
            self.settings.save()
        except OSError as e:
            print("Could not save settings:", e)
            return
        print(f"Startup profile: {STARTUP_MODES[self.settings.startup_mode]}")

    @staticmethod
    def _formatNumber(value: float) -> str:
        """Profile number as line-edit text: whole numbers without decimals (40, not 40.0)."""
        value = float(value)
        return str(int(value)) if value.is_integer() else repr(value)

    def profileParams(self) -> dict[str, Any]:
        """Current values of everything stored in a profile (same layout as profile_templates/default.json)."""
        params: dict[str, Any] = {} if self.board is None else {"board": self.board}
        params["grating"] = {"x_um": float(self.dx), "y_um": float(self.dy), "phase_levels": self.param_P, "wavelength_nm": round(self.wavelength_m * 1e9, 6)}
        params["corrections"] = {"manufacturer": self._ManufacturerCorrectionChkBx.isChecked(), "aberration": self._zernikeCorrectionChkBx.isChecked()}
        pupil: dict[str, Any] = {"enabled": self._pupilChkBx.isChecked(), "radius_mm": self.pupilEdt.text().strip()}
        for key, edit in (("x0_mm", self.pupilCenterEdt_x0), ("y0_mm", self.pupilCenterEdt_y0), ("move_out_um", self.pupilOutLineEdit)):
            try:
                pupil[key] = float(edit.text())
            except ValueError:
                print(f"Profile: pupil {key} = {edit.text()!r} is not a number and is not saved")
        params["pupil"] = pupil
        params["aberration"] = {label: float(v) for label, v in zip(self.aberration_label_list, self.aberration_values)}
        return params

    def applyProfile(self, params: dict[str, Any]) -> None:
        """Set the GUI from full profile parameters (template merged with a profile file); the hologram is updated once at the end."""
        board = parseBoard(params.get("board"))
        if "board" in params and board is None:
            print(f"Ignoring invalid board {params['board']!r} in profile {self.current_profile!r}")
        self.board = board
        grating, corrections, pupil, aberration = params["grating"], params["corrections"], params["pupil"], params["aberration"]
        missing = [label for label in self.aberration_label_list if label not in aberration]
        if missing:
            raise ValueError(f"Profile has no value for aberration(s) {missing}: check {PROFILE_TEMPLATE_FILE}")
        self._suspend_phase_updates = True
        try:
            self.param_P = grating["phase_levels"]
            self.elementary.paramPEdt.setText(self._formatNumber(grating["phase_levels"]))
            self.wavelength_m = grating["wavelength_nm"] / 1e9
            self.elementary.paramWavelengthEdit.setText(self._formatNumber(grating["wavelength_nm"]))
            for box, checked in ((self._ManufacturerCorrectionChkBx, corrections["manufacturer"]), (self._zernikeCorrectionChkBx, corrections["aberration"]), (self._pupilChkBx, pupil["enabled"])):
                box.blockSignals(True)
                box.setChecked(checked)
                box.blockSignals(False)
            self.pupilEdt.setText(pupil["radius_mm"])
            self.pupilCenterEdt_x0.setText(self._formatNumber(pupil["x0_mm"]))
            self.pupilCenterEdt_y0.setText(self._formatNumber(pupil["y0_mm"]))
            self.pupilOutLineEdit.setText(self._formatNumber(pupil["move_out_um"]))
            self._advanced.update_k_dump()
            self.aberration_values = np.zeros(len(self.aberration_label_list))
            self.aberration.updateAberration(aber_list=[aberration[label] for label in self.aberration_label_list])
            for edit, label in zip(self.aber_lineEdit_list, self.aberration_label_list):
                edit.setText(self._formatNumber(aberration[label]))
            self.dx, self.dy = grating["x_um"], grating["y_um"]
            self.elementary.createGratingPattern(self.dx, self.dy)
            if pupil["enabled"]:
                self._advanced.calculatePupil()
        finally:
            self._suspend_phase_updates = False
        self.updatePhase()

    def setWavelength(self, wavelength_m: float) -> None:
        """Change the laser wavelength: recompute every λ-dependent term, then send one consistent frame.

        λ converts µm in the focal plane into phase on the SLM (∝ 1/λ): grating shift, pupil "move out" grating and
        k_dump follow at once. The trap hologram (phasePattern) is only rebuilt by Run Algorithm / Load Hologram,
        so a warning is shown when one is in use.
        """
        old_m = self.wavelength_m
        if wavelength_m == old_m:
            return
        self.wavelength_m = wavelength_m
        self.elementary.paramWavelengthEdit.setText(self._formatNumber(wavelength_m * 1e9))
        self._suspend_phase_updates = True
        try:
            self._advanced.update_k_dump()
            self.elementary.createGratingPattern(self.dx, self.dy)
            if self._pupilChkBx.isChecked():
                self._advanced.calculatePupil()
        finally:
            self._suspend_phase_updates = False
        self.updatePhase()
        print(f"Wavelength changed: {old_m * 1e9:g} nm -> {wavelength_m * 1e9:g} nm (grating, pupil and k_dump recomputed)")
        if np.any(self.phasePattern):
            text = f"The wavelength changed from {old_m * 1e9:g} nm to {wavelength_m * 1e9:g} nm.\n\nThe grating and pupil now use the new wavelength, but the trap hologram on the SLM was computed for {old_m * 1e9:g} nm: with a {wavelength_m * 1e9:g} nm laser its traps land at {wavelength_m / old_m:.3f}× the intended distances.\n\nPress Run Algorithm (or reload the hologram) to recompute it."
            QtWidgets.QMessageBox.warning(self, "Wavelength changed", text)

    def loadProfile(self, name: str) -> None:
        """Make ``name`` the active profile: repository defaults overridden by ``profiles/<name>.json`` (if it exists)."""
        self.current_profile = name
        try:
            params, state = loadProfileParams(name)
        except ProfileTemplateError as e:
            QtWidgets.QMessageBox.critical(self, "Default profile missing", f"{e}\n\nThe SLM Controller needs this file from the repository to start.")
            raise
        if state == PROFILE_LOADED:
            print(f"Loaded profile {name!r}")
        elif state == PROFILE_MISSING:
            print(f"Profile {name!r}: no saved file yet, using the default values")
        else:
            print(f"Profile {name!r} could not be read: using the default values (Profiles ▸ Restore default profile repairs 'default')")
        self.applyProfile(params)

    def saveProfile(self, name: Optional[str] = None) -> Path:
        """Write ``profileParams()`` to ``profiles/<name>.json`` (default: active profile) and return the path."""
        return writeProfile(name or self.current_profile, self.profileParams())

    def _create_profiles_menu(self) -> QtWidgets.QMenu:
        """Profiles drop-down: load, save and duplicate the profiles in profiles/."""
        menu = QtWidgets.QMenu(self)
        menu.addAction("Load…").triggered.connect(lambda _checked=False: self._loadProfileDialog())
        menu.addAction("Save").triggered.connect(lambda _checked=False: self._saveProfileClicked())
        menu.addAction("Duplicate…").triggered.connect(lambda _checked=False: self._duplicateProfileDialog())
        menu.addSeparator()
        menu.addAction("Restore default profile").triggered.connect(lambda _checked=False: self._restoreDefaultProfileClicked())
        return menu

    def _restoreDefaultProfileClicked(self) -> None:
        """Profiles ▸ Restore default profile: overwrite profiles/default.json with the repository template."""
        target = profilePath(DEFAULT_PROFILE)
        active = self.current_profile == DEFAULT_PROFILE
        text = f"Overwrite\n{target}\nwith the default values from the repository\n{PROFILE_TEMPLATE_FILE}?"
        if active:
            text += f"\n\n'{DEFAULT_PROFILE}' is the active profile: the default values are applied now."
        if QtWidgets.QMessageBox.question(self, "Restore default profile", text) != QtWidgets.QMessageBox.Yes:
            return
        try:
            path = restoreDefaultProfile()
        except (OSError, ProfileTemplateError) as e:
            print("Could not restore the default profile:", e)
            QtWidgets.QMessageBox.warning(self, "Restore default profile", f"Could not restore the default profile:\n{e}")
            return
        print(f"Restored the default profile: {path}")
        if active:
            connected_board = self.board if self.isSLMConnected() else None
            self.applyProfile(loadProfileParams(DEFAULT_PROFILE)[0])
            if connected_board is not None:
                self.board = connected_board
            self._updateWindowTitle()

    def _chooseProfile(self, title: str, label: str) -> Optional[str]:
        """Pick a profile name from profiles/ (the active one is always listed and preselected)."""
        names = sorted(set(listProfiles()) | {self.current_profile})
        name, ok = QtWidgets.QInputDialog.getItem(self, title, label, names, names.index(self.current_profile), False)
        return name if ok else None

    def _saveProfileClicked(self) -> None:
        try:
            path = self.saveProfile()
        except OSError as e:
            print("Could not save the profile:", e)
            QtWidgets.QMessageBox.warning(self, "Save profile", f"Could not save profile {self.current_profile!r}:\n{e}")
            return
        print(f"Saved profile {self.current_profile!r}: {path}")

    def _loadProfileDialog(self) -> None:
        """Profiles ▸ Load…: make another profile active. Never changes the SLM connection: a profile for another SLM is refused."""
        name = self._chooseProfile("Load profile", "Profile to load:")
        if name is None:
            return
        params, state = loadProfileParams(name)
        if state != PROFILE_LOADED:
            reason = "has no saved file yet" if state == PROFILE_MISSING else "could not be read (damaged file)"
            QtWidgets.QMessageBox.warning(self, "Load profile", f"Profile {name!r} {reason}.")
            return
        board = parseBoard(params.get("board"))
        connected = self.isSLMConnected()
        if connected and board is not None and board != self.board:
            print(f"Profile {name!r} not loaded: it is for {self._boardLabel(board)}, the connected SLM is {self._boardLabel(self.board)}")
            QtWidgets.QMessageBox.warning(self, "Load profile", f"Profile {name!r} is for {self._boardLabel(board)}, but the connected SLM is {self._boardLabel(self.board)}.\n\nOnly profiles for the connected SLM can be loaded.")
            return
        try:
            self.saveProfile()
        except OSError as e:
            print(f"Could not save profile {self.current_profile!r} before loading {name!r}:", e)
        connected_board = self.board
        self.current_profile = name
        self.applyProfile(params)
        if connected:
            self.board = connected_board
        self.settings.last_profile = name
        try:
            self.settings.save()
        except OSError as e:
            print("Could not save settings:", e)
        self._updateWindowTitle()
        hint = "" if connected or self.board is None else f" ({self._boardLabel(self.board)}: Device ▸ Connect… to connect)"
        print(f"Loaded profile {name!r}{hint}")

    def _duplicateProfileDialog(self) -> None:
        """Profiles ▸ Duplicate…: save the active profile, with its current values, under a new name in profiles/."""
        source = self.current_profile
        target, ok = QtWidgets.QInputDialog.getText(self, "Duplicate profile", f"Name of the new profile (copy of the active profile {source!r}):", QtWidgets.QLineEdit.Normal, f"{source}_copy")
        if not ok:
            return
        target = target.strip()
        if not isValidProfileName(target):
            QtWidgets.QMessageBox.warning(self, "Duplicate profile", f"{target!r} is not a valid name.\nUse letters, digits, spaces, '-' or '_'.")
            return
        if target in listProfiles():
            QtWidgets.QMessageBox.warning(self, "Duplicate profile", f"Profile {target!r} already exists.")
            return
        params = self.profileParams()
        try:
            path = writeProfile(target, params)
        except OSError as e:
            print("Could not write the profile copy:", e)
            QtWidgets.QMessageBox.warning(self, "Duplicate profile", f"Could not write {target!r}:\n{e}")
            return
        print(f"Copied profile {source!r} to {target!r}: {path}")

    def createDocks(self) -> None:
        self._d1 = dockarea.Dock("All", size=(700, 700))
        self.area.addDock(self._d1, "left")
        self._d1.hideTitleBar()

        self._global_layout = pg.LayoutWidget()

        # Left column in its own VBox so Elementary is not stretched to the
        # Arbitrary Arrays row height in the outer grid.
        left_col = QtWidgets.QWidget()
        left_lay = QtWidgets.QVBoxLayout(left_col)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(PANEL_GAP_PX)

        self.elementary = ElementaryControlPanel(self)
        elem_frame = framed_panel(self.elementary)
        elem_frame.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        left_lay.addWidget(elem_frame)

        self._advanced = AdvancedControlPanel(self)
        adv_frame = framed_panel(self._advanced)
        adv_frame.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
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
        aber_frame.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
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
        self._slm_info = SlmInfoDock(self, PANEL_GAP_PX)
        dock_splitter = self.d_output.container()
        if isinstance(dock_splitter, QtWidgets.QSplitter):
            dock_splitter.setHandleWidth(PANEL_GAP_PX)
        self.area.layout.setContentsMargins(PANEL_GAP_PX, PANEL_GAP_PX, PANEL_GAP_PX, PANEL_GAP_PX)  # gaps to the window edges and below the top menu

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        # Re-apply the SLM info / main / Output height split on every window resize until
        # the user drags the splitter themselves. Deferred so the dock
        # splitter has already been resized to the new window height.
        if hasattr(self, "d_output") and not self._dock_ratio_user_set:
            QtCore.QTimer.singleShot(0, self._apply_dock_ratio)

    def _on_dock_splitter_moved(self, *_: Any) -> None:
        self._dock_ratio_user_set = True

    def _apply_dock_ratio(self) -> None:
        """Give the SLM info dock SLM_INFO_HEIGHT_FRACTION and the Output dock OUTPUT_HEIGHT_FRACTION of the
        dock area height, and the rest to the docks between them."""
        splitter = self.d_output.container()
        if not isinstance(splitter, QtWidgets.QSplitter):
            return
        if not self._dock_splitter_connected:
            splitter.splitterMoved.connect(self._on_dock_splitter_moved)
            self._dock_splitter_connected = True
        sizes = splitter.sizes()
        total = sum(sizes)
        fixed = {splitter.indexOf(dock): fraction for dock, fraction in ((self.d_output, OUTPUT_HEIGHT_FRACTION), (getattr(self, "d_slm_info", None), SLM_INFO_HEIGHT_FRACTION)) if dock is not None and splitter.indexOf(dock) >= 0}
        if splitter.indexOf(self.d_output) < 0 or len(sizes) < 2 or total <= 0:
            return
        new_sizes = list(sizes)
        for i, fraction in fixed.items():
            new_sizes[i] = int(round(total * fraction))
        # The main dock may be wrapped in its own container, so give the
        # remaining height to all other splitter entries proportionally.
        others = [i for i in range(len(sizes)) if i not in fixed]
        other_total = sum(sizes[i] for i in others) or len(others)
        rest = total - sum(new_sizes[i] for i in fixed)
        for i in others:
            new_sizes[i] = int(rest * (sizes[i] or 1) / other_total)
        splitter.setSizes(new_sizes)

    def _install_gui_stdio(self) -> None:
        self._stdout_prev = sys.stdout
        self._stderr_prev = sys.stderr
        self._stdout_tee = StreamToPlainText(self._stdout_prev, self.output_log, parent=self)
        self._stderr_tee = StreamToPlainText(self._stderr_prev, self.output_log, parent=self)
        sys.stdout = self._stdout_tee
        sys.stderr = self._stderr_tee

    def _restore_stdio(self) -> None:
        if getattr(self, "_stdout_prev", None) is not None:
            sys.stdout = self._stdout_prev
        if getattr(self, "_stderr_prev", None) is not None:
            sys.stderr = self._stderr_prev

    def adjustImageForS1028(self, array: np.ndarray) -> np.ndarray:
        v_s, h_s = 1024, 1272
        v, h = array.shape
        converted_array = np.zeros([v_s, h_s])
        converted_array[(v_s - v) // 2 : (v_s + v) // 2, (h_s - h) // 2 : (h_s + h) // 2] = np.fliplr(array)
        return converted_array

    def updatePhase(self) -> None:
        if self._suspend_phase_updates:
            return
        self.totalPhase = self.im_grating + self.fresnelPhase + self.phasePattern

        if self._ManufacturerCorrectionChkBx.isChecked() and self.factory_lut is not None:
            self.totalPhase = self.totalPhase + self.factory_lut
        if self._zernikeCorrectionChkBx.isChecked():
            self.totalPhase = self.totalPhase + self.im_zernike
        if self._pupilChkBx.isChecked():
            self._advanced.applyPupil()

        self.totalPhase = np.mod(self.totalPhase, 2 * np.pi)

        self.totalPhase = np.round(self.param_P * self.totalPhase / (2 * np.pi)).astype("uint8")
        if self.SLM is not None:
            self.SLM.uploadHologram(self.totalPhase)
        self._publish_static_changes()
        self._preview._update_last_sent_preview()

    def __del__(self) -> None:
        plt.close("all")
        print("closing all")
        slm, self.SLM = getattr(self, "SLM", None), None
        if slm is not None:
            slm.disconnect()

    def closeEvent(self, err: QtGui.QCloseEvent) -> None:
        try:
            print("Saved profile:", self.saveProfile())
            self.settings.last_profile = self.current_profile
            self.settings.save()
        except OSError as e:
            print("Could not save profile / settings:", e)
        timer = getattr(self, "_slm_temp_timer", None)
        if timer is not None:
            timer.stop()
        log = getattr(self, "_temperature_log", None)
        if log is not None:
            log.close()
        self._stop_hero()
        self._restore_stdio()
        self.__del__()
