"""Shared GUI helpers."""

import argparse
import ast
import os
import re
import subprocess
import sys

import numpy as np
from pyqtgraph.Qt import QtGui, QtWidgets

from gui.constants import LOCAL_DIR, SH_CORRECTION_NPY


def load_sh_correction_phase_file(ny: int, nx: int) -> np.ndarray:
    """Load optional SH phase map in radians; shape must match SLM. Otherwise return zeros."""
    out = np.zeros((ny, nx), dtype=np.float64)
    if not os.path.isfile(SH_CORRECTION_NPY):
        return out
    arr = np.load(SH_CORRECTION_NPY)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.shape == (ny, nx):
        return arr
    print(
        "SH_correction.npy shape",
        arr.shape,
        "does not match SLM (%d, %d); Shack–Hartmann correction disabled until fixed." % (ny, nx),
    )
    return out


def parse_pasted_coordinates(text: str | None) -> np.ndarray:
    """
    Parse pasted trap coordinates into an (N, 2) or (N, 3) float array.

    Accepts Python lists/tuples (``[[x, y], ...]``), ``np.array([...])`` / ``array([...], dtype=...)``
    reprs, and numpy ``print`` output without commas (``[[1. 2.]\\n [3. 4.]]``).
    """
    s = (text or "").strip()
    if not s:
        raise ValueError("empty input")
    s = re.sub(r"^\s*(?:np\.|numpy\.)?(?:array|asarray)\s*\(", "", s)
    s = re.sub(r",\s*dtype\s*=\s*[\w.\'\"]+\s*", "", s)
    s = s.strip().rstrip(";").strip()
    while s.endswith(")") and s.count(")") > s.count("("):
        s = s[:-1].rstrip()

    try:
        arr = np.asarray(ast.literal_eval(s), dtype=np.float64)
    except (ValueError, SyntaxError, TypeError):
        num = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
        rows = [re.findall(num, grp) for grp in re.findall(r"\[([^\[\]]*)\]", s)]
        rows = [r for r in rows if r]
        if not rows:
            raise ValueError("could not find any numbers")
        if len({len(r) for r in rows}) != 1:
            raise ValueError("rows have different lengths")
        arr = np.asarray(rows, dtype=np.float64)

    if arr.ndim == 1 and arr.size in (2, 3):
        arr = arr[None, :]
    if arr.ndim == 2 and arr.shape[0] in (2, 3) and arr.shape[1] not in (2, 3):
        arr = arr.T
    if arr.ndim != 2 or arr.shape[1] not in (2, 3) or arr.shape[0] < 1:
        raise ValueError("expected shape (N, 2) or (N, 3), got %s" % (arr.shape,))
    if not np.all(np.isfinite(arr)):
        raise ValueError("coordinates contain NaN or inf")
    return arr


def framed_panel(inner_widget: QtWidgets.QWidget) -> QtWidgets.QFrame:
    """Wrap a `pg.LayoutWidget` (or any QWidget) in a bordered frame so regions are visually distinct."""
    frame = QtWidgets.QFrame()
    frame.setObjectName("slm_panel_frame")
    frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
    frame.setFrameShadow(QtWidgets.QFrame.Plain)
    frame.setLineWidth(1)
    frame.setStyleSheet("#slm_panel_frame { border: 1px solid #8a8a8a; border-radius: 4px; background: transparent; }")
    lay = QtWidgets.QVBoxLayout(frame)
    lay.setContentsMargins(6, 6, 6, 6)
    lay.setSpacing(0)
    lay.addWidget(inner_widget)
    return frame


def register_windows_taskbar_app_id() -> None:
    """Set AppUserModelID so the Windows taskbar uses our icon. Call before QApplication()."""
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Petanque.SLMController.GUI.1")


# Local pipe the running GUI listens on, so the taskbar's Restart (a separate process) can reach it.
CONTROL_SERVER_NAME = "Petanque.SLMController.GUI.1.control"
RESTART_REQUEST = b"restart\n"
RESTART_RUNNING_FLAG = "--restart-running"


def restart_command(argv: list[str] | None = None) -> list[str]:
    """Command line that starts the SLM Controller again: same Python, same launch flags."""
    if argv is None:
        argv = sys.argv
    flags = [a for a in argv[1:] if a != RESTART_RUNNING_FLAG]
    return [sys.executable, os.path.join(LOCAL_DIR, "GUI_SLM.py"), *flags]


def jump_list_restart_command(argv: list[str] | None = None) -> tuple[str, list[str]]:
    """(program, arguments) for the taskbar's Restart entry: asks the running GUI to restart.

    Uses pythonw.exe when available so no console window flashes up.
    """
    command = restart_command(argv)
    program = command[0]
    pythonw = os.path.join(os.path.dirname(program), "pythonw.exe")
    if sys.platform == "win32" and os.path.isfile(pythonw):
        program = pythonw
    return program, [*command[1:], RESTART_RUNNING_FLAG]


def request_running_instance_restart(timeout_ms: int = 2000) -> bool:
    """Ask an already running SLM Controller to restart. False if none answers.

    Needs a QCoreApplication / QApplication.
    """
    from PyQt5 import QtNetwork

    socket = QtNetwork.QLocalSocket()
    socket.connectToServer(CONTROL_SERVER_NAME)
    if not socket.waitForConnected(timeout_ms):
        return False
    socket.write(RESTART_REQUEST)
    sent = socket.waitForBytesWritten(timeout_ms)
    socket.disconnectFromServer()
    return sent


def launch_detached(command: list[str]) -> None:
    """Start ``command`` independent of this process (no console window on Windows).

    Raises OSError if the process cannot be started.
    """
    if sys.platform == "win32":
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        subprocess.Popen(command, cwd=LOCAL_DIR, creationflags=flags, close_fds=True,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    else:
        subprocess.Popen(command, cwd=LOCAL_DIR, start_new_session=True,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)


def application_icon(base_dir: str | None = None) -> QtGui.QIcon:
    """Load the GUI icon. Requires QApplication (QIcon may use QPixmap internally)."""
    if base_dir is None:
        base_dir = LOCAL_DIR
    return QtGui.QIcon(os.path.join(base_dir, "Icon", "Petanque array.ico"))


def parse_launch_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI flags: HERO options. (Simulate mode is chosen by selecting the Simulated SLM in the device list.)"""
    parser = argparse.ArgumentParser(description="SLM Controller Petanque")
    parser.add_argument(
        "--no-hero", action="store_true", help="Do not expose the GUI as a HERO (remote grating / Zernike control)."
    )
    parser.add_argument(
        "--hero-name", default=None, help="HERO name (default: SLM_Controller, SLM_Controller_sim when the profile's SLM is the Simulated SLM)."
    )
    parser.add_argument(
        RESTART_RUNNING_FLAG, action="store_true",
        help="Ask the running SLM Controller to restart (used by the taskbar); start normally if none is running.",
    )
    args, _unknown = parser.parse_known_args(argv)
    return args
