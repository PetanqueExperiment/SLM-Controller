"""Shared GUI helpers."""
import ast
import os
import re
import sys

import numpy as np
from pyqtgraph.Qt import QtGui, QtWidgets

from gui.constants import LOCAL_DIR, SH_CORRECTION_NPY


def load_sh_correction_phase_file(ny, nx):
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
        "does not match SLM (%d, %d); Shack–Hartmann correction disabled until fixed."
        % (ny, nx),
    )
    return out


def parse_pasted_coordinates(text):
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


def register_windows_taskbar_app_id():
    """Set AppUserModelID so the Windows taskbar uses our icon. Call before QApplication()."""
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "Petanque.SLMController.GUI.1"
        )


def application_icon(base_dir=None):
    """Load the GUI icon. Requires QApplication (QIcon may use QPixmap internally)."""
    if base_dir is None:
        base_dir = LOCAL_DIR
    return QtGui.QIcon(os.path.join(base_dir, "Icon", "Petanque array.ico"))


def parse_launch_args(argv=None):
    """CLI flags for offline GUI testing without Meadowlark hardware."""
    import argparse

    parser = argparse.ArgumentParser(description="SLM Controller Petanque")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run without Blink SDK / SLM hardware (patterns stay in memory).",
    )
    args, _unknown = parser.parse_known_args(argv)
    return bool(args.simulate)
