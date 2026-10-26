"""SLM Controller GUI entry point (kept for Start_SLM_GUI*.bat)."""
import os
import sys

_SLM_SOFTWARE_DIR = os.path.dirname(os.path.abspath(__file__))
if _SLM_SOFTWARE_DIR not in sys.path:
    sys.path.insert(0, _SLM_SOFTWARE_DIR)

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")

from pyqtgraph.Qt import QtWidgets

from gui.helpers import (
    application_icon,
    parse_launch_args,
    register_windows_taskbar_app_id,
)
from gui.main_window import SLMGUI


if __name__ == "__main__":
    register_windows_taskbar_app_id()

    _simulate = parse_launch_args()

    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance()

    _app_icon = application_icon()
    app.setWindowIcon(_app_icon)

    slmGui = SLMGUI(simulate=_simulate)
    slmGui.setWindowIcon(_app_icon)
    slmGui.show()
    try:
        ret = app.exec_()
    except Exception as err:
        print("Close event has been raised ", type(err))
    finally:
        del slmGui
