"""SLM Controller GUI entry point (kept for Start_SLM_GUI*.bat)."""
import os
import sys

_SLM_SOFTWARE_DIR = os.path.dirname(os.path.abspath(__file__))
if _SLM_SOFTWARE_DIR not in sys.path:
    sys.path.insert(0, _SLM_SOFTWARE_DIR)

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")

from pyqtgraph.Qt import QtWidgets

from gui.helpers import (
    RESTART_RUNNING_FLAG,
    application_icon,
    parse_launch_args,
    register_windows_taskbar_app_id,
    request_running_instance_restart,
)


if __name__ == "__main__":
    register_windows_taskbar_app_id()

    _args = parse_launch_args()

    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance()

    if _args.restart_running:
        # Started from the taskbar's Restart: hand the request to the running GUI and exit.
        if request_running_instance_restart():
            sys.exit(0)
        # Nothing running: start normally.
        sys.argv = [a for a in sys.argv if a != RESTART_RUNNING_FLAG]

    # Imported here so a taskbar Restart request exits quickly, without loading the GUI.
    from gui.main_window import SLMGUI

    _app_icon = application_icon()
    app.setWindowIcon(_app_icon)

    slmGui = SLMGUI(start_hero=not _args.no_hero, hero_name=_args.hero_name)
    slmGui.setWindowIcon(_app_icon)
    # Tray app: closing the window (or a plot / dialog while it is hidden) must not end the app.
    app.setQuitOnLastWindowClosed(not slmGui.hasTray())
    # Also release the SLM when Windows ends the session (logoff / shutdown) without a close event.
    app.aboutToQuit.connect(slmGui.shutdown)
    slmGui.showMaximized()
    try:
        ret = app.exec_()
    except Exception as err:
        print("Close event has been raised ", type(err))
    finally:
        del slmGui
