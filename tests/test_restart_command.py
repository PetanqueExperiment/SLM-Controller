"""Command line used by the tray's Restart SLM Controller (pure function, no process started)."""
import os
import sys

from gui.constants import LOCAL_DIR
from gui.helpers import CONTROL_SERVER_NAME, RESTART_REQUEST, jump_list_restart_command, restart_command


def test_restart_uses_this_python_and_the_entry_point():
    command = restart_command(["GUI_SLM.py"])
    assert command == [sys.executable, os.path.join(LOCAL_DIR, "GUI_SLM.py")]


def test_restart_keeps_launch_flags():
    command = restart_command(["GUI_SLM.py", "--no-hero", "--hero-name", "SLM_test"])
    assert command[2:] == ["--no-hero", "--hero-name", "SLM_test"]


def test_restart_drops_the_taskbar_request_flag():
    command = restart_command(["GUI_SLM.py", "--no-hero", "--restart-running"])
    assert command[2:] == ["--no-hero"]


def test_jump_list_entry_asks_the_running_gui_to_restart():
    program, arguments = jump_list_restart_command(["GUI_SLM.py", "--no-hero"])
    assert arguments == [os.path.join(LOCAL_DIR, "GUI_SLM.py"), "--no-hero", "--restart-running"]
    if sys.platform == "win32":
        assert os.path.basename(program).lower() in ("pythonw.exe", os.path.basename(sys.executable).lower())


def test_restart_request_reaches_a_listening_gui():
    """Like the taskbar entry: a separate process sends the request to the GUI's pipe."""
    import subprocess

    from PyQt5 import QtCore, QtNetwork

    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    name = CONTROL_SERVER_NAME + ".test"
    server = QtNetwork.QLocalServer()
    assert server.listen(name)
    received = []

    def on_connection():
        socket = server.nextPendingConnection()
        socket.readyRead.connect(lambda: received.append(bytes(socket.readAll())))

    server.newConnection.connect(on_connection)
    client = (
        "import sys; from PyQt5 import QtCore; import gui.helpers as h; "
        "app = QtCore.QCoreApplication([]); h.CONTROL_SERVER_NAME = %r; "
        "sys.exit(0 if h.request_running_instance_restart(timeout_ms=3000) else 1)" % name
    )
    proc = subprocess.Popen([sys.executable, "-c", client], cwd=LOCAL_DIR)
    try:
        deadline = QtCore.QDeadlineTimer(10000)
        while (proc.poll() is None or not received) and not deadline.hasExpired():
            app.processEvents(QtCore.QEventLoop.AllEvents, 50)
    finally:
        server.close()
        if proc.poll() is None:
            proc.kill()
    assert proc.wait(timeout=5) == 0
    assert RESTART_REQUEST in b"".join(received)


def test_restart_request_without_a_running_gui_returns_false():
    from PyQt5 import QtCore
    import gui.helpers as helpers

    QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    original = helpers.CONTROL_SERVER_NAME
    helpers.CONTROL_SERVER_NAME = CONTROL_SERVER_NAME + ".nobody"
    try:
        assert helpers.request_running_instance_restart(timeout_ms=200) is False
    finally:
        helpers.CONTROL_SERVER_NAME = original
