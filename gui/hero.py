"""
HERO wrapper exposing the SLM Controller GUI remotely.

Same pattern as the Elliptec ELL14 HERO: one ``LocalHERO`` per device. Only the
grating shift and the Zernike coefficients are exposed for now.

HERO calls arrive on zenoh worker threads, while all phase updates touch Qt
widgets and the SLM. Every exposed method is therefore run on the GUI thread
via ``_GuiInvoker`` and blocks until it is done, so remote callers get the
return value (or the exception) back.
"""

from __future__ import annotations

import threading

from pyqtgraph.Qt import QtCore

from heros import LocalHERO

HERO_NAME = "SLM_Controller"

GRATING_X = "grating_x"
GRATING_Y = "grating_y"


class _GuiInvoker(QtCore.QObject):
    """Runs callables on the thread this object lives in (the GUI thread)."""

    _call = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._call.connect(self._run, QtCore.Qt.BlockingQueuedConnection)

    @QtCore.Slot(object)
    def _run(self, job):
        job()

    def invoke(self, fn, *args, **kwargs):
        if QtCore.QThread.currentThread() == self.thread():
            return fn(*args, **kwargs)

        result = {}

        def job():
            try:
                result["value"] = fn(*args, **kwargs)
            except Exception as exc:
                result["error"] = exc

        self._call.emit(job)
        if "error" in result:
            raise result["error"]
        return result.get("value")


class SLMHero(LocalHERO):
    """Remote access to the grating shift and Zernike coefficients of ``SLMGUI``."""

    def __init__(self, gui, name: str = HERO_NAME, *args, **kwargs):
        super().__init__(name, tags=["device:SLM"], *args, **kwargs)
        self._gui = gui
        self._invoker = _GuiInvoker()
        # Serialize remote callers: one phase update at a time.
        self._lock = threading.Lock()

    def _on_gui(self, fn, *args, **kwargs):
        with self._lock:
            return self._invoker.invoke(fn, *args, **kwargs)

    # ---- grating shift (um) ------------------------------------------------

    def set_grating_shift(self, dx: float, dy: float) -> list[float]:
        return self._on_gui(self._set_grating, float(dx), float(dy))

    def set_grating_x(self, dx: float) -> list[float]:
        return self._on_gui(self._set_grating, float(dx), None)

    def set_grating_y(self, dy: float) -> list[float]:
        return self._on_gui(self._set_grating, None, float(dy))

    def get_grating_shift(self) -> list[float]:
        return self._on_gui(lambda: [float(self._gui.dx), float(self._gui.dy)])

    # ---- Zernike coefficients ----------------------------------------------
    # Applied to the SLM only while "Zernike correction" is checked in the GUI.

    def get_zernike_names(self) -> list[str]:
        return list(self._gui.aberration_label_list)

    def get_zernike_coefficients(self) -> dict:
        return self._on_gui(self._zernike_dict)

    def set_zernike(self, name: str, value: float) -> dict:
        return self._on_gui(self._set_zernike, {name: value})

    def set_zernike_coefficients(self, coefficients: dict) -> dict:
        """Set several coefficients at once (``{name: value}``); others are kept."""
        return self._on_gui(self._set_zernike, dict(coefficients))

    # ---- sequencer ---------------------------------------------------------

    def sequencer_apply_static(self, param_id: str, value: float) -> float:
        """``param_id`` is ``grating_x``, ``grating_y`` or a Zernike name."""
        if param_id == GRATING_X:
            return self.set_grating_x(value)[0]
        if param_id == GRATING_Y:
            return self.set_grating_y(value)[1]
        return self.set_zernike(param_id, value)[param_id]

    # ---- GUI-thread internals ----------------------------------------------

    def _set_grating(self, dx, dy):
        if dx is not None:
            self._gui.dx = dx
        if dy is not None:
            self._gui.dy = dy
        self._gui.elementary.createGratingPattern(self._gui.dx, self._gui.dy)
        return [float(self._gui.dx), float(self._gui.dy)]

    def _zernike_dict(self):
        return {
            label: float(v)
            for label, v in zip(self._gui.aberration_label_list, self._gui.aberration_values)
        }

    def _set_zernike(self, coefficients):
        labels = list(self._gui.aberration_label_list)
        unknown = [k for k in coefficients if k not in labels]
        if unknown:
            raise KeyError(f"unknown Zernike name(s) {unknown}; valid names: {labels}")
        values = self._zernike_dict()
        values.update({k: float(v) for k, v in coefficients.items()})
        self._gui.aberration.updateAberration(aber_list=[values[k] for k in labels])
        return self._zernike_dict()
