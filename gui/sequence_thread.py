"""Worker thread of the dynamic-mode sequence: composes the frames and runs gui/sequence_run.py off the GUI thread."""

from __future__ import annotations

import threading
from typing import Sequence

import numpy as np
from pyqtgraph.Qt import QtCore

from gui.sequence_run import RUN_ERROR, RUN_STOPPED, SequenceRunResult, SequenceSLM, playSequence
from gui.slm_frame import FrameTerms, composeSlmFrame


class SequenceRunThread(QtCore.QThread):
    """One sequence run. ``armed`` is emitted once frame 0 is shown; ``result`` is set when the thread finishes."""

    armed = QtCore.Signal()

    def __init__(self, slm: SequenceSLM, patterns_rad: Sequence[np.ndarray], paths: Sequence[str], terms: FrameTerms, timeout_ms: int, flip_immediate: bool, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.slm = slm
        self.patterns_rad = list(patterns_rad)
        self.paths = list(paths)
        self.terms = terms
        self.timeout_ms = int(timeout_ms)
        self.flip_immediate = bool(flip_immediate)
        self.frames: list[np.ndarray] = []
        self.result: SequenceRunResult | None = None
        self._stop = threading.Event()

    def requestStop(self) -> None:
        self._stop.set()

    def stopRequested(self) -> bool:
        return self._stop.is_set()

    def run(self) -> None:
        try:
            frames = []
            for pattern_rad in self.patterns_rad:
                if self._stop.is_set():
                    self.result = SequenceRunResult(RUN_STOPPED, 0, None, 'stopped before the frames were loaded')
                    return
                frames.append(composeSlmFrame(pattern_rad, self.terms))
            self.frames = frames
            self.result = playSequence(self.slm, frames, timeout_ms=self.timeout_ms, flip_immediate=self.flip_immediate, stop_requested=self._stop.is_set, on_armed=self.armed.emit)
        except Exception as e:
            self.result = SequenceRunResult(RUN_ERROR, 0, None, f'{type(e).__name__}: {e}')
