"""Tests for gui/sequence_run.py (trigger loop of the dynamic mode) and gui/slm_frame.py (frame composition)."""

import time

import numpy as np
import pytest

from gui.sequence_run import RUN_ERROR, RUN_NO_TRIGGER, RUN_STOPPED, playSequence
from gui.slm_frame import FrameTerms, composeSlmFrame
from libslm.slm import MeadowlarkSLM

SHAPE = (4, 6)


class FakeSequenceSLM:
    """Records Select_image calls. Like the SLM, the first trigger wait completes at once without a trigger
    (``spurious_first``). The following waits follow ``waits`` (True = trigger, False = timeout), then time out;
    ``triggers=n`` is ``waits=[True] * n``. ``first_wait_s`` delays the first trigger wait (a real trigger)."""

    def __init__(self, triggers=0, load_error=None, select_error_at=None, waits=None, spurious_first=True, first_wait_s=0.0):
        self.waits = list(waits) if waits is not None else [True] * triggers
        self.load_error = load_error
        self.select_error_at = select_error_at
        self.spurious_first = spurious_first
        self.first_wait_s = first_wait_s
        self.trigger_waits = 0
        self.sequence_length = None
        self.selected = []
        self.loaded = None

    def loadSequence(self, frames):
        if self.load_error:
            raise RuntimeError(self.load_error)
        self.loaded = list(frames)
        self.sequence_length = len(self.loaded)

    def selectFrame(self, frame, *, wait_for_trigger=True, timeout_ms=5000, flip_immediate=False, output_pulse=False):
        self.selected.append((frame, wait_for_trigger, timeout_ms, flip_immediate))
        if self.select_error_at is not None and len(self.selected) == self.select_error_at:
            raise RuntimeError("DLL failure")
        if not wait_for_trigger:
            return True
        self.trigger_waits += 1
        if self.trigger_waits == 1:
            if self.first_wait_s:
                time.sleep(self.first_wait_s)
            if self.spurious_first:
                return True
        return self.waits.pop(0) if self.waits else False

    def lastErrorMessage(self):
        return "timeout"


def _frames(n):
    return [np.full(SHAPE, i, np.uint8) for i in range(n)]


def _run(slm, n, stop=lambda: False, **kwargs):
    return playSequence(slm, _frames(n), timeout_ms=kwargs.pop("timeout_ms", 10), flip_immediate=kwargs.pop("flip_immediate", False), stop_requested=stop, **kwargs)


def test_frames_advance_per_trigger_and_wrap_to_frame_0():
    slm = FakeSequenceSLM(triggers=4)
    armed, counts = [], []
    result = _run(slm, 3, timeout_ms=200, flip_immediate=True, on_armed=lambda: armed.append(True), on_trigger=counts.append)
    assert armed == [True] and len(slm.loaded) == 3 and counts == [1, 2, 3, 4]
    assert [s[0] for s in slm.selected] == [0, 0, 1, 2, 0, 1, 2]
    assert slm.selected[0] == (0, False, 5000, True)
    assert all(s[1:] == (True, 200, True) for s in slm.selected[1:])
    assert (result.reason, result.triggers, result.last_frame, result.message) == (RUN_NO_TRIGGER, 4, 1, "timeout")


def test_spurious_first_completion_is_not_counted():
    slm = FakeSequenceSLM(triggers=0)
    lines, counts = [], []
    result = _run(slm, 2, stop=lambda: len(slm.selected) >= 4, on_trigger=counts.append, log=lines.append)
    assert counts == [] and [s[0] for s in slm.selected] == [0, 0, 1, 1]
    assert lines[0].startswith("Arm: first trigger wait completed after") and lines[0].endswith("(ignored)")
    assert (result.reason, result.triggers, result.last_frame) == (RUN_STOPPED, 0, 0)


def test_real_trigger_during_the_extra_wait_is_counted_and_keeps_alignment():
    slm = FakeSequenceSLM(waits=[True, True], first_wait_s=0.01)
    lines = []
    result = _run(slm, 3, log=lines.append)
    assert "the first trigger came" in lines[0]
    assert [s[0] for s in slm.selected] == [0, 0, 2, 0, 1]
    assert (result.reason, result.triggers, result.last_frame) == (RUN_NO_TRIGGER, 3, 0)


def test_without_spurious_completion_the_run_still_waits_for_the_first_trigger():
    slm = FakeSequenceSLM(waits=[False, False, True], spurious_first=False)
    lines = []
    result = _run(slm, 2, log=lines.append)
    assert lines[0].startswith("Arm: no early completion")
    assert [s[0] for s in slm.selected] == [0, 0, 1, 1, 0]
    assert (result.reason, result.triggers, result.last_frame) == (RUN_NO_TRIGGER, 1, 1)


def test_single_frame_sequence_keeps_frame_0():
    slm = FakeSequenceSLM(triggers=2)
    result = _run(slm, 1)
    assert [s[0] for s in slm.selected] == [0, 0, 0, 0, 0]
    assert (result.reason, result.triggers, result.last_frame) == (RUN_NO_TRIGGER, 2, 0)


def test_stop_ends_after_the_current_wait():
    slm = FakeSequenceSLM(triggers=100)
    result = _run(slm, 2, stop=lambda: len(slm.selected) >= 4)
    assert (result.reason, result.triggers, result.last_frame) == (RUN_STOPPED, 2, 0)


def test_no_timeout_exit_before_the_first_trigger():
    slm = FakeSequenceSLM(waits=[False, False, False, True, True])
    result = _run(slm, 2)
    assert [s[0] for s in slm.selected] == [0, 0, 1, 1, 1, 1, 0, 1]
    assert (result.reason, result.triggers, result.last_frame) == (RUN_NO_TRIGGER, 2, 0)


def test_timeout_after_stop_counts_as_stopped():
    slm = FakeSequenceSLM(triggers=0)
    result = _run(slm, 2, stop=lambda: len(slm.selected) >= 3)
    assert (result.reason, result.triggers, result.last_frame) == (RUN_STOPPED, 0, 0)


def test_load_failure_shows_nothing():
    slm = FakeSequenceSLM(triggers=1, load_error="Load_sequence returned -1")
    armed = []
    result = _run(slm, 2, on_armed=lambda: armed.append(True))
    assert armed == [] and slm.selected == []
    assert (result.reason, result.last_frame) == (RUN_ERROR, None) and "Load_sequence returned -1" in result.message


def test_dll_error_during_run():
    slm = FakeSequenceSLM(triggers=5, select_error_at=4)
    result = _run(slm, 3)
    assert (result.reason, result.triggers, result.last_frame) == (RUN_ERROR, 1, 1) and "DLL failure" in result.message


def test_wait_log():
    slm = FakeSequenceSLM(waits=[False, True, True, True])
    lines = []
    _run(slm, 2, log=lines.append)
    assert [s[0] for s in slm.selected] == [0, 0, 1, 1, 0, 1, 0]
    assert len(lines) == 6
    assert lines[1].startswith("Sequence wait 1: frame 2 selected, no trigger after")
    assert lines[2].startswith("Sequence wait 2: frame 2 selected, trigger after") and lines[3].startswith("Sequence wait 3: frame 1 selected, trigger")


def test_simulated_slm_waits_for_the_first_trigger_until_stop():
    slm = MeadowlarkSLM(1, simulate=True, expected_shape=SHAPE)
    slm.connect()
    waits = []
    try:
        result = _run(slm, 2, timeout_ms=1, stop=lambda: waits.append(1) or len(waits) > 5)
        assert (result.reason, result.triggers, result.last_frame) == (RUN_STOPPED, 0, 0)
        assert np.array_equal(slm.getLastHologram(), _frames(2)[0])
    finally:
        slm.disconnect()


def _terms(**kwargs):
    zeros = np.zeros(SHAPE)
    values = dict(grating_rad=zeros, fresnel_rad=zeros, wfc_rad=None, zernike_rad=None, pupil_mask=None, pupil_outside_rad=None, phase_levels=256)
    values.update(kwargs)
    return FrameTerms(**values)


def test_compose_sums_terms_mod_2pi_as_gray_levels():
    pattern_rad = np.full(SHAPE, np.pi / 2)
    grating_rad = np.full(SHAPE, 2 * np.pi)
    wfc_rad = np.full(SHAPE, np.pi / 4)
    frame = composeSlmFrame(pattern_rad, _terms(grating_rad=grating_rad, wfc_rad=wfc_rad))
    assert frame.dtype == np.uint8 and np.all(frame == 96)


def test_compose_pupil_replaces_phase_outside():
    mask = np.zeros(SHAPE)
    mask[0, 0] = 1
    outside_rad = np.full(SHAPE, np.pi)
    outside_rad[0, 0] = 0
    frame = composeSlmFrame(np.full(SHAPE, np.pi / 2), _terms(pupil_mask=mask, pupil_outside_rad=outside_rad))
    assert frame[0, 0] == 64 and np.all(frame.ravel()[1:] == 128)


def test_terms_copy_is_independent():
    terms = _terms(zernike_rad=np.zeros(SHAPE))
    copied = terms.copy()
    terms.zernike_rad[0, 0] = 1.0
    assert copied.zernike_rad[0, 0] == 0.0 and copied.wfc_rad is None

