"""Tests for gui/sequence_run.py (trigger loop of the dynamic mode) and gui/slm_frame.py (frame composition)."""

import numpy as np
import pytest

from gui.sequence_run import RUN_ERROR, RUN_NO_TRIGGER, RUN_STOPPED, playSequence
from gui.slm_frame import FrameTerms, composeSlmFrame
from libslm.slm import MeadowlarkSLM

SHAPE = (4, 6)


class FakeSequenceSLM:
    """Records Select_image calls; ``triggers`` triggers arrive, then every wait times out."""

    def __init__(self, triggers, load_error=None, select_error_at=None):
        self.triggers_left = triggers
        self.load_error = load_error
        self.select_error_at = select_error_at
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
        if self.triggers_left == 0:
            return False
        self.triggers_left -= 1
        return True

    def lastErrorMessage(self):
        return "timeout"


def _frames(n):
    return [np.full(SHAPE, i, np.uint8) for i in range(n)]


def test_frames_advance_per_trigger_and_wrap_to_frame_0():
    slm = FakeSequenceSLM(triggers=4)
    armed = []
    result = playSequence(slm, _frames(3), timeout_ms=200, flip_immediate=True, stop_requested=lambda: False, on_armed=lambda: armed.append(True))
    assert armed == [True] and len(slm.loaded) == 3
    assert [s[0] for s in slm.selected] == [0, 1, 2, 0, 1, 2]
    assert slm.selected[0] == (0, False, 5000, True)
    assert all(s[1:] == (True, 200, True) for s in slm.selected[1:])
    assert (result.reason, result.triggers, result.last_frame, result.message) == (RUN_NO_TRIGGER, 4, 1, "timeout")


def test_single_frame_sequence_keeps_frame_0():
    slm = FakeSequenceSLM(triggers=2)
    result = playSequence(slm, _frames(1), timeout_ms=10, flip_immediate=False, stop_requested=lambda: False)
    assert [s[0] for s in slm.selected] == [0, 0, 0, 0]
    assert (result.reason, result.triggers, result.last_frame) == (RUN_NO_TRIGGER, 2, 0)


def test_stop_ends_after_the_current_wait():
    slm = FakeSequenceSLM(triggers=100)
    result = playSequence(slm, _frames(2), timeout_ms=10, flip_immediate=False, stop_requested=lambda: len(slm.selected) >= 3)
    assert (result.reason, result.triggers, result.last_frame) == (RUN_STOPPED, 2, 0)


def test_timeout_after_stop_counts_as_stopped():
    slm = FakeSequenceSLM(triggers=0)
    result = playSequence(slm, _frames(2), timeout_ms=10, flip_immediate=False, stop_requested=lambda: len(slm.selected) >= 2)
    assert (result.reason, result.triggers, result.last_frame) == (RUN_STOPPED, 0, 0)


def test_load_failure_shows_nothing():
    slm = FakeSequenceSLM(triggers=1, load_error="Load_sequence returned -1")
    armed = []
    result = playSequence(slm, _frames(2), timeout_ms=10, flip_immediate=False, stop_requested=lambda: False, on_armed=lambda: armed.append(True))
    assert armed == [] and slm.selected == []
    assert (result.reason, result.last_frame) == (RUN_ERROR, None) and "Load_sequence returned -1" in result.message


def test_dll_error_during_run():
    slm = FakeSequenceSLM(triggers=5, select_error_at=3)
    result = playSequence(slm, _frames(3), timeout_ms=10, flip_immediate=False, stop_requested=lambda: False)
    assert (result.reason, result.triggers, result.last_frame) == (RUN_ERROR, 1, 1) and "DLL failure" in result.message


def test_simulated_slm_run_ends_after_one_timeout():
    slm = MeadowlarkSLM(simulate=True, expected_shape=SHAPE)
    slm.connect()
    try:
        result = playSequence(slm, _frames(2), timeout_ms=1, flip_immediate=False, stop_requested=lambda: False)
        assert (result.reason, result.triggers, result.last_frame) == (RUN_NO_TRIGGER, 0, 0)
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
