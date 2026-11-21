"""Tests for gui/panels/modes/logic.py (no Qt): mode list, ordinals, dynamic hologram list, status text, file reading."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gui import hologram_handler
from gui.panels.modes import DEFAULT_OPERATION_MODE, OPERATION_MODES
from gui.panels.modes.logic import DYNAMIC_EDITING_TEXT, DYNAMIC_READY_TEXT, DynamicMode, HologramSlot, OperationMode, StaticMode, ordinal, readHologramFile


@pytest.mark.parametrize('n, text', [(1, '1st'), (2, '2nd'), (3, '3rd'), (4, '4th'), (11, '11th'), (12, '12th'), (13, '13th'), (21, '21st'), (102, '102nd'), (111, '111th')])
def test_ordinal(n, text):
    assert ordinal(n) == text


def test_dynamic_hologram_list():
    mode = DynamicMode()
    assert mode.holograms == [HologramSlot()]
    assert mode.removeLastSlot() is False and len(mode.holograms) == 1
    assert mode.addSlot() == 1 and mode.addSlot() == 2
    mode.setHologram(1, 'a.pkl', {'phasePattern': 0})
    assert mode.holograms[1] == HologramSlot('a.pkl', {'phasePattern': 0})
    assert mode.removeLastSlot() is True
    assert [s.path for s in mode.holograms] == [None, 'a.pkl']


def test_modes_and_default():
    assert [m.name for m in OPERATION_MODES] == ['Static', 'Dynamic']
    assert DEFAULT_OPERATION_MODE is StaticMode
    for mode_cls in (StaticMode, DynamicMode):
        mode = mode_cls()
        assert isinstance(mode, OperationMode)
        mode.activate()
        mode.deactivate()


def test_status_text():
    assert StaticMode().statusText(SimpleNamespace(dynamicIsReady=True)) is None
    assert DynamicMode().statusText(SimpleNamespace(dynamicIsReady=True)) == DYNAMIC_READY_TEXT
    assert DynamicMode().statusText(SimpleNamespace(dynamicIsReady=False)) == DYNAMIC_EDITING_TEXT


def test_read_hologram_file(tmp_path):
    pkl = str(tmp_path / 'h.pkl')
    hologram_handler.writeTrapsDict(pkl, {'phasePattern': np.ones((2, 2))})
    np.testing.assert_array_equal(readHologramFile(pkl)['phasePattern'], np.ones((2, 2)))
    bad = str(tmp_path / 'bad.pkl')
    hologram_handler.writeTrapsDict(bad, {'traps_pos': 1})
    with pytest.raises(ValueError):
        readHologramFile(bad)
    with pytest.raises(ValueError):
        readHologramFile(str(tmp_path / 'x.json'))


def test_modes_logic_does_not_load_qt():
    code = "import sys; import gui.panels.modes.logic, gui.panels.modes; print('PyQt5.QtWidgets' in sys.modules)"
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1], check=True)
    assert out.stdout.strip() == 'False'


def test_clamp_trigger_timeout():
    from gui.panels.modes.logic import MAX_TRIGGER_TIMEOUT_MS, MIN_TRIGGER_TIMEOUT_MS, clampTriggerTimeoutMs
    assert clampTriggerTimeoutMs(250) == 250 and clampTriggerTimeoutMs(250.4) == 250
    assert clampTriggerTimeoutMs(0) == MIN_TRIGGER_TIMEOUT_MS and clampTriggerTimeoutMs(1e9) == MAX_TRIGGER_TIMEOUT_MS


def test_dynamic_holograms_profile_round_trip(tmp_path):
    good = tmp_path / "frame.npy"
    np.save(good, np.ones((2, 3)))
    mode = DynamicMode()
    assert mode.profileHolograms() == ["none"]
    problems = mode.loadProfileHolograms([str(good), "none", str(tmp_path / "missing.npy"), 5])
    assert len(problems) == 1 and "frame 3" in problems[0] and "missing.npy" in problems[0]
    assert [slot.path for slot in mode.holograms] == [str(good), None, None, None]
    assert np.array_equal(mode.holograms[0].data["phasePattern"], np.ones((2, 3)))
    assert mode.profileHolograms() == [str(good), "none", "none", "none"]
    assert mode.loadProfileHolograms([]) == [] and len(mode.holograms) == 1 and mode.holograms[0].path is None
