"""Tests for GPU device selection in libslm.gpu (no GPU needed: fake devices)."""
from types import SimpleNamespace

import pytest

gpu = pytest.importorskip("libslm.gpu")


def _dev(name):
    return SimpleNamespace(name=name)


def test_prefers_named_device_even_if_not_first():
    devices = [_dev("NVIDIA T1000"), _dev("NVIDIA RTX 4000 Ada Generation")]
    assert gpu.select_device(devices).name == "NVIDIA RTX 4000 Ada Generation"


def test_falls_back_to_first_device():
    devices = [_dev("NVIDIA T1000"), _dev("Intel HD Graphics 530")]
    assert gpu.select_device(devices).name == "NVIDIA T1000"


def test_no_device_raises():
    with pytest.raises(RuntimeError):
        gpu.select_device([])


def test_default_pupil_gives_single_trap_grating():
    """Without a pupil, one trap must give the plain grating k_x*i + k_y*j."""
    np = pytest.importorskip("numpy")
    shape = (60, 80)
    k_list = np.array([[0.1, 0.05, 0.0]])
    phi = gpu.trap_phase_to_slm(k_list, np.zeros(1), shape=shape)
    j, i = np.mgrid[0:shape[0], 0:shape[1]]
    expected = 0.1 * i + 0.05 * j
    wrapped_diff = np.angle(np.exp(1j * (phi - expected)))
    assert not np.isnan(phi).any()
    assert np.abs(wrapped_diff).max() < 1e-4
