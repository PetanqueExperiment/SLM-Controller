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
