"""Tests for libslm/slm.py MeadowlarkSLM: simulate mode and the hardware connect path against a fake Blink DLL."""

import numpy as np
import pytest

import libslm.slm as slm_module
from libslm.slm import MeadowlarkSLM, SerialMismatchError


class _DLLFunction:
    """Callable that accepts ctypes' ``argtypes`` / ``restype`` attributes, like a real DLL export."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args):
        return self.fn(*args)


class FakeBlinkDLL:
    """Stands in for Blink_C_wrapper.dll; records the calls the driver makes."""

    def __init__(self, n_boards=1, serial=6801, has_serial=True):
        self.calls = []
        self.n_boards = n_boards
        self.serial = serial
        exports = {"Create_SDK": self._create_sdk, "Load_LUT_file": self._load_lut, "Write_image": self._write_image, "ImageWriteComplete": self._write_complete, "Read_SLM_temperature": lambda board: 33.5, "Delete_SDK": lambda: self.calls.append("Delete_SDK"), "Get_last_error_message": lambda: b""}
        if has_serial:
            exports["Read_Serial_Number"] = self._read_serial
        for name, fn in exports.items():
            setattr(self, name, _DLLFunction(fn))

    def _create_sdk(self, bit_depth, n_boards_ref, constructed_ok_ref, *args):
        self.calls.append("Create_SDK")
        n_boards_ref._obj.value = self.n_boards
        constructed_ok_ref._obj.value = 1

    def _read_serial(self, board):
        self.calls.append(f"Read_Serial_Number({board})")
        return self.serial

    def _load_lut(self, board, path):
        self.calls.append(f"Load_LUT_file({board})")
        return 0

    def _write_image(self, board, *args):
        self.calls.append(f"Write_image({board})")
        return 0

    def _write_complete(self, board, timeout):
        self.calls.append(f"ImageWriteComplete({board})")
        return 0


@pytest.fixture
def hw(tmp_path, monkeypatch):
    """Factory for a hardware-mode MeadowlarkSLM wired to a FakeBlinkDLL (DLL / LUT files exist in tmp_path)."""
    dll_path, lut_path = tmp_path / "Blink_C_wrapper.dll", tmp_path / "test.LUT"
    dll_path.write_bytes(b"")
    lut_path.write_bytes(b"")
    created = []

    def make(fake, **kwargs):
        monkeypatch.setattr(MeadowlarkSLM, "_load_blink_dll", staticmethod(lambda path: fake))
        slm = MeadowlarkSLM(dll_path=dll_path, lut_path=lut_path, **kwargs)
        created.append(slm)
        return slm

    yield make
    for slm in created:
        slm.disconnect()
    MeadowlarkSLM._active = None


def test_simulate_round_trip():
    slm = MeadowlarkSLM(simulate=True)
    slm.connect()
    try:
        frame = np.full((slm_module.H, slm_module.W), 7, dtype=np.uint8)
        slm.uploadHologram(frame)
        assert slm.isSimulated() and slm.serial is None
        assert slm.readTemperature() == slm_module.SIMULATED_TEMPERATURE_C
        assert np.array_equal(slm.getLastHologram(), frame)
    finally:
        slm.disconnect()
    assert not slm.connected


def test_only_one_connected_instance():
    a, b = MeadowlarkSLM(simulate=True), MeadowlarkSLM(2, simulate=True)
    a.connect()
    try:
        with pytest.raises(RuntimeError, match="already in use"):
            b.connect()
    finally:
        a.disconnect()
    b.connect()
    b.disconnect()


def test_connect_reads_serial(hw):
    fake = FakeBlinkDLL(serial=6801)
    slm = hw(fake)
    slm.connect()
    assert slm.connected and slm.serial == 6801
    assert fake.calls == ["Create_SDK", "Read_Serial_Number(1)", "Load_LUT_file(1)"]


def test_matching_serial_connects(hw):
    fake = FakeBlinkDLL(serial=6801)
    slm = hw(fake, expected_serial=6801)
    slm.connect()
    assert slm.connected and slm.serial == 6801


def test_serial_mismatch_refuses_before_lut(hw):
    fake = FakeBlinkDLL(serial=1234)
    slm = hw(fake, expected_serial=6801)
    with pytest.raises(SerialMismatchError) as err:
        slm.connect()
    assert (err.value.board, err.value.expected, err.value.actual) == (1, 6801, 1234)
    assert fake.calls == ["Create_SDK", "Read_Serial_Number(1)", "Delete_SDK"]
    assert not slm.connected and slm.serial is None and MeadowlarkSLM._active is None


def test_serial_mismatch_callback_yes_continues(hw):
    fake = FakeBlinkDLL(serial=1234)
    asked = []
    slm = hw(fake, expected_serial=6801, on_serial_mismatch=lambda exp, act: asked.append((exp, act)) or True)
    slm.connect()
    assert asked == [(6801, 1234)]
    assert slm.connected and slm.serial == 1234
    assert fake.calls == ["Create_SDK", "Read_Serial_Number(1)", "Load_LUT_file(1)"]


def test_serial_mismatch_callback_no_aborts(hw):
    fake = FakeBlinkDLL(serial=1234)
    slm = hw(fake, expected_serial=6801, on_serial_mismatch=lambda exp, act: False)
    with pytest.raises(SerialMismatchError):
        slm.connect()
    assert fake.calls == ["Create_SDK", "Read_Serial_Number(1)", "Delete_SDK"]


def test_matching_serial_does_not_ask(hw):
    asked = []
    slm = hw(FakeBlinkDLL(serial=6801), expected_serial=6801, on_serial_mismatch=lambda exp, act: asked.append(1) or False)
    slm.connect()
    assert slm.connected and asked == []


@pytest.mark.parametrize("fake", [FakeBlinkDLL(has_serial=False), FakeBlinkDLL(serial=-1)])
def test_unreadable_serial_is_not_verified(hw, fake):
    slm = hw(fake, expected_serial=6801)
    slm.connect()
    assert slm.connected and slm.serial is None


def test_board_out_of_range(hw):
    fake = FakeBlinkDLL(n_boards=1)
    slm = hw(fake, board=2)
    with pytest.raises(RuntimeError, match="Board 2 not found"):
        slm.connect()
    assert fake.calls == ["Create_SDK", "Delete_SDK"]


def test_upload_uses_own_board(hw):
    fake = FakeBlinkDLL(n_boards=2)
    slm = hw(fake, board=2)
    slm.connect()
    slm.uploadHologram(np.zeros((slm_module.H, slm_module.W), dtype=np.uint8))
    assert fake.calls[-2:] == ["Write_image(2)", "ImageWriteComplete(2)"]
    assert slm.readTemperature() == 33.5
