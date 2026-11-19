"""Tests for libslm/slm.py MeadowlarkSLM: simulate mode and the hardware connect path against a fake Blink DLL."""

import numpy as np
import pytest

import libslm.slm as slm_module
from libslm.slm import MeadowlarkSLM, SerialMismatchError, SlmSizeMismatchError

H, W = 1024, 1024
SHAPE = (H, W)


class _DLLFunction:
    """Callable that accepts ctypes' ``argtypes`` / ``restype`` attributes, like a real DLL export."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args):
        return self.fn(*args)


class FakeBlinkDLL:
    """Stands in for Blink_C_wrapper.dll; records the calls the driver makes."""

    def __init__(self, n_boards=1, serial=6801, has_serial=True, width=W, height=H, depth=8, has_geometry=True, has_sequence=True):
        self.calls = []
        self.n_boards = n_boards
        self.serial = serial
        self.written_sizes = []
        self.sequence = None
        self.load_sequence_rc = 0
        self.select_rc = 0
        self.select_args = []
        self.write_complete_rc = 0
        self.write_complete_timeouts = []
        self.write_complete_delay_s = 0.0
        exports = {"Create_SDK": self._create_sdk, "Load_LUT_file": self._load_lut, "Write_image": self._write_image, "ImageWriteComplete": self._write_complete, "Read_SLM_temperature": lambda board: 33.5, "Delete_SDK": lambda: self.calls.append("Delete_SDK"), "Get_last_error_message": lambda: b""}
        if has_serial:
            exports["Read_Serial_Number"] = self._read_serial
        if has_geometry:
            exports["Get_image_width"] = lambda board: width
            exports["Get_image_height"] = lambda board: height
            exports["Get_image_depth"] = lambda board: depth
        if has_sequence:
            exports["Load_sequence"] = self._load_sequence
            exports["Select_image"] = self._select_image
        for name, fn in exports.items():
            setattr(self, name, _DLLFunction(fn))

    def _load_sequence(self, board, buf, image_size, list_length, *args):
        self.calls.append(f"Load_sequence({board})")
        self.sequence = (image_size, list_length, np.ctypeslib.as_array(buf, shape=(image_size * list_length,)).copy(), args)
        return self.load_sequence_rc

    def _select_image(self, board, frame, *args):
        self.calls.append(f"Select_image({board}, {frame})")
        self.select_args.append(args)
        return self.select_rc

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

    def _write_image(self, board, buf, image_size, *args):
        self.calls.append(f"Write_image({board})")
        self.written_sizes.append(image_size)
        return 0

    def _write_complete(self, board, timeout):
        self.calls.append(f"ImageWriteComplete({board})")
        self.write_complete_timeouts.append(timeout)
        if self.write_complete_delay_s:
            import time
            time.sleep(self.write_complete_delay_s)
        return self.write_complete_rc


@pytest.fixture
def hw(tmp_path, monkeypatch):
    """Factory for a hardware-mode MeadowlarkSLM wired to a FakeBlinkDLL (DLL / LUT files exist in tmp_path)."""
    dll_path, lut_path = tmp_path / "Blink_C_wrapper.dll", tmp_path / "test.LUT"
    dll_path.write_bytes(b"")
    lut_path.write_bytes(b"")
    created = []

    def make(fake, **kwargs):
        monkeypatch.setattr(MeadowlarkSLM, "_load_blink_dll", staticmethod(lambda path: fake))
        kwargs.setdefault("lut_path", lut_path)
        slm = MeadowlarkSLM(dll_path=dll_path, **kwargs)
        created.append(slm)
        return slm

    yield make
    for slm in created:
        slm.disconnect()
    MeadowlarkSLM._active = None


def test_simulate_round_trip():
    slm = MeadowlarkSLM(simulate=True, expected_shape=SHAPE)
    slm.connect()
    try:
        frame = np.full((H, W), 7, dtype=np.uint8)
        slm.uploadHologram(frame)
        assert slm.connected and slm.simulate and slm.serial is None
        assert slm.readTemperature() == slm_module.SIMULATED_TEMPERATURE_C
        assert np.array_equal(slm.getLastHologram(), frame)
    finally:
        slm.disconnect()
    assert not slm.connected


def test_load_correction_pattern_in_radians(tmp_path):
    from PIL import Image
    path = tmp_path / "WFC.bmp"
    grey = np.zeros((H, W), dtype=np.uint8)
    grey[0, 0], grey[0, 1] = 255, 51
    Image.fromarray(grey).save(path)
    c = slm_module.loadCorrectionPattern(path, SHAPE)
    assert c.shape == (H, W) and c.dtype == np.float32
    assert np.isclose(c[0, 0], 2 * np.pi) and np.isclose(c[0, 1], 51 * 2 * np.pi / 255) and c[1, 1] == 0


def test_missing_wfc_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="WFC"):
        slm_module.loadCorrectionPattern(tmp_path / "missing_WFC.bmp", SHAPE)


def test_wrong_size_wfc_is_rejected(tmp_path):
    from PIL import Image
    path = tmp_path / "small_WFC.bmp"
    Image.fromarray(np.zeros((10, 20), dtype=np.uint8)).save(path)
    with pytest.raises(ValueError, match="shape"):
        slm_module.loadCorrectionPattern(path, SHAPE)


def test_connect_without_lut_loads_none(hw):
    fake = FakeBlinkDLL()
    slm = hw(fake, lut_path=None)
    slm.connect()
    assert slm.connected and slm.lut_path is None
    assert fake.calls == ["Create_SDK", "Read_Serial_Number(1)"]


def test_missing_lut_fails_before_sdk(hw, tmp_path):
    fake = FakeBlinkDLL()
    slm = hw(fake, lut_path=tmp_path / "missing.LUT")
    with pytest.raises(FileNotFoundError, match="LUT"):
        slm.connect()
    assert fake.calls == [] and not slm.connected


def test_only_one_connected_instance():
    a, b = MeadowlarkSLM(simulate=True, expected_shape=SHAPE), MeadowlarkSLM(2, simulate=True, expected_shape=SHAPE)
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
    slm.uploadHologram(np.zeros((H, W), dtype=np.uint8))
    assert fake.calls[-2:] == ["Write_image(2)", "ImageWriteComplete(2)"]
    assert slm.readTemperature() == 33.5


def test_no_hard_coded_lut_or_wfc():
    assert not hasattr(slm_module, "LUT_FILE") and not hasattr(slm_module, "WFC_FILE")
    assert MeadowlarkSLM(simulate=True, expected_shape=SHAPE).lut_path is None


def _write_lut(path, levels=256):
    path.write_text("".join(f"{i} {1000 + 10 * i}\n" for i in range(levels)), encoding="ascii")
    return path


def test_check_lut_file(tmp_path):
    assert slm_module.checkLutFile(_write_lut(tmp_path / "ok.LUT")) == 256
    meadowlark_lut = slm_module.Path(r"C:\Program Files\Meadowlark Optics\Blink OverDrive Plus\LUT Files\slm6801_at785_33C.LUT")
    if meadowlark_lut.is_file():
        assert slm_module.checkLutFile(meadowlark_lut) == 256
    bad_level = tmp_path / "gap.LUT"
    bad_level.write_text("0 1\n2 3\n", encoding="ascii")
    bad_text = tmp_path / "text.LUT"
    bad_text.write_text("0 1\nx y\n", encoding="ascii")
    empty = tmp_path / "empty.LUT"
    empty.write_text("\n", encoding="ascii")
    for path in (bad_level, bad_text, empty):
        with pytest.raises(ValueError):
            slm_module.checkLutFile(path)
    with pytest.raises(FileNotFoundError):
        slm_module.checkLutFile(tmp_path / "missing.LUT")


def test_load_lut_reloads_controller(hw, tmp_path):
    fake = FakeBlinkDLL()
    slm = hw(fake)
    slm.connect()
    new_lut = _write_lut(tmp_path / "other.LUT")
    slm.loadLut(new_lut)
    assert fake.calls[-1] == "Load_LUT_file(1)" and fake.calls.count("Load_LUT_file(1)") == 2
    assert slm.lut_path == new_lut
    with pytest.raises(FileNotFoundError):
        slm.loadLut(tmp_path / "missing.LUT")
    assert slm.lut_path == new_lut


def test_load_lut_requires_connection_and_simulate_only_remembers(tmp_path):
    new_lut = _write_lut(tmp_path / "other.LUT")
    slm = MeadowlarkSLM(simulate=True, expected_shape=SHAPE)
    with pytest.raises(RuntimeError):
        slm.loadLut(new_lut)
    slm.connect()
    try:
        slm.loadLut(new_lut)
        assert slm.lut_path == new_lut
    finally:
        slm.disconnect()


def test_image_size_is_read_from_hardware(hw):
    fake = FakeBlinkDLL(width=1920, height=1152, depth=12)
    slm = hw(fake)
    slm.connect()
    assert (slm.image_shape, slm.bit_depth) == ((1152, 1920), 12)
    slm.uploadHologram(np.zeros((1152, 1920), dtype=np.uint8))
    assert fake.written_sizes == [1920 * 1152]
    with pytest.raises(ValueError, match="1152, 1920"):
        slm.uploadHologram(np.zeros((H, W), dtype=np.uint8))
    slm.disconnect()
    assert slm.image_shape is None and slm.bit_depth is None


def test_size_mismatch_refuses_before_lut(hw):
    fake = FakeBlinkDLL(width=1920, height=1152)
    slm = hw(fake, expected_shape=SHAPE)
    with pytest.raises(SlmSizeMismatchError, match="1920 x 1152"):
        slm.connect()
    assert "Load_LUT_file(1)" not in fake.calls and fake.calls[-1] == "Delete_SDK"
    assert not slm.connected and slm.image_shape is None


def test_matching_size_connects(hw):
    fake = FakeBlinkDLL()
    slm = hw(fake, expected_shape=SHAPE)
    slm.connect()
    assert slm.connected and slm.image_shape == SHAPE


@pytest.mark.parametrize("kwargs", [dict(has_geometry=False), dict(width=0), dict(height=-1)])
def test_missing_or_implausible_size_refuses(hw, kwargs):
    fake = FakeBlinkDLL(**kwargs)
    slm = hw(fake)
    with pytest.raises(RuntimeError, match="image size"):
        slm.connect()
    assert not slm.connected and fake.calls[-1] == "Delete_SDK"


def test_simulate_needs_expected_shape():
    with pytest.raises(ValueError, match="expected_shape"):
        MeadowlarkSLM(simulate=True).connect()
    slm = MeadowlarkSLM(simulate=True, expected_shape=(600, 800))
    slm.connect()
    try:
        assert slm.image_shape == (600, 800)
        slm.uploadHologram(np.zeros((600, 800), dtype=np.uint8))
    finally:
        slm.disconnect()


def test_image_shape_only_while_connected(hw):
    slm = hw(FakeBlinkDLL(), expected_shape=SHAPE)
    assert slm.image_shape is None and slm.expected_shape == SHAPE
    slm.connect()
    assert slm.image_shape == SHAPE
    slm.disconnect()
    assert slm.image_shape is None and slm.expected_shape == SHAPE


def test_no_hard_coded_frame_size():
    for name in ("W", "H", "IMAGE_SIZE"):
        assert not hasattr(slm_module, name)


def test_read_pixel_pitch(hw):
    fake = FakeBlinkDLL()
    fake.Get_pixel_pitch = _DLLFunction(lambda board: 17.0)
    slm = hw(fake)
    with pytest.raises(RuntimeError):
        slm.readPixelPitchUm()
    slm.connect()
    assert slm.readPixelPitchUm() == 17.0
    sim = MeadowlarkSLM(2, simulate=True, expected_shape=SHAPE)
    slm.disconnect()
    sim.connect()
    try:
        assert sim.readPixelPitchUm() is None
    finally:
        sim.disconnect()


SEQ_SHAPE = (4, 6)


def _seq_frames(n, shape=SEQ_SHAPE):
    return [np.full(shape, i, dtype=np.uint8) for i in range(n)]


@pytest.fixture
def seq_slm(hw):
    fake = FakeBlinkDLL(width=SEQ_SHAPE[1], height=SEQ_SHAPE[0])
    slm = hw(fake, lut_path=None)
    slm.connect()
    return slm, fake


def test_load_sequence_sends_frames_back_to_back(seq_slm):
    slm, fake = seq_slm
    frames = _seq_frames(3)
    frames[1][0, 1] = 200
    slm.loadSequence(frames)
    image_size, n, data, args = fake.sequence
    assert (image_size, n) == (24, 3) and slm.sequence_length == 3
    assert np.array_equal(data, np.stack(frames).ravel())
    assert args == (0, 0, 0, 0, slm_module.TRIGGER_TIMEOUT_MS)


def test_load_sequence_accepts_a_3d_array(seq_slm):
    slm, fake = seq_slm
    slm.loadSequence(np.stack(_seq_frames(2)))
    assert slm.sequence_length == 2 and fake.sequence[1] == 2


@pytest.mark.parametrize("frames", [[], [np.zeros((4, 5), np.uint8)], np.zeros((4, 6), np.uint8)])
def test_load_sequence_rejects_bad_frames(seq_slm, frames):
    slm, fake = seq_slm
    with pytest.raises(ValueError):
        slm.loadSequence(frames)
    assert fake.sequence is None and slm.sequence_length is None


def test_load_sequence_rejects_too_many_frames(seq_slm, monkeypatch):
    slm, fake = seq_slm
    monkeypatch.setattr(slm_module, "MAX_SEQUENCE_FRAMES", 2)
    with pytest.raises(ValueError, match="1 … 2"):
        slm.loadSequence(_seq_frames(3))
    assert fake.sequence is None


def test_load_sequence_failure_leaves_no_sequence(seq_slm):
    slm, fake = seq_slm
    slm.loadSequence(_seq_frames(2))
    fake.load_sequence_rc = -1
    with pytest.raises(RuntimeError, match="Load_sequence returned -1"):
        slm.loadSequence(_seq_frames(2))
    assert slm.sequence_length is None


def test_load_sequence_needs_the_dll_functions(hw):
    slm = hw(FakeBlinkDLL(width=SEQ_SHAPE[1], height=SEQ_SHAPE[0], has_sequence=False), lut_path=None)
    slm.connect()
    with pytest.raises(RuntimeError, match="Load_sequence"):
        slm.loadSequence(_seq_frames(1))


def test_select_frame_passes_trigger_options(seq_slm):
    slm, fake = seq_slm
    slm.loadSequence(_seq_frames(3))
    assert slm.selectFrame(2, timeout_ms=150, output_pulse=True)
    assert slm.selectFrame(0, wait_for_trigger=False, flip_immediate=True)
    assert fake.calls[-4:] == ["Select_image(1, 2)", "ImageWriteComplete(1)", "Select_image(1, 0)", "ImageWriteComplete(1)"]
    assert fake.write_complete_timeouts[-2:] == [150, slm_module.TRIGGER_TIMEOUT_MS]
    assert fake.select_args == [(1, 0, 1, 0, 150), (0, 1, 0, 0, slm_module.TRIGGER_TIMEOUT_MS)]


def test_select_frame_reports_failure(seq_slm):
    slm, fake = seq_slm
    slm.loadSequence(_seq_frames(1))
    fake.select_rc = -1
    assert slm.selectFrame(0, timeout_ms=10) is False
    assert fake.calls[-1] == "Select_image(1, 0)"


def test_select_frame_no_trigger_reported_by_write_complete(seq_slm):
    slm, fake = seq_slm
    slm.loadSequence(_seq_frames(1))
    fake.write_complete_rc = -1
    assert slm.selectFrame(0, timeout_ms=10) is False


def test_select_frame_wait_reaching_timeout_counts_as_no_trigger(seq_slm):
    slm, fake = seq_slm
    slm.loadSequence(_seq_frames(1))
    fake.write_complete_delay_s = 0.03
    assert slm.selectFrame(0, timeout_ms=20) is False
    assert slm.selectFrame(0, wait_for_trigger=False, timeout_ms=20) is True


def test_select_frame_checks_sequence_and_index(seq_slm):
    slm, fake = seq_slm
    with pytest.raises(RuntimeError, match="No frame sequence"):
        slm.selectFrame(0)
    slm.loadSequence(_seq_frames(2))
    for frame in (-1, 2):
        with pytest.raises(ValueError, match="out of range"):
            slm.selectFrame(frame)
    with pytest.raises(ValueError, match="timeout_ms"):
        slm.selectFrame(0, timeout_ms=-1)
    assert not any(c.startswith("Select_image") for c in fake.calls)


def test_upload_discards_the_sequence(seq_slm):
    slm, fake = seq_slm
    slm.loadSequence(_seq_frames(2))
    slm.uploadHologram(np.zeros(SEQ_SHAPE, np.uint8))
    assert slm.sequence_length is None
    with pytest.raises(RuntimeError, match="No frame sequence"):
        slm.selectFrame(0)


def test_disconnect_discards_the_sequence(seq_slm):
    slm, fake = seq_slm
    slm.loadSequence(_seq_frames(2))
    slm.disconnect()
    assert slm.sequence_length is None


def test_sequence_requires_connection():
    slm = MeadowlarkSLM(simulate=True, expected_shape=SEQ_SHAPE)
    with pytest.raises(RuntimeError, match="not connected"):
        slm.loadSequence(_seq_frames(1))
    with pytest.raises(RuntimeError, match="not connected"):
        slm.selectFrame(0)


def test_simulated_sequence():
    slm = MeadowlarkSLM(simulate=True, expected_shape=SEQ_SHAPE)
    slm.connect()
    try:
        slm.loadSequence(_seq_frames(3))
        assert slm.sequence_length == 3
        assert slm.selectFrame(2, wait_for_trigger=False)
        assert np.array_equal(slm.getLastHologram(), np.full(SEQ_SHAPE, 2, np.uint8))
        assert slm.selectFrame(1, timeout_ms=1) is False
        assert np.array_equal(slm.getLastHologram(), np.full(SEQ_SHAPE, 2, np.uint8))
        assert slm.lastErrorMessage() == ""
    finally:
        slm.disconnect()
