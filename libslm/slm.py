"""Meadowlark XY Phase 1024×1024 — minimal SDK upload via Blink_C_wrapper.dll (ctypes).

Requires Meadowlark Blink / PCIe install or USB SDK: Blink_C_wrapper.dll. The LUT file is chosen per device by the
caller (GUI: Device ▸ Set LUT…); without one, no LUT is loaded and the controller keeps the LUT it already has.
See PCIe User Manual §4 (Create_SDK, Load_LUT_file, Write_image, ImageWriteComplete, Delete_SDK).

Usage for live GUI: create a :class:`MeadowlarkSLM` for the board, call :meth:`MeadowlarkSLM.connect` once
when opening the device, then :meth:`MeadowlarkSLM.uploadHologram` for each frame, and
:meth:`MeadowlarkSLM.disconnect` when shutting down. Per-frame work is only Write_image +
ImageWriteComplete; Create_SDK / Load_LUT / Delete_SDK run once per session.

For offline testing without hardware, create it with ``simulate=True`` (the GUI does this when the
Simulated SLM is selected in the device list). Uploads stay in memory and temperature returns a stub
value. Simulation is never selected automatically.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, ClassVar

import numpy as np
from ctypes import CDLL, POINTER, byref, c_char_p, c_double, c_int, c_ubyte, c_uint
from PIL import Image


BLINK_DLL: Path = Path(r"C:\Program Files\Meadowlark Optics\Blink OverDrive Plus\SDK\Blink_C_wrapper.dll")

BOARD: int = 1
SLM_BIT_DEPTH: int = 8  # 8-bit frames; manual lists 8 for 512, 12 for 1920 — use 8 for 1024×1024 8-bit API
TRIGGER_TIMEOUT_MS: int = 5000
# Nominal Meadowlark operating temperature used by the offline stub (°C).
SIMULATED_TEMPERATURE_C: float = 33.0


def checkLutFile(path: Path) -> int:
    """Check that ``path`` is a Blink LUT text file: one "<grey level> <drive value>" line per level, levels 0, 1, 2, …

    Returns the number of levels. Raises FileNotFoundError / ValueError if the file cannot be used.
    """
    if not path.is_file():
        raise FileNotFoundError(f"LUT not found: {path}")
    levels: list[int] = []
    with open(path, "r", encoding="ascii", errors="strict") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 2 or not all(p.isdigit() for p in parts):
                raise ValueError(f"line {line_no}: expected '<grey level> <drive value>', got {line.strip()!r}")
            levels.append(int(parts[0]))
    if not levels:
        raise ValueError("the file has no entries")
    if levels != list(range(len(levels))):
        raise ValueError("grey levels must run 0, 1, 2, … without gaps")
    return len(levels)


def loadCorrectionPattern(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Manufacturer correction BMP as a float32 phase map in radians (gray level × 2π / 255) of ``shape`` = (height, width)."""
    if not path.is_file():
        raise FileNotFoundError(f"WFC correction not found: {path}")
    with Image.open(path) as im:
        if im.mode != "L":
            im = im.convert("L")
        pattern: np.ndarray = np.asarray(im, dtype=np.float32) * 2 * np.pi / 255
    if pattern.shape != tuple(shape):
        raise ValueError(f"WFC BMP {path} has shape {pattern.shape}, expected {tuple(shape)}")
    return pattern


class SerialMismatchError(RuntimeError):
    """The SLM found on the board is not the one expected (different serial number)."""

    def __init__(self, board: int, expected: int, actual: int) -> None:
        super().__init__(f"Board {board}: expected SLM serial number {expected}, found {actual}.")
        self.board: int = board
        self.expected: int = expected
        self.actual: int = actual


class SlmSizeMismatchError(RuntimeError):
    """The SLM reports a different image size than the one the caller computes holograms for."""

    def __init__(self, board: int, expected: tuple[int, int], actual: tuple[int, int]) -> None:
        super().__init__(f"Board {board}: the SLM is {actual[1]} x {actual[0]} px, but holograms are {expected[1]} x {expected[0]} px (parameters.json).")
        self.board: int = board
        self.expected: tuple[int, int] = expected
        self.actual: tuple[int, int] = actual


class MeadowlarkSLM:
    """One Meadowlark SLM on a Blink PCIe board.

    The Blink SDK is a single instance per process (Create_SDK / Delete_SDK are global in the DLL),
    so only one MeadowlarkSLM may be connected at a time.

    If ``expected_serial`` is given, :meth:`connect` reads the serial number right after Create_SDK, before the LUT
    or any image reaches the SLM. On a mismatch it calls ``on_serial_mismatch(expected, actual)``: True continues
    connecting, False (or no callback) closes the SDK again and raises SerialMismatchError.

    The image size is read from the hardware at connect (Get_image_width / Get_image_height) and stored as
    ``width`` / ``height``; frames must have that shape. If ``image_shape`` = (height, width) is given and the hardware
    differs, :meth:`connect` raises SlmSizeMismatchError before the LUT or any image reaches the SLM. In simulate mode
    there is no hardware, so ``image_shape`` is required and becomes the size.
    """

    _active: ClassVar[MeadowlarkSLM | None] = None

    def __init__(self, board: int = BOARD, *, dll_path: Path = BLINK_DLL, lut_path: Path | None = None, simulate: bool = False, expected_serial: int | None = None, on_serial_mismatch: Callable[[int, int], bool] | None = None, image_shape: tuple[int, int] | None = None) -> None:
        self.board: int = board
        self.dll_path: Path = dll_path
        self.lut_path: Path | None = lut_path
        self.simulate: bool = simulate
        self.expected_serial: int | None = expected_serial
        self.on_serial_mismatch: Callable[[int, int], bool] | None = on_serial_mismatch
        self.serial: int | None = None
        self.image_shape: tuple[int, int] | None = None if image_shape is None else (int(image_shape[0]), int(image_shape[1]))
        self.width: int | None = None
        self.height: int | None = None
        self.bit_depth: int | None = None
        self._dll: CDLL | None = None
        self._connected: bool = False
        self._last_image: np.ndarray | None = None

    @property
    def dll(self) -> CDLL | None:
        """The loaded Blink_C_wrapper.dll, or None when not connected or simulated."""
        return self._dll

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        """Load the DLL, Create_SDK, check the serial number and image size, then Load_LUT_file if a LUT is set.

        Safe to call again if already connected.
        """
        if self._connected:
            return
        active = MeadowlarkSLM._active
        if active is not None and active is not self:
            raise RuntimeError(f"Meadowlark SDK already in use by board {active.board}. Disconnect it first.")

        if self.simulate:
            if self.image_shape is None:
                raise ValueError("Simulated SLM needs image_shape = (height, width): there is no hardware to ask.")
            self._dll = None
            self._last_image = None
            self.serial = None
            self.height, self.width = self.image_shape
            self.bit_depth = SLM_BIT_DEPTH
            self._connected = True
            MeadowlarkSLM._active = self
            print("Meadowlark SLM: SIMULATE mode (no Blink SDK / hardware). Phase patterns are kept in memory only.")
            return

        if not self.dll_path.is_file():
            raise FileNotFoundError(f"Blink DLL not found: {self.dll_path}")
        if self.lut_path is not None and not self.lut_path.is_file():
            raise FileNotFoundError(f"LUT not found: {self.lut_path}")

        dll: CDLL = self._load_blink_dll(self.dll_path)
        self._bind_blink_api(dll)

        n_boards: c_uint = c_uint(0)
        constructed_ok: c_int = c_int(0)
        created: bool = False
        try:
            dll.Create_SDK(SLM_BIT_DEPTH, byref(n_boards), byref(constructed_ok), 1, 1, 1, 10, None)
            if constructed_ok.value != 1:
                raise RuntimeError(f"Create_SDK failed (constructed_ok={constructed_ok.value}, n_boards={n_boards.value}). {self._last_error(dll)}")
            created = True
            if not 1 <= self.board <= n_boards.value:
                raise RuntimeError(f"Board {self.board} not found: the SDK reports {n_boards.value} board(s).")

            self.serial = self._read_serial(dll, self.board)
            if self.expected_serial is not None and self.serial is not None and self.serial != self.expected_serial:
                if self.on_serial_mismatch is None or not self.on_serial_mismatch(self.expected_serial, self.serial):
                    raise SerialMismatchError(self.board, self.expected_serial, self.serial)

            width, height, depth = self._read_image_geometry(dll, self.board)
            if self.image_shape is not None and (height, width) != self.image_shape:
                raise SlmSizeMismatchError(self.board, self.image_shape, (height, width))
            if depth != SLM_BIT_DEPTH:
                print(f"Board {self.board}: Get_image_depth reports {depth} bit; frames are sent as {SLM_BIT_DEPTH}-bit images.")
            self.width, self.height, self.bit_depth = width, height, depth

            if self.lut_path is not None:
                lut_bytes: bytes = str(self.lut_path.resolve()).encode("utf-8")
                rc: int = dll.Load_LUT_file(self.board, c_char_p(lut_bytes))
                if rc != 0:
                    raise RuntimeError(f"Load_LUT_file returned {rc}. {self._last_error(dll)}")

            self._dll = dll
            self._connected = True
            MeadowlarkSLM._active = self
        except Exception:
            if created:
                try:
                    dll.Delete_SDK()
                except Exception:
                    pass
            self._dll = None
            self.serial = None
            self.width = self.height = self.bit_depth = None
            self._connected = False
            raise

    def loadLut(self, lut_path: Path) -> None:
        """Load another LUT file into the controller of the connected board (Load_LUT_file) and remember it.

        In simulate mode only the path is remembered. Raises if not connected, the file is missing, or the SDK fails.
        """
        if not self._connected:
            raise RuntimeError("Meadowlark SDK not connected. Call MeadowlarkSLM.connect() before loading a LUT.")
        if not lut_path.is_file():
            raise FileNotFoundError(f"LUT not found: {lut_path}")
        if self.simulate:
            self.lut_path = lut_path
            return
        if self._dll is None:
            raise RuntimeError("Meadowlark SDK not connected. Call MeadowlarkSLM.connect() before loading a LUT.")
        lut_bytes: bytes = str(lut_path.resolve()).encode("utf-8")
        rc: int = self._dll.Load_LUT_file(self.board, c_char_p(lut_bytes))
        if rc != 0:
            raise RuntimeError(f"Load_LUT_file returned {rc}. {self._last_error(self._dll)}")
        self.lut_path = lut_path

    def disconnect(self) -> None:
        """Delete_SDK and release session state."""
        if not self._connected:
            return
        try:
            if not self.simulate and self._dll is not None:
                self._dll.Delete_SDK()
        finally:
            self._dll = None
            self._last_image = None
            self.serial = None
            self.width = self.height = self.bit_depth = None
            self._connected = False
            if MeadowlarkSLM._active is self:
                MeadowlarkSLM._active = None

    def isSimulated(self) -> bool:
        """True when the current session is the offline stub (no real SLM writes)."""
        return self.simulate and self._connected

    def readPixelPitchUm(self) -> float | None:
        """Pixel pitch reported by the hardware (Get_pixel_pitch) in µm; None in simulate mode or if not available."""
        if not self._connected:
            raise RuntimeError("Meadowlark SDK not connected. Call MeadowlarkSLM.connect() before reading the pixel pitch.")
        if self.simulate or self._dll is None or not hasattr(self._dll, "Get_pixel_pitch"):
            return None
        pitch_um = float(self._dll.Get_pixel_pitch(self.board))
        return pitch_um if pitch_um > 0 else None

    def readTemperature(self) -> float:
        """Return the SLM temperature in degrees Celsius. Requires :meth:`connect` first."""
        if not self._connected:
            raise RuntimeError("Meadowlark SDK not connected. Call MeadowlarkSLM.connect() before reading temperature.")
        if self.simulate:
            return float(SIMULATED_TEMPERATURE_C)
        if self._dll is None:
            raise RuntimeError("Meadowlark SDK not connected. Call MeadowlarkSLM.connect() before reading temperature.")
        return float(self._dll.Read_SLM_temperature(self.board))

    def uploadHologram(self, image_u8: np.ndarray) -> None:
        """Send a single (height, width) uint8 frame to the SLM (blocking). Requires :meth:`connect` first.

        In simulate mode the frame is stored in memory (see :meth:`getLastHologram`) and not sent
        to hardware.
        """
        if not self._connected:
            raise RuntimeError("Meadowlark SDK not connected. Call MeadowlarkSLM.connect() before upload.")

        if image_u8.shape != (self.height, self.width):
            raise ValueError(f"Expected shape ({self.height}, {self.width}), got {image_u8.shape}")
        if image_u8.dtype != np.uint8:
            image_u8 = np.asarray(image_u8, dtype=np.uint8)

        if self.simulate:
            self._last_image = np.array(image_u8, copy=True)
            return

        if self._dll is None:
            raise RuntimeError("Meadowlark SDK not connected. Call MeadowlarkSLM.connect() before upload.")

        dll: CDLL = self._dll
        flat: np.ndarray = np.ascontiguousarray(image_u8.ravel(order="C"))
        buf = flat.ctypes.data_as(POINTER(c_ubyte))

        rc: int = dll.Write_image(self.board, buf, self.height * self.width, 0, 0, 0, 0, TRIGGER_TIMEOUT_MS)
        if rc != 0:
            raise RuntimeError(f"Write_image returned {rc}")

        rc = dll.ImageWriteComplete(self.board, TRIGGER_TIMEOUT_MS)
        if rc != 0:
            raise RuntimeError(f"ImageWriteComplete returned {rc}")

    def getLastHologram(self) -> np.ndarray | None:
        """Return a copy of the last simulated upload, or None if none / not in simulate mode."""
        if self._last_image is None:
            return None
        return np.array(self._last_image, copy=True)

    @staticmethod
    def _read_serial(dll: CDLL, board: int) -> int | None:
        """Serial number reported by Read_Serial_Number(board), or None if the DLL lacks it or reports failure (-1)."""
        if not hasattr(dll, "Read_Serial_Number"):
            return None
        serial: int = dll.Read_Serial_Number(board)
        return None if serial < 0 else serial

    @staticmethod
    def _read_image_geometry(dll: CDLL, board: int) -> tuple[int, int, int]:
        """(width, height, bit depth) reported by the hardware; raises RuntimeError if not available or not positive."""
        names = ("Get_image_width", "Get_image_height", "Get_image_depth")
        missing = [name for name in names if not hasattr(dll, name)]
        if missing:
            raise RuntimeError(f"The Blink DLL has no {', '.join(missing)}: cannot read the SLM image size.")
        width, height, depth = (int(getattr(dll, name)(board)) for name in names)
        if width <= 0 or height <= 0 or depth <= 0:
            raise RuntimeError(f"Board {board}: implausible image size from the SLM (width {width}, height {height}, depth {depth}).")
        return width, height, depth

    @staticmethod
    def _last_error(dll: CDLL) -> str:
        if not hasattr(dll, "Get_last_error_message"):
            return ""
        raw: bytes | None = dll.Get_last_error_message()
        return raw.decode("utf-8", errors="replace") if raw else ""

    @staticmethod
    def _load_blink_dll(dll_path: Path) -> CDLL:
        dll_dir: str = str(dll_path.parent)
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(dll_dir)
        # Also help legacy PATH resolution for transitive DLLs
        os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
        return CDLL(str(dll_path))

    @staticmethod
    def _bind_blink_api(dll: CDLL) -> None:
        # void Create_SDK(unsigned int, unsigned int*, int*, int, int, int, int, char*)
        dll.Create_SDK.argtypes = [c_uint, POINTER(c_uint), POINTER(c_int), c_int, c_int, c_int, c_int, c_char_p]
        dll.Create_SDK.restype = None

        # int Load_LUT_file(int board, char* LUT_file)
        dll.Load_LUT_file.argtypes = [c_int, c_char_p]
        dll.Load_LUT_file.restype = c_int

        # int Write_image(int board, unsigned char* image, unsigned int image_size, ...)
        dll.Write_image.argtypes = [c_int, POINTER(c_ubyte), c_uint, c_int, c_int, c_int, c_int, c_uint]
        dll.Write_image.restype = c_int

        # int ImageWriteComplete(int board, unsigned int trigger_timeout_ms)
        dll.ImageWriteComplete.argtypes = [c_int, c_uint]
        dll.ImageWriteComplete.restype = c_int

        # void Delete_SDK()
        dll.Delete_SDK.argtypes = []
        dll.Delete_SDK.restype = None

        # double Read_SLM_temperature(int board)
        dll.Read_SLM_temperature.argtypes = [c_int]
        dll.Read_SLM_temperature.restype = c_double

        # int Read_Serial_Number(int board)  (-1 on failure; signature as used by slmsuite, not yet checked against Blink_C_wrapper.h)
        if hasattr(dll, "Read_Serial_Number"):
            dll.Read_Serial_Number.argtypes = [c_int]
            dll.Read_Serial_Number.restype = c_int

        for name in ("Get_image_width", "Get_image_height", "Get_image_depth"):
            if hasattr(dll, name):
                getattr(dll, name).argtypes = [c_int]
                getattr(dll, name).restype = c_int

        if hasattr(dll, "Get_pixel_pitch"):
            dll.Get_pixel_pitch.argtypes = [c_int]
            dll.Get_pixel_pitch.restype = c_double

        if hasattr(dll, "Get_last_error_message"):
            dll.Get_last_error_message.argtypes = []
            dll.Get_last_error_message.restype = c_char_p
