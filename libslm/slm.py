"""Meadowlark XY Phase 1024×1024 — minimal SDK upload via Blink_C_wrapper.dll (ctypes).

Requires Meadowlark Blink / PCIe install or USB SDK: Blink_C_wrapper.dll and your *.lut file.
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

# --- Configure ---
# Blink_C_wrapper.dll usually lives under the Meadowlark Blink PCIe install (Program Files), not always on the USB.
# Point BLINK_DLL at the real path on this PC.
BLINK_DLL: Path = Path(r"C:\Program Files\Meadowlark Optics\Blink OverDrive Plus\SDK\Blink_C_wrapper.dll")
# LUT: your USB stick includes e.g. 1024x1024_linearVoltage.lut — prefer the calibration shipped for your SLM.
LUT_FILE: Path = Path(r"C:\Program Files\Meadowlark Optics\Blink OverDrive Plus\LUT Files\slm6801_at785_33C.LUT")
# Manufacturer wavefront correction (WFC) of the same SLM head, 8-bit BMP; added to every hologram by the GUI.
WFC_FILE: Path = Path(__file__).resolve().parent / "slm6801_at785_33C_WFC.bmp"

BOARD: int = 1
W: int = 1024
H: int = 1024
IMAGE_SIZE: int = W * H
SLM_BIT_DEPTH: int = 8  # 8-bit frames; manual lists 8 for 512, 12 for 1920 — use 8 for 1024×1024 8-bit API
TRIGGER_TIMEOUT_MS: int = 5000
# Nominal Meadowlark operating temperature used by the offline stub (°C).
SIMULATED_TEMPERATURE_C: float = 33.0


class SerialMismatchError(RuntimeError):
    """The SLM found on the board is not the one expected (different serial number)."""

    def __init__(self, board: int, expected: int, actual: int) -> None:
        super().__init__(f"Board {board}: expected SLM serial number {expected}, found {actual}.")
        self.board: int = board
        self.expected: int = expected
        self.actual: int = actual


class MeadowlarkSLM:
    """One Meadowlark SLM on a Blink PCIe board.

    The Blink SDK is a single instance per process (Create_SDK / Delete_SDK are global in the DLL),
    so only one MeadowlarkSLM may be connected at a time.

    If ``expected_serial`` is given, :meth:`connect` reads the serial number right after Create_SDK, before the LUT
    or any image reaches the SLM. On a mismatch it calls ``on_serial_mismatch(expected, actual)``: True continues
    connecting, False (or no callback) closes the SDK again and raises SerialMismatchError.
    """

    _active: ClassVar[MeadowlarkSLM | None] = None

    def __init__(self, board: int = BOARD, *, dll_path: Path = BLINK_DLL, lut_path: Path = LUT_FILE, wfc_path: Path = WFC_FILE, simulate: bool = False, expected_serial: int | None = None, on_serial_mismatch: Callable[[int, int], bool] | None = None) -> None:
        self.board: int = board
        self.dll_path: Path = dll_path
        self.lut_path: Path = lut_path
        self.wfc_path: Path = wfc_path
        self.correction_pattern: np.ndarray | None = None
        self.simulate: bool = simulate
        self.expected_serial: int | None = expected_serial
        self.on_serial_mismatch: Callable[[int, int], bool] | None = on_serial_mismatch
        self.serial: int | None = None
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
        """Load the WFC correction, then the DLL, Create_SDK, and Load_LUT_file. Safe to call again if already connected."""
        if self._connected:
            return
        active = MeadowlarkSLM._active
        if active is not None and active is not self:
            raise RuntimeError(f"Meadowlark SDK already in use by board {active.board}. Disconnect it first.")

        self.correction_pattern = self._load_correction(self.wfc_path)

        if self.simulate:
            self._dll = None
            self._last_image = None
            self.serial = None
            self._connected = True
            MeadowlarkSLM._active = self
            print("Meadowlark SLM: SIMULATE mode (no Blink SDK / hardware). Phase patterns are kept in memory only.")
            return

        if not self.dll_path.is_file():
            raise FileNotFoundError(f"Blink DLL not found: {self.dll_path}")
        if not self.lut_path.is_file():
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
            self._connected = False
            raise

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
            self._connected = False
            if MeadowlarkSLM._active is self:
                MeadowlarkSLM._active = None

    def isSimulated(self) -> bool:
        """True when the current session is the offline stub (no real SLM writes)."""
        return self.simulate and self._connected

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
        """Send a single 1024×1024 uint8 frame to the SLM (blocking). Requires :meth:`connect` first.

        In simulate mode the frame is stored in memory (see :meth:`getLastHologram`) and not sent
        to hardware.
        """
        if not self._connected:
            raise RuntimeError("Meadowlark SDK not connected. Call MeadowlarkSLM.connect() before upload.")

        if image_u8.shape != (H, W):
            raise ValueError(f"Expected shape ({H}, {W}), got {image_u8.shape}")
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

        rc: int = dll.Write_image(self.board, buf, IMAGE_SIZE, 0, 0, 0, 0, TRIGGER_TIMEOUT_MS)
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
    def _load_correction(path: Path) -> np.ndarray:
        """Manufacturer WFC BMP as a (H, W) float32 phase map in radians (gray level × 2π / 255)."""
        print("Manufacturer wfc path: ", path)
        if not path.is_file():
            raise FileNotFoundError(f"WFC correction not found: {path}")
        with Image.open(path) as im:
            if im.mode != "L":
                im = im.convert("L")
            pattern: np.ndarray = np.asarray(im, dtype=np.float32) * 2 * np.pi / 255
        if pattern.shape != (H, W):
            raise ValueError(f"WFC BMP {path} has shape {pattern.shape}, expected ({H}, {W})")
        return pattern

    @staticmethod
    def _read_serial(dll: CDLL, board: int) -> int | None:
        """Serial number reported by Read_Serial_Number(board), or None if the DLL lacks it or reports failure (-1)."""
        if not hasattr(dll, "Read_Serial_Number"):
            return None
        serial: int = dll.Read_Serial_Number(board)
        return None if serial < 0 else serial

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

        if hasattr(dll, "Get_last_error_message"):
            dll.Get_last_error_message.argtypes = []
            dll.Get_last_error_message.restype = c_char_p
