"""Meadowlark XY Phase 1024×1024 — minimal SDK upload via Blink_C_wrapper.dll (ctypes).

Requires Meadowlark Blink / PCIe install or USB SDK: Blink_C_wrapper.dll and your *.lut file.
See PCIe User Manual §4 (Create_SDK, Load_LUT_file, Write_image, ImageWriteComplete, Delete_SDK).

Usage for live GUI: call :func:`connect` once when opening the device, then :func:`upload_image_to_slm`
for each frame, and :func:`disconnect` when shutting down. Per-frame work is only Write_image +
ImageWriteComplete; Create_SDK / Load_LUT / Delete_SDK run once per session.

For offline GUI testing without hardware, call :func:`connect` with ``simulate=True``
(e.g. ``GUI_SLM.py --simulate`` / ``Start_SLM_GUI_simulate.bat``). Uploads stay in memory
and temperature returns a stub value. Simulation is never selected automatically.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from ctypes import (
    CDLL,
    POINTER,
    byref,
    c_char_p,
    c_double,
    c_int,
    c_ubyte,
    c_uint,
)

# --- Configure ---
# Blink_C_wrapper.dll usually lives under the Meadowlark Blink PCIe install (Program Files), not always on the USB.
# Point BLINK_DLL at the real path on this PC.
BLINK_DLL = Path(
    r"C:\Program Files\Meadowlark Optics\Blink OverDrive Plus\SDK\Blink_C_wrapper.dll"
)
# LUT: your USB stick includes e.g. 1024x1024_linearVoltage.lut — prefer the calibration shipped for your SLM.
LUT_FILE = Path(
    r"C:\Program Files\Meadowlark Optics\Blink OverDrive Plus\LUT Files\slm6801_at785_33C.LUT"
)

BOARD = 1
W = H = 1024
IMAGE_SIZE = W * H
SLM_BIT_DEPTH = 8  # 8-bit frames; manual lists 8 for 512, 12 for 1920 — use 8 for 1024×1024 8-bit API
TRIGGER_TIMEOUT_MS = 5000
# Nominal Meadowlark operating temperature used by the offline stub (°C).
SIMULATED_TEMPERATURE_C = 33.0

_blink_dll: CDLL | None = None
_sdk_connected: bool = False
_simulate: bool = False
_last_image: np.ndarray | None = None


def is_simulated() -> bool:
    """True when the current session is the offline stub (no real SLM writes)."""
    return _simulate and _sdk_connected


def _load_blink_dll(dll_path: Path) -> CDLL:
    dll_dir = str(dll_path.parent)
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(dll_dir)
    # Also help legacy PATH resolution for transitive DLLs
    os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
    return CDLL(str(dll_path))


def _bind_blink_api(dll: CDLL) -> None:
    # void Create_SDK(unsigned int, unsigned int*, int*, int, int, int, int, char*)
    dll.Create_SDK.argtypes = [
        c_uint,
        POINTER(c_uint),
        POINTER(c_int),
        c_int,
        c_int,
        c_int,
        c_int,
        c_char_p,
    ]
    dll.Create_SDK.restype = None

    # int Load_LUT_file(int board, char* LUT_file)
    dll.Load_LUT_file.argtypes = [c_int, c_char_p]
    dll.Load_LUT_file.restype = c_int

    # int Write_image(int board, unsigned char* image, unsigned int image_size, ...)
    dll.Write_image.argtypes = [
        c_int,
        POINTER(c_ubyte),
        c_uint,
        c_int,
        c_int,
        c_int,
        c_int,
        c_uint,
    ]
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

    if hasattr(dll, "Get_last_error_message"):
        dll.Get_last_error_message.argtypes = []
        dll.Get_last_error_message.restype = c_char_p


def connect(
    *,
    dll_path: Path = BLINK_DLL,
    lut_path: Path = LUT_FILE,
    simulate: bool = False,
) -> None:
    """Load the DLL, Create_SDK, and Load_LUT_file. Safe to call again if already connected.

    Parameters
    ----------
    simulate
        If True, skip hardware and keep frames in memory. Default False (real hardware).
    """
    global _blink_dll, _sdk_connected, _simulate, _last_image
    if _sdk_connected:
        return

    if simulate:
        _blink_dll = None
        _last_image = None
        _simulate = True
        _sdk_connected = True
        print(
            "Meadowlark SLM: SIMULATE mode (no Blink SDK / hardware). "
            "Phase patterns are kept in memory only."
        )
        return

    if not dll_path.is_file():
        raise FileNotFoundError(f"Blink DLL not found: {dll_path}")
    if not lut_path.is_file():
        raise FileNotFoundError(f"LUT not found: {lut_path}")

    dll = _load_blink_dll(dll_path)
    _bind_blink_api(dll)

    n_boards = c_uint(0)
    constructed_ok = c_int(0)
    created = False
    try:
        dll.Create_SDK(
            SLM_BIT_DEPTH,
            byref(n_boards),
            byref(constructed_ok),
            1,
            1,
            1,
            10,
            None,
        )
        if constructed_ok.value != 1:
            msg = ""
            if hasattr(dll, "Get_last_error_message"):
                raw = dll.Get_last_error_message()
                if raw:
                    msg = raw.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Create_SDK failed (constructed_ok={constructed_ok.value}, "
                f"n_boards={n_boards.value}). {msg}"
            )
        created = True

        lut_bytes = str(lut_path.resolve()).encode("utf-8")
        rc = dll.Load_LUT_file(BOARD, c_char_p(lut_bytes))
        if rc != 0:
            msg = ""
            if hasattr(dll, "Get_last_error_message"):
                raw = dll.Get_last_error_message()
                if raw:
                    msg = raw.decode("utf-8", errors="replace")
            raise RuntimeError(f"Load_LUT_file returned {rc}. {msg}")

        _blink_dll = dll
        _simulate = False
        _sdk_connected = True
    except Exception:
        if created:
            try:
                dll.Delete_SDK()
            except Exception:
                pass
        _blink_dll = None
        _simulate = False
        _sdk_connected = False
        raise


def disconnect() -> None:
    """Delete_SDK and release session state."""
    global _blink_dll, _sdk_connected, _simulate, _last_image
    if not _sdk_connected:
        return
    try:
        if _simulate:
            return
        if _blink_dll is not None:
            _blink_dll.Delete_SDK()
    finally:
        _blink_dll = None
        _simulate = False
        _last_image = None
        _sdk_connected = False


def read_temperature(board: int = BOARD) -> float:
    """Return the SLM temperature in degrees Celsius. Requires :func:`connect` first."""
    if not _sdk_connected:
        raise RuntimeError(
            "Meadowlark SDK not connected. Call Meadowlark_SLM.communication.connect() before reading temperature."
        )
    if _simulate:
        return float(SIMULATED_TEMPERATURE_C)
    if _blink_dll is None:
        raise RuntimeError(
            "Meadowlark SDK not connected. Call Meadowlark_SLM.communication.connect() before reading temperature."
        )
    return float(_blink_dll.Read_SLM_temperature(board))


def upload_image_to_slm(
    image_u8: np.ndarray,
) -> None:
    """Send a single 1024×1024 uint8 frame to the SLM (blocking). Requires :func:`connect` first.

    In simulate mode the frame is stored in memory (see :func:`get_last_image`) and not sent
    to hardware.
    """
    global _blink_dll, _sdk_connected, _last_image
    if not _sdk_connected:
        raise RuntimeError(
            "Meadowlark SDK not connected. Call Meadowlark_SLM.communication.connect() before upload."
        )

    if image_u8.shape != (H, W):
        raise ValueError(f"Expected shape ({H}, {W}), got {image_u8.shape}")
    if image_u8.dtype != np.uint8:
        image_u8 = np.asarray(image_u8, dtype=np.uint8)

    if _simulate:
        _last_image = np.array(image_u8, copy=True)
        return

    if _blink_dll is None:
        raise RuntimeError(
            "Meadowlark SDK not connected. Call Meadowlark_SLM.communication.connect() before upload."
        )

    dll = _blink_dll
    flat = np.ascontiguousarray(image_u8.ravel(order="C"))
    buf = flat.ctypes.data_as(POINTER(c_ubyte))

    rc = dll.Write_image(
        BOARD,
        buf,
        IMAGE_SIZE,
        0,
        0,
        0,
        0,
        TRIGGER_TIMEOUT_MS,
    )
    if rc != 0:
        raise RuntimeError(f"Write_image returned {rc}")

    rc = dll.ImageWriteComplete(BOARD, TRIGGER_TIMEOUT_MS)
    if rc != 0:
        raise RuntimeError(f"ImageWriteComplete returned {rc}")


def get_last_image() -> np.ndarray | None:
    """Return a copy of the last simulated upload, or None if none / not in simulate mode."""
    if _last_image is None:
        return None
    return np.array(_last_image, copy=True)
