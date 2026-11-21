"""Check the connection to the Meadowlark SLM through the same driver code the GUI uses.

Close the SLM GUI first: only one process can hold the Blink SDK. Run from the SLM-Controller folder
(this script is not collected by pytest):

    SLM_venv\\Scripts\\python.exe tests\\hardware\\check_slm_connection.py
    SLM_venv\\Scripts\\python.exe tests\\hardware\\check_slm_connection.py --write-blank
    SLM_venv\\Scripts\\python.exe tests\\hardware\\check_slm_connection.py --simulate
    SLM_venv\\Scripts\\python.exe tests\\hardware\\check_slm_connection.py --lut "C:\\...\\slm6801_at785_33C.LUT"
"""
import argparse
import sys
import time
from pathlib import Path
from ctypes import c_double, c_int

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import libslm.slm as comm
from libslm.slm import MeadowlarkSLM
from gui.parameters import loadParameters, param
from gui.settings import SIMULATED_BOARD, deviceImageShape


def _query(dll, name, restype, board):
    fn = getattr(dll, name, None)
    if fn is None:
        return None
    fn.argtypes = [c_int]
    fn.restype = restype
    return fn(board)


def _check(label, actual, expected):
    ok = actual == expected
    print(f"  {label:<14} device={actual!s:<10} config={expected!s:<10} {'OK' if ok else 'MISMATCH'}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write-blank", action="store_true", help="upload one all-zero frame and wait for completion")
    parser.add_argument("--simulate", action="store_true", help="run against the offline stub instead of hardware")
    parser.add_argument("--board", type=int, default=1, help="Blink PCIe board number (default: 1)")
    parser.add_argument("--lut", type=Path, default=None, help="LUT file to load at connect (default: none, the controller keeps its LUT)")
    args = parser.parse_args()

    for warning in loadParameters():
        print("Parameters:", warning)

    print("DLL:", comm.BLINK_DLL, "(found)" if comm.BLINK_DLL.is_file() else "(MISSING)")
    print("LUT:", "none (use --lut)" if args.lut is None else f"{args.lut} {'(found)' if args.lut.is_file() else '(MISSING)'}")
    print("Board:", args.board)
    slm = MeadowlarkSLM(args.board, simulate=args.simulate, lut_path=args.lut, expected_shape=deviceImageShape(SIMULATED_BOARD) if args.simulate else None)

    t0 = time.perf_counter()
    try:
        slm.connect()
    except Exception as e:
        print(f"\nFAILED to connect: {type(e).__name__}: {e}")
        return 1
    print(f"\nConnected in {time.perf_counter() - t0:.2f} s{' (SIMULATE)' if slm.simulate else ''}")

    all_ok = True
    try:
        dll = slm.dll
        if dll is not None:
            width = _query(dll, "Get_image_width", c_int, slm.board)
            height = _query(dll, "Get_image_height", c_int, slm.board)
            depth = _query(dll, "Get_image_depth", c_int, slm.board)
            pitch = _query(dll, "Get_pixel_pitch", c_double, slm.board)
            serial = _query(dll, "Read_Serial_Number", c_int, slm.board)
            listed_height, listed_width = deviceImageShape(slm.board)
            print("\nDevice vs configuration (image size: device list, default if not recorded yet):")
            all_ok &= _check("width", width, listed_width)
            all_ok &= _check("height", height, listed_height)
            all_ok &= _check("driver W x H", f"{width}x{height}", f"{slm.image_shape[1]}x{slm.image_shape[0]}")
            all_ok &= _check("bit depth", depth, comm.SLM_BIT_DEPTH)
            print(f"  {'pixel pitch':<14} device={pitch!s:<10} config={param.pslm * 1e6:g} um")
            print(f"  {'serial number':<14} {serial}")

        temperature_c = slm.readTemperature()
        print("\nTemperature:", "not available (simulate)" if temperature_c is None else f"{temperature_c:.2f} C")

        if args.write_blank:
            frame = np.zeros(slm.image_shape, dtype=np.uint8)
            t0 = time.perf_counter()
            slm.uploadHologram(frame)
            print(f"Blank frame written in {(time.perf_counter() - t0) * 1e3:.1f} ms")
    except Exception as e:
        print(f"\nFAILED during test: {type(e).__name__}: {e}")
        all_ok = False
    finally:
        slm.disconnect()
        print("Disconnected")

    print("\nRESULT:", "OK" if all_ok else "PROBLEMS FOUND")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
