"""Check the connection to the Meadowlark SLM through the same driver code the GUI uses.

Close the SLM GUI first: only one process can hold the Blink SDK.

    SLM_venv\\Scripts\\python.exe test_slm_connection.py
    SLM_venv\\Scripts\\python.exe test_slm_connection.py --write-blank
    SLM_venv\\Scripts\\python.exe test_slm_connection.py --simulate
    SLM_venv\\Scripts\\python.exe test_slm_connection.py --lut "C:\\...\\slm6801_at785_33C.LUT"
"""
import argparse
import sys
import time
from pathlib import Path
from ctypes import c_double, c_int

import numpy as np

import libslm.slm as comm
from libslm.slm import MeadowlarkSLM
from gui.parameters import loadParameters, param


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
    parser.add_argument("--lut", type=Path, default=None, help="LUT file to load at connect (default: none, the controller keeps its LUT)")
    args = parser.parse_args()

    for warning in loadParameters():
        print("Parameters:", warning)

    print("DLL:", comm.BLINK_DLL, "(found)" if comm.BLINK_DLL.is_file() else "(MISSING)")
    print("LUT:", "none (use --lut)" if args.lut is None else f"{args.lut} {'(found)' if args.lut.is_file() else '(MISSING)'}")
    print("Board:", comm.BOARD)
    slm = MeadowlarkSLM(comm.BOARD, simulate=args.simulate, lut_path=args.lut, image_shape=(param.ImgResY, param.ImgResX) if args.simulate else None)

    t0 = time.perf_counter()
    try:
        slm.connect()
    except Exception as e:
        print(f"\nFAILED to connect: {type(e).__name__}: {e}")
        return 1
    print(f"\nConnected in {time.perf_counter() - t0:.2f} s{' (SIMULATE)' if slm.isSimulated() else ''}")

    all_ok = True
    try:
        dll = slm.dll
        if dll is not None:
            width = _query(dll, "Get_image_width", c_int, slm.board)
            height = _query(dll, "Get_image_height", c_int, slm.board)
            depth = _query(dll, "Get_image_depth", c_int, slm.board)
            pitch = _query(dll, "Get_pixel_pitch", c_double, slm.board)
            serial = _query(dll, "Read_Serial_Number", c_int, slm.board)
            print("\nDevice vs configuration:")
            all_ok &= _check("width", width, param.ImgResX)
            all_ok &= _check("height", height, param.ImgResY)
            all_ok &= _check("driver W x H", f"{width}x{height}", f"{slm.width}x{slm.height}")
            all_ok &= _check("bit depth", depth, comm.SLM_BIT_DEPTH)
            print(f"  {'pixel pitch':<14} device={pitch!s:<10} config={param.pslm * 1e6:g} um")
            print(f"  {'serial number':<14} {serial}")

        print(f"\nTemperature: {slm.readTemperature():.2f} C")

        if args.write_blank:
            frame = np.zeros((slm.height, slm.width), dtype=np.uint8)
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
