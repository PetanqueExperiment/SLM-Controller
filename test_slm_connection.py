"""Check the connection to the Meadowlark SLM through the same driver code the GUI uses.

Close the SLM GUI first: only one process can hold the Blink SDK.

    SLM_venv\\Scripts\\python.exe test_slm_connection.py
    SLM_venv\\Scripts\\python.exe test_slm_connection.py --write-blank
    SLM_venv\\Scripts\\python.exe test_slm_connection.py --simulate
"""
import argparse
import sys
import time
from ctypes import c_double, c_int

import numpy as np

import libslm.slm as comm
import libslm.parameters as param


def _query(dll, name, restype):
    fn = getattr(dll, name, None)
    if fn is None:
        return None
    fn.argtypes = [c_int]
    fn.restype = restype
    return fn(comm.BOARD)


def _check(label, actual, expected):
    ok = actual == expected
    print(f"  {label:<14} device={actual!s:<10} config={expected!s:<10} {'OK' if ok else 'MISMATCH'}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write-blank", action="store_true", help="upload one all-zero frame and wait for completion")
    parser.add_argument("--simulate", action="store_true", help="run against the offline stub instead of hardware")
    args = parser.parse_args()

    print("DLL:", comm.BLINK_DLL, "(found)" if comm.BLINK_DLL.is_file() else "(MISSING)")
    print("LUT:", comm.LUT_FILE, "(found)" if comm.LUT_FILE.is_file() else "(MISSING)")
    print("Board:", comm.BOARD)

    t0 = time.perf_counter()
    try:
        comm.connect(simulate=args.simulate)
    except Exception as e:
        print(f"\nFAILED to connect: {type(e).__name__}: {e}")
        return 1
    print(f"\nConnected in {time.perf_counter() - t0:.2f} s{' (SIMULATE)' if comm.isSimulated() else ''}")

    all_ok = True
    try:
        dll = comm._blink_dll
        if dll is not None:
            width = _query(dll, "Get_image_width", c_int)
            height = _query(dll, "Get_image_height", c_int)
            depth = _query(dll, "Get_image_depth", c_int)
            pitch = _query(dll, "Get_pixel_pitch", c_double)
            serial = _query(dll, "Read_Serial_Number", c_int)
            print("\nDevice vs configuration:")
            all_ok &= _check("width", width, param.ImgResX)
            all_ok &= _check("height", height, param.ImgResY)
            all_ok &= _check("comm W x H", f"{width}x{height}", f"{comm.W}x{comm.H}")
            all_ok &= _check("bit depth", depth, comm.SLM_BIT_DEPTH)
            print(f"  {'pixel pitch':<14} device={pitch!s:<10} config={param.pslm * 1e6:g} um")
            print(f"  {'serial number':<14} {serial}")

        print(f"\nTemperature: {comm.readTemperature():.2f} C")

        if args.write_blank:
            frame = np.zeros((comm.H, comm.W), dtype=np.uint8)
            t0 = time.perf_counter()
            comm.uploadHologram(frame)
            print(f"Blank frame written in {(time.perf_counter() - t0) * 1e3:.1f} ms")
    except Exception as e:
        print(f"\nFAILED during test: {type(e).__name__}: {e}")
        all_ok = False
    finally:
        comm.disconnect()
        print("Disconnected")

    print("\nRESULT:", "OK" if all_ok else "PROBLEMS FOUND")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
