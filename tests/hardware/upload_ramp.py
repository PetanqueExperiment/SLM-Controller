"""Send one linear phase ramp to the Meadowlark SLM through the driver the GUI uses (converted from test.ipynb).

Close the SLM GUI first: only one process can hold the Blink SDK. Run from the SLM-Controller folder (this script is
not collected by pytest):

    SLM_venv\\Scripts\\python.exe tests\\hardware\\upload_ramp.py --simulate
    SLM_venv\\Scripts\\python.exe tests\\hardware\\upload_ramp.py
    SLM_venv\\Scripts\\python.exe tests\\hardware\\upload_ramp.py --slope 2 --angle-deg 45 --plot

Without --simulate the ramp replaces whatever the SLM displays until something else is sent.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from libslm.slm import MeadowlarkSLM
from gui.settings import SIMULATED_BOARD, deviceImageShape

DEFAULT_LUT_FILE: Path = Path(r"C:\Program Files\Meadowlark Optics\Blink OverDrive Plus\LUT Files\slm6801_at785_33C.LUT")


def phaseRamp(shape: tuple[int, int], slope_gray_per_px: float, angle_deg: float) -> np.ndarray:
    """uint8 ramp of ``shape`` = (height, width): slope [gray levels / px] along ``angle_deg`` from the x axis,
    centred on the SLM, wrapped with mod 255 (as in test.ipynb)."""
    height, width = shape
    angle_rad = np.deg2rad(angle_deg)
    y_px, x_px = np.indices((height, width))
    x_centered_px = x_px - width / 2
    y_centered_px = y_px - height / 2
    gradient = slope_gray_per_px * (np.cos(angle_rad) * x_centered_px + np.sin(angle_rad) * y_centered_px)
    return np.mod(gradient, 255).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--simulate", action="store_true", help="use the Simulated SLM: nothing is sent to the hardware")
    parser.add_argument("--board", type=int, default=1, help="Blink PCIe board number (default: 1)")
    parser.add_argument("--lut", type=Path, default=DEFAULT_LUT_FILE, help="LUT loaded at connect (default: %(default)s)")
    parser.add_argument("--slope", type=float, default=1.0, help="ramp slope in gray levels per pixel (default: 1)")
    parser.add_argument("--angle-deg", type=float, default=90.0, help="ramp direction from the x axis in degrees (default: 90)")
    parser.add_argument("--plot", action="store_true", help="show the frame that was sent")
    args = parser.parse_args()

    if args.simulate:
        slm = MeadowlarkSLM(SIMULATED_BOARD, simulate=True, expected_shape=deviceImageShape(SIMULATED_BOARD))
    else:
        slm = MeadowlarkSLM(args.board, lut_path=args.lut)
    try:
        slm.connect()
    except Exception as e:
        print(f"FAILED to connect: {type(e).__name__}: {e}")
        return 1
    try:
        frame = phaseRamp(slm.image_shape, args.slope, args.angle_deg)
        slm.uploadHologram(frame)
        print(f"Sent a {frame.shape[1]} x {frame.shape[0]} ramp (slope {args.slope:g} gray levels/px, {args.angle_deg:g} deg){' (SIMULATE)' if slm.simulate else ''}")
        if slm.simulate:
            sent = slm.getLastHologram()
            if sent is None or not np.array_equal(sent, frame):
                print("FAILED: the simulated SLM did not keep the frame that was sent")
                return 1
    except Exception as e:
        print(f"FAILED to send the ramp: {type(e).__name__}: {e}")
        return 1
    finally:
        slm.disconnect()

    if args.plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 5))
        plt.imshow(frame, cmap="gray", interpolation="nearest")
        plt.title("Last pattern sent (ramp)")
        plt.colorbar()
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
