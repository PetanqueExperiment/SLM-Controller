
import numpy as np
from pathlib import Path
from PIL import Image


class SLMDevice:
    def __init__(self, name='No name given', simulate=False, board=1, expected_serial=None, on_serial_mismatch=None):
        """
        Parameters
        ----------
        name
            Device identifier (currently ``"Meadowlark SLM"``).
        simulate
            If True, use the offline stub (no Blink SDK / hardware).
            Default False: connect to real hardware (raises on failure).
        board
            Blink PCIe board number the SLM is connected to (1 = first board).
        expected_serial
            If given, the serial number of the SLM on ``board`` is checked before the LUT is loaded.
        on_serial_mismatch
            Called as ``on_serial_mismatch(expected, actual)`` on a mismatch: True continues connecting,
            False (or None) aborts with libslm.slm.SerialMismatchError.
        """
        self.name = name
        self.simulated = bool(simulate)
        self.board = int(board)
        self.updateArray = self._update_array_placeholder

        if name == "Meadowlark SLM":

            import libslm.parameters as parameters
            from libslm.slm import MeadowlarkSLM

            _wfc = Path(__file__).resolve().parent / "libslm" / "slm6801_at785_33C_WFC.bmp"
            print("Manufacturer wfc path: ", _wfc)
            with Image.open(_wfc) as _im:
                if _im.mode != "L":
                    _im = _im.convert("L")
                self.correction_pattern = np.asarray(_im, dtype=np.float32)
                self.correction_pattern = self.correction_pattern * 2*np.pi / 255 # convert to radians

            if self.correction_pattern.shape != (parameters.ImgResY, parameters.ImgResX):
                raise ValueError(
                    f"WFC BMP {_wfc} has shape {self.correction_pattern.shape}, "
                    f"expected ({parameters.ImgResY}, {parameters.ImgResX})"
                )

            self.parameters = parameters
            self.communication = MeadowlarkSLM(self.board, simulate=self.simulated, expected_serial=expected_serial, on_serial_mismatch=on_serial_mismatch)
            self.communication.connect()
            self.updateArray = self.communication.uploadHologram

        else:
            raise ValueError(f"SLM name {name} not supported")

    def _update_array_placeholder(self, array):
        print("No updateArray method implemented for this SLM device")
        pass

    def getSize(self):
        return self.parameters.ImgResX, self.parameters.ImgResY

    def close(self):
        print("Closing SLM device")
        if self.name == "Meadowlark SLM":
            self.communication.disconnect()
        del self
