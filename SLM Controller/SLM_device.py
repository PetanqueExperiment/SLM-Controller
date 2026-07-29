
import numpy as np
from pathlib import Path
from PIL import Image

class SLMDevice:
    def __init__(self, name='No name given'):
        self.name = name
        self.updateArray = self._update_array_placeholder

        if name == "Meadowlark SLM":

            import Meadowlark_SLM.parameters as parameters
            import Meadowlark_SLM.communication as communication

            _wfc = Path(__file__).resolve().parent / "Meadowlark_SLM" / "slm6801_at785_33C_WFC.bmp"
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
            self.communication = communication
            communication.connect()
            self.updateArray = communication.upload_image_to_slm

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
            import Meadowlark_SLM.communication as communication

            communication.disconnect()
        del self