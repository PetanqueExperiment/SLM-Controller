"""Paths and numeric constants for the SLM GUI."""
import os

import numpy as np

import Meadowlark_SLM.parameters as param

_SLM_SOFTWARE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PUPIL_VERTICAL_OFFSET = 0

N_Zernike = 15
SLM_DIFFRACTION_EFFICIENCY_HWHM = 90e-6
position_to_step_per_pixel = 2 * np.pi / param.lamb * (param.M / param.f) * param.pslm
position_to_step_per_pixel_for_z = (
    (np.pi) / param.lamb * (param.M / param.f) ** 2 * (param.pslm ** 2)
)

LOCAL_DIR = _SLM_SOFTWARE_DIR
SLM_PATTERNS_DIR = os.path.join(LOCAL_DIR, "SLM_patterns")
ZERNIKE_CORRECTIONS_DIR = os.path.join(LOCAL_DIR, "Zernike_corrections")
TWEEZER_ROIS_DIR = os.path.join(LOCAL_DIR, "Tweezer_ROIs")
ZERNIKE_DEFAULT_COEFFS_JSON = "zernike_coefficients_default.json"

SH_CORRECTION_NPY = os.path.join(LOCAL_DIR, "SH_correction.npy")

DIR_SHARED_DETEKTOR = "Y:\\Public"
MESSAGE_FILE = DIR_SHARED_DETEKTOR + "\\SYNC_SLM_CAM.txt"
FILE_INTENSITY = DIR_SHARED_DETEKTOR + "\\file_intensity.dat"

DEFAULT_SLM = "Meadowlark SLM"

ROI_JSON_UM_PER_CAMERA_PIXEL = 1.0
SYSTEM_MAGNIFICATION = 1.5
