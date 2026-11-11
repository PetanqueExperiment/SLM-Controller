"""Paths and numeric constants for the SLM GUI."""

import os

import numpy as np

import libslm.parameters as param

_SLM_SOFTWARE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PUPIL_VERTICAL_OFFSET: int = 0

N_Zernike: int = 15
SLM_DIFFRACTION_EFFICIENCY_HWHM: float = 90e-6
position_to_step_per_pixel: float = 2 * np.pi / param.lamb * (param.M / param.f) * param.pslm
position_to_step_per_pixel_for_z: float = (np.pi) / param.lamb * (param.M / param.f) ** 2 * (param.pslm**2)

LOCAL_DIR: str = _SLM_SOFTWARE_DIR
SLM_PATTERNS_DIR: str = os.path.join(LOCAL_DIR, "SLM_patterns")
ZERNIKE_CORRECTIONS_DIR: str = os.path.join(LOCAL_DIR, "Zernike_corrections")
TWEEZER_ROIS_DIR: str = os.path.join(LOCAL_DIR, "Tweezer_ROIs")
ZERNIKE_DEFAULT_COEFFS_JSON: str = "zernike_coefficients_default.json"

SH_CORRECTION_NPY: str = os.path.join(LOCAL_DIR, "SH_correction.npy")

DIR_SHARED_DETEKTOR: str = "Y:\\Public"
MESSAGE_FILE: str = DIR_SHARED_DETEKTOR + "\\SYNC_SLM_CAM.txt"
FILE_INTENSITY: str = DIR_SHARED_DETEKTOR + "\\file_intensity.dat"

ROI_JSON_UM_PER_CAMERA_PIXEL: float = 1.0
SYSTEM_MAGNIFICATION: float = 1.0
