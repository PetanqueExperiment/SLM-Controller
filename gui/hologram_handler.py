"""Save / Load Hologram files. Current format: a pickled ``traps_dict`` (.pkl); older .npy / .bmp phase maps still load."""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np
from PIL import Image


def writeTrapsDict(path: str, traps_dict: dict[str, Any]) -> None:
    """Pickle ``traps_dict`` (traps_pos, traps_k, theta_traps, weight_traps, pupil_traps, phasePattern) to ``path``."""
    a_file = open(path, "wb")
    pickle.dump(traps_dict, a_file)
    a_file.close()


def readTrapsDict(path: str) -> dict[str, Any]:
    a_file = open(path, "rb")
    try:
        return pickle.load(a_file)
    finally:
        a_file.close()


def readLegacyPattern(path: str) -> np.ndarray | None:
    """Phase map [rad] from an obsolete .npy (cropped to 792 columns) or 8-bit .bmp file; None for other extensions."""
    if path[-3:] == 'npy':
        phasePattern = np.load(path)
        return phasePattern[:, :792]  # in case that the pattern size is 600x800
    if path[-3:] == 'bmp':
        phasePattern = np.asarray(Image.open(path), dtype=np.float64)
        phasePattern *= 2 * np.pi / 256
        return phasePattern
    return None
