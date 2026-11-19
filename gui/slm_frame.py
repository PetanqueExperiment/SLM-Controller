"""Composition of the uint8 frame sent to the SLM from a hologram phase pattern and the correction terms (no Qt).

Used by SLMGUI.updatePhase (one frame) and by the dynamic-mode sequence (one frame per hologram).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrameTerms:
    """Phase terms added to every hologram, as set in the GUI. Arrays are (Ny, Nx) [rad]; None = term not applied."""

    grating_rad: np.ndarray
    fresnel_rad: np.ndarray
    wfc_rad: np.ndarray | None
    zernike_rad: np.ndarray | None
    pupil_mask: np.ndarray | None
    pupil_outside_rad: np.ndarray | None
    phase_levels: float

    def copy(self) -> FrameTerms:
        """Independent copy (the GUI may change its arrays in place later)."""
        def dup(a):
            return None if a is None else np.array(a, copy=True)
        return FrameTerms(dup(self.grating_rad), dup(self.fresnel_rad), dup(self.wfc_rad), dup(self.zernike_rad), dup(self.pupil_mask), dup(self.pupil_outside_rad), float(self.phase_levels))


def applyPupilMask(phase_rad: np.ndarray, pupil_mask: np.ndarray, pupil_outside_rad: np.ndarray) -> np.ndarray:
    """Keep ``phase_rad`` inside the pupil, replace it by the outside grating elsewhere (in place; returns the array)."""
    phase_rad *= pupil_mask
    phase_rad += pupil_outside_rad
    return phase_rad


def composeSlmFrame(pattern_rad: np.ndarray, terms: FrameTerms) -> np.ndarray:
    """uint8 SLM frame: (grating + Fresnel + pattern [+ WFC] [+ Zernike], pupil) mod 2π, as gray levels (phase_levels per 2π)."""
    total_rad = terms.grating_rad + terms.fresnel_rad + pattern_rad
    if terms.wfc_rad is not None:
        total_rad = total_rad + terms.wfc_rad
    if terms.zernike_rad is not None:
        total_rad = total_rad + terms.zernike_rad
    if terms.pupil_mask is not None:
        total_rad = applyPupilMask(total_rad, terms.pupil_mask, terms.pupil_outside_rad)
    total_rad = np.mod(total_rad, 2 * np.pi)
    return np.round(terms.phase_levels * total_rad / (2 * np.pi)).astype("uint8")
