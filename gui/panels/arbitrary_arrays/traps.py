"""Trap positions → the per-trap arrays the hologram algorithms use: k-vectors, random phases, weights, pupils."""

from __future__ import annotations

from typing import Any

import numpy as np

import libslm.parameters as param
from gui.constants import positionToStepPerPixel, positionToStepPerPixelForZ


def trapArrays(traps_pos: np.ndarray, wavelength_m: float, pupil_text: str) -> dict[str, Any]:
    """Build k-vectors, random phases, unit weights and pupils for ``traps_pos`` (N, 4): index, x, y, z [m].

    ``pupil_text`` is the pupil field: one number for circular/square, two for rectangular, three for rotated / traps
    (axes in mm on the SLM, angle in degrees). Returns N_traps, traps_k, theta_traps, weight_traps, pupil_traps.
    """
    N_traps = int(traps_pos.shape[0])
    print('Traps = ', N_traps)
    traps_k = np.copy(traps_pos[:, 1:4])
    traps_k[:, 0:2] *= positionToStepPerPixel(wavelength_m)
    traps_k[:, 2] *= positionToStepPerPixelForZ(wavelength_m)

    theta_traps = np.random.rand(N_traps) * 2 * np.pi
    weight_traps = np.ones(N_traps)

    txt = (pupil_text or '').strip()
    parts = [p.strip() for p in txt.split(',') if p.strip()]
    if not parts:
        shortaxis, longaxis, θ = 100.0, 100.0, 100.0
    elif len(parts) == 1:
        v = float(parts[0])
        shortaxis, longaxis, θ = v, v, 0.0
    elif len(parts) == 2:
        shortaxis, longaxis, θ = float(parts[0]), float(parts[1]), 0.0
    else:
        shortaxis, longaxis, θ = float(parts[0]), float(parts[1]), float(parts[2])

    longaxis /= param.pslm * 1e3
    shortaxis /= param.pslm * 1e3

    θ *= np.pi / 180

    pupil_traps = {
        'pupil_longaxis': np.array([longaxis] * N_traps),
        'pupil_shortaxis': np.array([shortaxis] * N_traps),
        'pupil_angle': np.array([θ] * N_traps),
    }
    return {'N_traps': N_traps, 'traps_k': traps_k, 'theta_traps': theta_traps, 'weight_traps': weight_traps, 'pupil_traps': pupil_traps}
