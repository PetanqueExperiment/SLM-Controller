"""Hologram algorithms for trap arrays: plain superposition (computeHologram), weighted GS (WGS) and GS.

All take the trap arrays explicitly and return the SLM phase [rad] instead of reading / writing GUI state.
``progress`` is called after each iteration (the GUI passes a function that keeps the window responsive).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from gui.parameters import param
from libslm.gpu import slm_to_trap_amp, trap_phase_to_slm
from gui.constants import SLM_DIFFRACTION_EFFICIENCY_HWHM


def _noProgress() -> None:
    pass


def computeHologram(traps_k: np.ndarray, theta_traps: np.ndarray, weight_traps: np.ndarray, pupil_traps: dict[str, np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """SLM phase of the weighted superposition of all trap gratings (GPU)."""
    return trap_phase_to_slm(traps_k, theta_traps, weight_traps, pupil_traps, shape=shape)


def wgs(traps_pos: np.ndarray, traps_k: np.ndarray, theta_traps: np.ndarray, weight_traps: np.ndarray, pupil_traps: dict[str, np.ndarray], shape: tuple[int, int], iterations: int, dx_um: float, dy_um: float, correct_diffraction_efficiency: bool = False, progress: Callable[[], None] = _noProgress) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Weighted Gerchberg−Saxton: optimise the phase and weight of each trap grating.

    ``weight_traps`` is updated in place. Returns (phi, theta_traps, weight_traps), or None when the SLM
    diffraction-efficiency model is not valid for these traps.
    """
    N_traps = int(traps_pos.shape[0])
    if correct_diffraction_efficiency:
        # Calculate the radial displacement of each traps
        traps_displacements = ((traps_pos[:, 1] + dx_um * 1e-6) ** 2 + (traps_pos[:, 2] + dy_um * 1e-6) ** 2) ** 0.5

        # The SLM diffraction efficiency depends on the displacement of the traps
        SLM_diffraction_efficiency = 1.0 - 0.5 * traps_displacements / SLM_DIFFRACTION_EFFICIENCY_HWHM

        # The linear model reaches zero at 2*HWHM; beyond that the weights would become NaN
        if np.any(SLM_diffraction_efficiency <= 0):
            print(
                'WGS aborted: traps reach %.0f um from the center, but the SLM efficiency '
                'model is only valid below %.0f um. Use plain WGS for this trap file.'
                % (np.max(traps_displacements) * 1e6, 2 * SLM_DIFFRACTION_EFFICIENCY_HWHM * 1e6)
            )
            return None

        # Correct for the limited diffraction efficiency
        target_intensities = 1.0 / SLM_diffraction_efficiency
        target_amplitudes = target_intensities**0.5
        target_amplitudes_norm = target_amplitudes / np.mean(target_amplitudes)

    else:
        target_intensities = np.ones(N_traps)
        target_amplitudes_norm = np.ones(N_traps)

    # Calculate phase pattern with homemade gpu function
    phi = computeHologram(traps_k, theta_traps, weight_traps, pupil_traps, shape)

    # Measure the trap to trap deviation of intensity
    traps_amplitude = 1.0 / param.ImgResY / param.ImgResX * slm_to_trap_amp(phi, traps_k)
    trap_intensities = np.real(traps_amplitude * np.conj(traps_amplitude))
    deviation_from_target = trap_intensities / target_intensities
    rel_dev = deviation_from_target**0.5 / np.mean(deviation_from_target**0.5)
    efficiency = np.sum(trap_intensities)
    uniformity = np.std(deviation_from_target) / np.mean(deviation_from_target)

    print('Initial calculation:')
    print('Efficiency std = %.1f' % (efficiency * 100))
    print('Intensity std = %.1f' % (uniformity * 100))
    progress()

    # Start the iterations
    for k in range(iterations):
        print('    WGS %d/%d' % (k + 1, iterations))
        # Calculate new phases and weights
        theta_traps = np.angle(traps_amplitude)
        weight_traps *= 1.0 / rel_dev

        # Calculate new hologram
        phi = computeHologram(traps_k, theta_traps, weight_traps, pupil_traps, shape)

        # Calculate new trap intensities
        traps_amplitude = 1.0 / param.ImgResY / param.ImgResX * slm_to_trap_amp(phi, traps_k)
        trap_intensities = np.real(traps_amplitude * np.conj(traps_amplitude))
        deviation_from_target = trap_intensities / target_intensities
        rel_dev = deviation_from_target**0.5 / np.mean(deviation_from_target**0.5)
        efficiency = np.sum(trap_intensities)
        uniformity = np.std(deviation_from_target) / np.mean(deviation_from_target)
        print('Efficiency std = %.1f' % (efficiency * 100))
        print('Intensity std = %.1f' % (uniformity * 100))
        progress()

    return phi, theta_traps, weight_traps


def gs(traps_k: np.ndarray, pupil_traps: dict[str, np.ndarray], shape: tuple[int, int], iterations: int, progress: Callable[[], None] = _noProgress) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gerchberg−Saxton: optimise the phase of each trap grating (unit weights). Returns (phi, theta_traps, weight_traps)."""
    N_traps = int(traps_k.shape[0])

    # Get initial random phases
    theta_traps = np.random.rand(N_traps) * 2 * np.pi
    weight_traps = np.ones(N_traps)

    # Calculate phase pattern with homemade gpu function
    phi = computeHologram(traps_k, theta_traps, weight_traps, pupil_traps, shape)

    # (a) Calculate the intensity at trap position
    traps_amplitude = 1.0 / param.ImgResY / param.ImgResX * slm_to_trap_amp(phi, traps_k)

    trap_intensities = np.real(traps_amplitude * np.conj(traps_amplitude))
    efficiency = np.sum(trap_intensities)
    uniformity = np.std(trap_intensities) / np.mean(trap_intensities)

    print('1st calculation:')
    print('Efficiency std = %.1f' % (efficiency * 100))
    print('Intensity std = %.1f' % (uniformity * 100))
    progress()

    # Start the iterations5
    for k in range(iterations):
        print('    GS ' + str(k + 1) + '/' + str(iterations))

        # (a) Calculate the phase of each trap
        if k == 0:  # (b0) if first time: start with random phases
            theta_traps = np.random.rand(N_traps) * 2 * np.pi
            weight_traps = np.ones(N_traps)
        else:  # (b0)else  apply the GS technique
            theta_traps = np.angle(traps_amplitude)

        # (c) Calculate the new slm phase
        phi = computeHologram(traps_k, theta_traps, weight_traps, pupil_traps, shape)

        # (a) Calculate the intensity at trap position
        traps_amplitude = 1.0 / param.ImgResY / param.ImgResX * slm_to_trap_amp(phi, traps_k)

        trap_intensities = np.real(traps_amplitude * np.conj(traps_amplitude))
        efficiency = np.sum(trap_intensities)
        uniformity = np.std(trap_intensities) / np.mean(trap_intensities)
        print('Efficiency std = %.1f' % (efficiency * 100))
        print('Intensity std = %.1f' % (uniformity * 100))
        progress()

    return phi, theta_traps, weight_traps
