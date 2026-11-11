"""Trap equalization with the atom signal (dormant: its buttons are commented out in the panel).

A mixin of ArbitraryArraysPanel: it uses the panel's ``gui``, ``computeHologram``, ``_ensure_traps_loaded`` and
``savePattern``. The code is unchanged from the single-file panel.
"""

from __future__ import annotations

import random as rd
import time

import numpy as np

from gui.constants import DIR_SHARED_DETEKTOR, LOCAL_DIR


class EqualizationMixin:
    # =============================================================================
    # Trap equalization using atom signal
    # ============================================================================
    def equalize_with_atoms(self) -> None:
        print('equalize with atoms')
        signal = np.load(DIR_SHARED_DETEKTOR + '/map.npy')
        # signal += 1-np.mean(signal)
        print('current stdev: %.1f' % (np.std(signal) * 100))

        self.gui.equalize_gain = 1.0  # float(self.equalize_gain_le.text())

        # Update hologram
        self.gui.phasePattern = self.GS_intensity(signal, self.gui.equalize_gain)
        if self.gui.phasePattern is None:
            return

        self.gui.updatePhase()

        # save hologram information
        timestr = time.strftime("%Y%m%d_%H%M%S")
        self.savePattern(name=LOCAL_DIR + '\\holograms\\equalized_with_atoms_' + timestr + '.pkl')

    def devide_slm_area(self, change: bool = True, signal: np.ndarray | int | None = None) -> None:
        """
        With Martin we change only part of the hologram to gain more precission in equalizing the trap intensity.
        """

        if signal == None:
            signal = np.load(DIR_SHARED_DETEKTOR + '/map.npy')

        else:
            signal = self.gui.map

        # signal += 1-np.mean(signal)
        if change:
            print('current stdev: %.1f' % (np.std(signal) * 100))
        else:
            print('no change of initial hologram')

        self.gui.equalize_gain = 1.0  # float(self.equalize_gain_le.text())

        # Update hologram
        phasePattern_orig = self.gui.phasePattern
        phasePattern_cal = self.GS_intensity(signal, self.gui.equalize_gain)
        if phasePattern_cal is None:
            return

        if np.std(signal) < 0.6:
            k = rd.randint(0, 2)
            l = rd.randint(0, 2)

            y_len = 792  # param.ImgResX
            x_len = 600  # param.ImgResY

            # type(param.ImgResX)
            d_phi = phasePattern_cal - phasePattern_orig

            phasePattern_cal += 3 * d_phi
            phasePattern_orig[k * (x_len // 2) : (k + 1) * (x_len // 2), l * (y_len // 2) : (l + 1) * (y_len // 2)] = (
                phasePattern_cal[k * (x_len // 2) : (k + 1) * (x_len // 2), l * (y_len // 2) : (l + 1) * (y_len // 2)]
            )
        else:
            phasePattern_orig = phasePattern_cal

        self.gui.phasePattern = phasePattern_orig

        self.gui.updatePhase()

        # save hologram information
        timestr = time.strftime("%Y%m%d_%H%M%S")

        hologram_save_file = LOCAL_DIR + '\\holograms\\equalized_with_atoms_' + timestr + '.pkl'
        self.savePattern(name=LOCAL_DIR + '\\holograms\\equalized_with_atoms_' + timestr + '.pkl')

        print('done partial phase correction, saved at : ' + hologram_save_file)

    def GS_intensity(self, measured_intensities: np.ndarray, equalize_gain: float = 1.0) -> np.ndarray | None:
        '''
        Adjust the weight of each trap according to the measured intensities
        Keep the same phase for each trap
        '''
        if not self._ensure_traps_loaded():
            return None
        self.gui.weight_traps *= measured_intensities**-0.5

        phi = self.computeHologram()
        if phi is None:
            return None
        print(phi)

        return phi
