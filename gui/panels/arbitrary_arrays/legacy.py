"""Old, unused code of the Arbitrary Arrays panel, kept unchanged for reference. Nothing in the GUI calls it.

Known not to work as is:
  - ``randomMask`` uses ``gui.y_slm`` and ``GAA`` uses ``gui.delta_traps``; neither exists any more.
  - ``devide_slm_area_loop`` needs ``trap_equalization_ramsey``, now only in ``docs/old codes - for reference/``.
  - ``Equalization_intensity_thread`` is never started.

``LegacyAlgorithmsMixin`` is a mixin of ArbitraryArraysPanel (uses ``gui``, ``computeHologram``, ``_ensure_traps_loaded``,
``iterationsEdit``, ``devide_slm_area``).
"""

from __future__ import annotations

from importlib import reload

import numpy as np
from PyQt5.QtCore import pyqtSignal
from pyqtgraph.Qt import QtCore

import libslm.parameters as param
from gui.constants import MESSAGE_FILE

try:
    import trap_equalization_ramsey as trap_equalization
except ImportError:
    trap_equalization = None


class Equalization_intensity_thread(QtCore.QThread):
    '''
    This thread continuously look at the file 'message.txt' (MESSAGE_FILE)
    to communicate with the trap camera computer

    When we want to equalize the trap intensities, this computer calculates
    the phase pattern, and the other computer measures the trap intensities

    Two messages can be written on MESSAGE_FILE:

        1) 'run GS' : written by the trap camera computer to indicate that
        the intensities have been measured and written in FILE_INTENSITY
        this computer can now do a new iteration of the GS algorithm

        2) 'measure intensity i' : written by this computer where i is the
        iteration we are doing. when the other computer reads this message,
        it will measure the intensity and then write the message 1)

    In this thread we launch the runNextIter signal as soon as we read 'run GS'
    If the thread instance self.last is True, we launch the finishIter signal

    These two signal are connected to the function iterate_GS and finish_equalization.
    -    The thread is created when the button Equalize traps is pressed
    The thread is destroyed at the end of the function finish_equalization
    '''

    sigNext = pyqtSignal()
    sigLast = pyqtSignal()

    def __init__(self) -> None:
        QtCore.QThread.__init__(self)
        self.go_on_reading_message = False
        self.timer = QtCore.QTimer()
        self.last = False

    def __del__(self) -> None:
        self.wait()

    def run(self) -> None:
        self.keepCheckingFile()

    def keepCheckingFile(self) -> None:

        if self.go_on_reading_message:
            # Check the file for instruction
            with open(MESSAGE_FILE, 'r') as message_read:
                text = [x.strip().split('\t') for x in message_read]
            message_read.close()

            # Check instruction
            try:
                instruction = text[0][0]
            except Exception:
                instruction = ''

            if instruction == 'run GS':
                self.go_on_reading_message = False

                if self.last:
                    ## send signal to main thread to finish
                    self.sigLast.emit()
                else:
                    ## send signal to main thread to run the GSI
                    self.sigNext.emit()

        # repeat the loop.
        self.timer.singleShot(0, self.keepCheckingFile)


class LegacyAlgorithmsMixin:
    def devide_slm_area_loop(self, loop: int | None = None) -> None:

        global trap_equalization
        trap_equalization = reload(trap_equalization)
        # best_hologram =
        folder_dir = self.gui.parent.experiment_directory

        change = True
        print('loop num', loop)

        if loop == 0:
            pass
        else:
            loop = int(loop)
            loop = int(loop - 1)
            if loop != 0.0:
                current_map = self.gui.map

            self.gui.map = trap_equalization.cal(folder_dir, loop=loop)

            std = np.std(self.gui.map)  # martin
            print(std)
            if std > self.gui.std and (self.gui.std * 100) < 0.5:  # martin
                change = False
                self.gui.map = current_map
                self.gui.phasePattern = self.gui.best_hologram

            else:  # martin
                self.gui.std = std  # martin
                self.gui.best_hologram = self.gui.phasePattern  # martin
                print('new point')  # martin

            self.devide_slm_area(change, signal=1)

    def randomMask(self) -> np.ndarray | None:
        '''
        random mask (RM)
        '''
        if not self._ensure_traps_loaded():
            return None
        randmat = np.ceil(np.random.rand(param.ImgResY, param.ImgResX) * self.gui.N_traps).astype(int)

        phi = np.zeros((param.ImgResY, param.ImgResX))

        for i in range(param.ImgResY):
            for j in range(param.ImgResX):
                random_trap = randmat[i, j] - 1

                k_x, k_y = self.gui.traps_k[random_trap]
                phi[i, j] = self.gui.i_slm * k_x + self.gui.y_slm * k_y

        return phi

    def superposition(self) -> np.ndarray | None:
        '''
        superposition of gratings and lenses (S)
        '''
        if not self._ensure_traps_loaded():
            return None
        self.gui.theta_traps = np.zeros(self.gui.N_traps)
        self.gui.weight_traps = np.ones(self.gui.N_traps)

        phi = self.computeHologram()

        return phi

    def randomSuperposition(self) -> np.ndarray | None:
        '''
        Superposition of gratings with random phases.
        '''
        if not self._ensure_traps_loaded():
            return None
        self.gui.theta_traps = np.random.rand(self.gui.N_traps) * 2 * np.pi
        self.gui.weight_traps = np.ones(self.gui.N_traps)

        phi = self.computeHologram()

        return phi

    def GAA(self) -> np.ndarray | None:
        '''
        Generalized adaptive additive
        '''
        if not self._ensure_traps_loaded():
            return None
        GAA_maxit = int(self.iterationsEdit.text())
        norm = 1 / param.ImgResY / param.ImgResX
        Vtraps_it = np.zeros(self.gui.N_traps, dtype=complex)

        phi = self.randomSuperposition()
        if phi is None:
            return None
        print('SR guess created')

        # Measure the trap to trap deviation of intensity
        for k in range(self.gui.N_traps):
            Vtraps_it[k] = norm * np.sum(np.sum(np.exp(1j * (phi - self.gui.delta_traps[:, :, k])), axis=0), axis=0)
        trap_intensities = np.real(Vtraps_it * np.conj(Vtraps_it))
        efficiency = np.sum(trap_intensities)
        uniformity = np.std(trap_intensities) / np.mean(trap_intensities)
        print('Efficiency std = %.1f' % (efficiency * 100))
        print('Intensity std = %.1f' % (uniformity * 100))

        for k in range(GAA_maxit):

            print('Iteration ' + str(k + 1) + '/' + str(GAA_maxit))
            Vtraps_it_mod = np.sqrt(Vtraps_it * np.conj(Vtraps_it))
            gain = 1 - self.gui.GAA_xi + self.gui.GAA_xi / Vtraps_it_mod
            phi = np.angle(np.sum(np.exp(1j * self.gui.delta_traps) * Vtraps_it / Vtraps_it_mod * gain, axis=2))

            for j in range(self.gui.N_traps):
                Vtraps_it[j] = norm * np.sum(np.sum(np.exp(1j * (phi - self.gui.delta_traps[:, :, j])), axis=0), axis=0)
            trap_intensities = Vtraps_it * np.conj(Vtraps_it)
            efficiency = np.real(np.sum(trap_intensities))
            uniformity = np.std(trap_intensities) / np.mean(trap_intensities)
            print('Efficiency std = %.1f' % (efficiency * 100))
            print('Intensity std = %.1f' % (uniformity * 100))

        return phi
