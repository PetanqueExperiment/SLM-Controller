"""The trap plot window: trap centres of ``rois.json`` files and of the loaded traps overlaid in camera pixels."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg

from gui.panels.arbitrary_arrays.rois import readCentersPx

if TYPE_CHECKING:
    from gui.main_window import SLMGUI


def plotTrapsFile(gui: SLMGUI, path: str) -> None:
    """Add the trap centers of one ``rois.json`` to the trap plot window (created if not open)."""
    try:
        centers = readCentersPx(path)
    except (OSError, ValueError, TypeError) as e:
        print('Could not read trap file:', path, '-', e)
        return
    if centers.size == 0:
        print('No valid "centers" in', path)
        return
    plotTrapCenters(gui, centers, os.path.basename(path))
    print('Plotted %d traps from %s' % (centers.shape[0], path))


def plotTrapCenters(gui: SLMGUI, centers: np.ndarray, label: str) -> None:
    """Add (N, 2) camera-pixel trap centres to the trap plot window (created if not open) under ``label``."""
    win = getattr(gui, '_trap_plot_window', None)
    if win is None or not win.isVisible():
        win = pg.PlotWidget()
        win.resize(600, 600)
        win.setAspectLocked(True)
        win.invertY(True)
        win.showGrid(x=True, y=True, alpha=0.3)
        win.setLabel('bottom', 'x (px)')
        win.setLabel('left', 'y (px)')
        win.addLegend()
        gui._trap_plot_window = win
        gui._trap_plot_count = 0

    color = pg.intColor(gui._trap_plot_count, hues=8)
    gui._trap_plot_count += 1
    name = '%s (%d)' % (label, centers.shape[0])
    win.plot(centers[:, 0], centers[:, 1], pen=None, symbol='o', symbolSize=8, symbolBrush=color, symbolPen=None, name=name)
    if centers.shape[0] <= 200:
        for i, (x, y) in enumerate(centers):
            txt = pg.TextItem(str(i), color=color, anchor=(0, 1))
            txt.setPos(x, y)
            win.addItem(txt)
    win.setWindowTitle('Traps: %s' % name if gui._trap_plot_count == 1 else 'Traps: %d files' % gui._trap_plot_count)
    win.show()
    win.raise_()
    win.activateWindow()
