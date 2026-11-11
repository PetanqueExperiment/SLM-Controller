"""Operation modes without Qt: mode classes, the dynamic hologram list and reading hologram files.

Each mode builds its page of controls with ``buildPage`` (static.py / dynamic.py, imported only then), so this file
can be imported and tested without Qt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from gui.hologram_handler import readLegacyPattern, readTrapsDict

if TYPE_CHECKING:
    from pyqtgraph.Qt import QtWidgets

    from gui.main_window import SLMGUI

SLOT_KEEP_UNTIL_NEXT_TRIGGER: str = 'keep until next trig'
SLOT_START_STOP: str = 'make start-stop'
SLOT_TIMINGS: tuple[str, ...] = (SLOT_KEEP_UNTIL_NEXT_TRIGGER, SLOT_START_STOP)

DYNAMIC_READY_TEXT: str = 'Ready'
DYNAMIC_EDITING_TEXT: str = 'Editing'


def ordinal(n: int) -> str:
    """1 → '1st', 2 → '2nd', 3 → '3rd', 4 → '4th', 11 → '11th', 21 → '21st'."""
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return '%d%s' % (n, suffix)


def readHologramFile(path: str) -> dict[str, Any]:
    """Contents of a hologram file as a dict with at least 'phasePattern' (.pkl traps_dict, or old .npy / .bmp)."""
    low = path.lower()
    if low.endswith('.pkl'):
        data = readTrapsDict(path)
        if not isinstance(data, dict) or 'phasePattern' not in data:
            raise ValueError('not a hologram file (no phasePattern)')
        return data
    if low.endswith('.npy') or low.endswith('.bmp'):
        return {'phasePattern': readLegacyPattern(path)}
    raise ValueError('unsupported file type (use .pkl, .npy or .bmp)')


class OperationMode:
    """Base class of an operation mode. ``name`` is what the Operation mode selector shows."""

    name: str = ''

    def activate(self) -> None:
        """Called when the mode is selected."""

    def deactivate(self) -> None:
        """Called when another mode is selected."""

    def buildPage(self, gui: SLMGUI) -> QtWidgets.QWidget | None:
        """The controls shown under the selector while this mode is selected (built once); None for an empty page."""
        return None

    def statusText(self, gui: SLMGUI) -> str | None:
        """Status shown after '|' next to the selector while this mode is selected; None shows nothing."""
        return None


class StaticMode(OperationMode):
    """One hologram on the SLM, recomputed and resent only when a setting changes (the current behaviour)."""

    name = 'Static'

    def buildPage(self, gui: SLMGUI) -> QtWidgets.QWidget:
        from gui.panels.modes.static import StaticModePage

        return StaticModePage(gui)


@dataclass
class HologramSlot:
    """One entry of the dynamic hologram list (paths and contents are None until loaded).

    ``timing`` is one of SLOT_TIMINGS. With SLOT_START_STOP the entry also has a stop hologram; it is kept
    (but unused) when the timing is switched back. ``data`` / ``stop_data`` are the hologram file's dict:
    a ``.pkl`` traps_dict, or ``{'phasePattern': array}`` for ``.npy`` / ``.bmp``.
    """

    path: str | None = None
    data: dict[str, Any] | None = None
    timing: str = SLOT_KEEP_UNTIL_NEXT_TRIGGER
    stop_path: str | None = None
    stop_data: dict[str, Any] | None = None


class DynamicMode(OperationMode):
    """Placeholder for the dynamic mode: holds the list of holograms; driving the SLM with them is not implemented yet."""

    name = 'Dynamic'

    def __init__(self) -> None:
        self.holograms: list[HologramSlot] = [HologramSlot()]

    def buildPage(self, gui: SLMGUI) -> QtWidgets.QWidget:
        from gui.panels.modes.dynamic import DynamicModePage

        return DynamicModePage(self)

    def statusText(self, gui: SLMGUI) -> str | None:
        return DYNAMIC_READY_TEXT if gui.dynamicIsReady else DYNAMIC_EDITING_TEXT

    def addSlot(self) -> int:
        """Append an empty slot and return its index."""
        self.holograms.append(HologramSlot())
        return len(self.holograms) - 1

    def removeLastSlot(self) -> bool:
        """Remove the last slot (the list always keeps one). Returns True if a slot was removed."""
        if len(self.holograms) <= 1:
            return False
        self.holograms.pop()
        return True

    def setHologram(self, index: int, path: str, data: dict[str, Any]) -> None:
        """Store the hologram loaded from ``path`` in slot ``index``."""
        slot = self.holograms[index]
        slot.path = path
        slot.data = data

    def setStopHologram(self, index: int, path: str, data: dict[str, Any]) -> None:
        """Store the stop hologram loaded from ``path`` in slot ``index``."""
        slot = self.holograms[index]
        slot.stop_path = path
        slot.stop_data = data

    def setTiming(self, index: int, timing: str) -> None:
        """Set the timing of slot ``index`` (one of SLOT_TIMINGS)."""
        if timing not in SLOT_TIMINGS:
            raise ValueError('unknown timing %r' % timing)
        self.holograms[index].timing = timing


OPERATION_MODES: tuple[type[OperationMode], ...] = (StaticMode, DynamicMode)
DEFAULT_OPERATION_MODE: type[OperationMode] = StaticMode
