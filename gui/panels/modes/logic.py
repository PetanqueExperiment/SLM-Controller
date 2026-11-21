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


DYNAMIC_READY_TEXT: str = 'Ready'
DYNAMIC_EDITING_TEXT: str = 'Editing'
MIN_TRIGGER_TIMEOUT_MS: int = 1
MAX_TRIGGER_TIMEOUT_MS: int = 60000
TRIGGER_TIMEOUT_PROFILE_KEY: str = 'trigger_timeout_ms'
HOLOGRAMS_PROFILE_KEY: str = 'holograms'
PROFILE_NO_HOLOGRAM: str = 'none'


def clampTriggerTimeoutMs(value: float) -> int:
    """``value`` as a whole number of ms within MIN_TRIGGER_TIMEOUT_MS … MAX_TRIGGER_TIMEOUT_MS."""
    return min(max(int(round(value)), MIN_TRIGGER_TIMEOUT_MS), MAX_TRIGGER_TIMEOUT_MS)


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
    """One frame of the dynamic hologram list: the file it was loaded from and its contents (None until loaded).

    ``data`` is the hologram file's dict: a ``.pkl`` traps_dict, or ``{'phasePattern': array}`` for ``.npy`` / ``.bmp``.
    """

    path: str | None = None
    data: dict[str, Any] | None = None


class DynamicMode(OperationMode):
    """Placeholder for the dynamic mode: holds the list of holograms; driving the SLM with them is not implemented yet."""

    name = 'Dynamic'

    def __init__(self) -> None:
        self.holograms: list[HologramSlot] = [HologramSlot()]

    def buildPage(self, gui: SLMGUI) -> QtWidgets.QWidget:
        from gui.panels.modes.dynamic import DynamicModePage

        return DynamicModePage(self, gui)

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

    def missingSlots(self) -> list[int]:
        """Indices of slots without a hologram."""
        return [i for i, slot in enumerate(self.holograms) if slot.data is None]

    def setHologram(self, index: int, path: str, data: dict[str, Any]) -> None:
        """Store the hologram loaded from ``path`` in slot ``index``."""
        slot = self.holograms[index]
        slot.path = path
        slot.data = data

    def profileHolograms(self) -> list[str]:
        """File of every slot for the profile (PROFILE_NO_HOLOGRAM for an empty slot)."""
        return [PROFILE_NO_HOLOGRAM if slot.path is None else slot.path for slot in self.holograms]

    def loadProfileHolograms(self, paths: list[Any]) -> list[str]:
        """Replace the slots by the profile's files (an empty list gives one empty slot).

        A file that is missing or unreadable leaves its slot empty; returns one description per such file.
        """
        slots: list[HologramSlot] = []
        problems: list[str] = []
        for index, path in enumerate(paths):
            if not isinstance(path, str) or not path or path == PROFILE_NO_HOLOGRAM:
                slots.append(HologramSlot())
                continue
            try:
                slots.append(HologramSlot(path, readHologramFile(path)))
            except Exception as e:
                problems.append('frame %d: %s (%s)' % (index + 1, path, e))
                slots.append(HologramSlot())
        self.holograms = slots or [HologramSlot()]
        return problems


OPERATION_MODES: tuple[type[OperationMode], ...] = (StaticMode, DynamicMode)
DEFAULT_OPERATION_MODE: type[OperationMode] = StaticMode
