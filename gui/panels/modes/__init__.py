"""SLM operation modes and the Operation mode panel.

logic.py: mode classes, dynamic hologram list, hologram file reading (no Qt); common.py: the panel shared by all
modes (selector, status, page of the selected mode); static.py / dynamic.py: the page of each mode.

``OperationModePanel`` is imported on first use, so logic.py can be used without Qt.
"""

from __future__ import annotations

from gui.panels.modes.logic import DEFAULT_OPERATION_MODE, OPERATION_MODES, DynamicMode, OperationMode, StaticMode

__all__ = ["DEFAULT_OPERATION_MODE", "OPERATION_MODES", "DynamicMode", "OperationMode", "OperationModePanel", "StaticMode"]


def __getattr__(name: str) -> type:
    if name == "OperationModePanel":
        from gui.panels.modes.common import OperationModePanel

        return OperationModePanel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
