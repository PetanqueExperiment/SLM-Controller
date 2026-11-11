"""Arbitrary Arrays panel package (see panel.py for the layout of the modules).

``ArbitraryArraysPanel`` is imported on first use, so the non-UI modules (rois, traps, hologram_files) can be used
without Qt or the OpenCL GPU module.
"""

from __future__ import annotations

__all__ = ["ArbitraryArraysPanel"]


def __getattr__(name: str) -> type:
    if name == "ArbitraryArraysPanel":
        from gui.panels.arbitrary_arrays.panel import ArbitraryArraysPanel

        return ArbitraryArraysPanel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
