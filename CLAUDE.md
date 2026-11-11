# SLM-Controller — production quality

PyQt5/pyqtgraph GUI driving a Meadowlark SLM (hologram computation on GPU
via pyopencl, optional HEROS remote control). I don't review GUI code line
by line; I judge it by behaviour. It must be robust in daily lab use.

## Standard
- Production quality: robust error handling, no crashes from bad input or a
  missing device, clear user-facing error messages, nothing that freezes the UI.
- Long work (hologram computation, device I/O, HEROS calls) off the GUI
  thread; UI updates only from the GUI thread (signals/slots).
- Must keep working in simulation mode (`Start_SLM_GUI_simulate.bat`,
  `--simulate`) and without HEROS (`--no-hero`).
- Follow the existing architecture: entry point `GUI_SLM.py`, window in
  `gui/main_window.py`, one panel per file in `gui/panels/`, shared values in
  `gui/constants.py`. Keep device code (`SLM_device.py`) separate from UI.
- Never change saved file formats (`SLM_patterns/`, `Tweezer_ROIs/`,
  `Zernike_corrections/`) without a backward-compatible loader.
- Add tests (pytest) for non-UI logic you touch, where feasible.

## How to work here
- You may refactor and restructure GUI code on your own judgment, but
  explain the change in a short summary of behaviour, not a line-by-line tour.
- Ask before touching anything that changes what is sent to the hardware.
- Physics/optics calculations (`tweezer_array.py`, `zernike.py`, trap
  equalization, fitting in `gui/panels/arbitrary_arrays.py`) follow the
  clarity-first rules of data analysis code instead: explicit formulas,
  units in names, small reviewable diffs, explain physics changes first.
- Ignore `docs/old codes - for reference/` and `SLM_venv/` unless asked.
