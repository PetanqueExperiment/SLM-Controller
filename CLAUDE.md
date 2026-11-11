# SLM-Controller — production quality

PyQt5/pyqtgraph GUI driving a Meadowlark SLM (hologram computation on GPU
via pyopencl, optional HEROS remote control). I don't review GUI code line
by line; I judge it by behaviour. It must be robust in daily lab use.

## Standard
- Production quality: robust error handling, no crashes from bad input or a
  missing device, clear user-facing error messages, nothing that freezes the UI.
- Long work (hologram computation, device I/O, HEROS calls) off the GUI
  thread; UI updates only from the GUI thread (signals/slots).
- Must keep working with the Simulated SLM (permanent device-list entry, board 0:
  selecting it is simulate mode; there is no --simulate flag), without HEROS
  (`--no-hero`), and with no SLM connected (the GUI starts fully; connecting is the
  last startup step).
- Never change saved file formats (`SLM_patterns/*.pkl`, `Tweezer_ROIs/rois.json`,
  `Zernike_corrections/`, the JSON files under `~/petanque_dev/SLM/`) without a
  backward-compatible loader.
- Add tests (pytest) for non-UI logic you touch, where feasible.

## Layout
- `GUI_SLM.py` — entry point (flags in `gui/helpers.py::parse_launch_args`).
- `libslm/` — device code, separate from the UI:
  - `slm.py` — `MeadowlarkSLM` (Blink SDK via ctypes): connect / disconnect,
    serial-number check before the LUT is loaded, `uploadHologram`,
    `readTemperature`, WFC correction, simulate mode. One SDK session per process.
  - `gpu.py` — OpenCL hologram kernels: `trap_phase_to_slm`, `slm_to_trap_amp`.
  - `parameters.py` — optical / device constants (`f`, `M`, `pslm`, image size).
    Wavelength and P are NOT here: they come from the profile.
- `gui/main_window.py` — `SLMGUI`: shared phase state, `updatePhase()` (the only
  path that sends frames), menus (Device / Profiles / Settings), docks.
- `gui/panels/` — one panel per file; `arbitrary_arrays/` is a package
  (`panel.py` UI; `rois.py`, `traps.py`, `algorithms.py`, `hologram_files.py`,
  `tweezer_array.py` non-UI; `equalization.py` dormant; `legacy.py` unused old code).
- `gui/settings.py` — settings, profiles, device list (JSON I/O, no Qt).
- `gui/defaults.py` — working-directory paths; `gui/constants.py` — shared values
  and the µm → phase factors `positionToStepPerPixel(wavelength_m)`.
- `profile_templates/default.json` — tracked default values of every profile
  parameter; working profiles only override it. Values do not belong in code.
- `tests/` — pytest: `python -m pytest tests` (fake Blink DLL in `tests/test_slm.py`).

## Working directory (`~/petanque_dev/SLM/`, outside the repo)
`settings/settings.json` (startup mode, last profile), `profiles/<name>.json`,
`devices/devices.json` (board, serial and `factory_lut` per real SLM; the Simulated SLM is built in and uses no correction), `devices/<factory_lut>.bmp` (per-SLM manufacturer correction, loaded as `SLMGUI.factory_lut`), `logs/` (temperature CSVs, also
written to `temperature_logs/` in the repo; not in simulate mode).

## How to work here
- You may refactor and restructure GUI code on your own judgment, but
  explain the change in a short summary of behaviour, not a line-by-line tour.
- Ask before touching anything that changes what is sent to the hardware.
  Verify refactors of the hologram path bit-for-bit against the previous commit.
- Physics/optics calculations (`gui/panels/arbitrary_arrays/` tweezer_array.py,
  traps.py, algorithms.py, rois.py; `gui/zernike.py`; `gui/constants.py`; trap
  equalization) follow the clarity-first rules of data analysis code instead:
  explicit formulas, units in names, small reviewable diffs, explain physics
  changes first.
- Don't rename existing attributes, widgets or methods unless asked; new public
  methods use camelCase.
- `dev/` holds development-only code (e.g. `gpu_benchmarks.py`), not used by the GUI.
- Ignore `docs/old codes - for reference/` and `SLM_venv/` unless asked.

## Known open points (not yet verified on the lab PC)
- Blink return codes: `slm.py` treats 0 as success for `Load_LUT_file` /
  `Write_image` / `ImageWriteComplete`; slmsuite uses 1. Check `Blink_C_wrapper.h`.
- `Read_Serial_Number(board)` signature taken from slmsuite.
- LUT / WFC are calibrated at 785 nm (`slm6801_at785_33C`) while the profile uses 850 nm.
- Saved `.pkl` holograms do not store the wavelength they were computed for.
