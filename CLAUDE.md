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
- `GUI_SLM.py` — entry point (flags `--no-hero`, `--hero-name` in `gui/helpers.py::parse_launch_args`);
  `Start_SLM_GUI.bat` / `.vbs` start it with `SLM_venv`.
- `libslm/` — device and math code, separate from the UI:
  - `slm.py` — `MeadowlarkSLM(board, …)` (Blink SDK via ctypes): connect / disconnect (serial-number
    check before the LUT is loaded), `uploadHologram`, `loadLut`, `readTemperature` (None in simulate
    mode), `readPixelPitchUm`, sequence mode (1k only: `loadSequence` ≤ `MAX_SEQUENCE_FRAMES_1024`,
    `selectFrame` = `Select_image` + `ImageWriteComplete`), simulate mode. One SDK session per process.
  - `gpu.py` — OpenCL hologram kernels: `trap_phase_to_slm`, `slm_to_trap_amp`.
  - `zernike.py` — Zernike polynomials (RMS-normalised, from opticspy).
- `gui/main_window.py` — `SLMGUI`: shared phase state, `updatePhase()` (the only path that sends
  single frames; the Dynamic mode sequence is the other path), menus (Device / Profile / Settings),
  docks, connect (`_openSimulatedSlm` / `_openRealSlm`), geometry rebuild (`_rebuildForGeometry`).
- `gui/slm_frame.py` — `composeSlmFrame`: hologram + grating, Fresnel, WFC, Zernike, pupil → uint8 frame
  (used by `updatePhase` and the sequence; no Qt).
- `gui/sequence_run.py` (trigger loop, no Qt) and `gui/sequence_thread.py` (its QThread) — Dynamic mode
  Arm / Stop. While a sequence runs, the GUI thread makes no SDK calls (temperature paused, Set LUT refused).
- `gui/panels/` — one panel per file; `arbitrary_arrays/` is a package (`panel.py` UI; `rois.py`,
  `traps.py`, `tweezer_array.py` non-UI; `equalization.py` dormant; `legacy.py` unused old code);
  `modes/` is the Operation mode panel (`logic.py` no Qt; `common.py`, `static.py`, `dynamic.py`).
- `gui/hologram_generator.py` — hologram algorithms (computeHologram, WGS, GS) on top of `libslm/gpu.py`;
  `gui/hologram_handler.py` — Save / Load Hologram files (.pkl `traps_dict`, old .npy / .bmp).
- `gui/parameters.py` — optical parameters as `param.<name>` (`f`, `M`), loaded by `SLMGUI` at every
  start from the working directory's `parameters/parameters.json` (Settings ▸ Parameters…,
  `gui/parameters_dialog.py`). The SLM geometry is NOT in that file: image size
  (`param.ImgResX/ImgResY`), pixel pitch (`param.pslm`) and Zernike correction radius
  (`param.max_radius`) come from the device (size and pitch reported by the SLM at connect, radius set
  by hand; the GUI rebuilds if they change). Wavelength and phase levels come from the profile.
- `gui/settings.py` — settings, profiles (template merge, old-layout migration), device list (JSON I/O,
  no Qt); `gui/device_dialog.py` — Device ▸ Set device parameters….
- `gui/paths.py` — repository and working-directory paths; `gui/constants.py` — shared values and the
  µm → phase factors `positionToStepPerPixel(wavelength_m)`.
- `gui/hero.py` — HEROS remote API (methods are called by name from outside: not dead code);
  `gui/stdio_tee.py` + `gui/log_timestamp.py` — Output log (every line starts with YYYY.MM.DD HH:MM:SS);
  `gui/temperature_log.py` — daily temperature CSVs.
- `defaults/` — tracked defaults; values do not belong in code:
  `profiles/default.json` (every profile parameter; working profiles only override it),
  `parameters/parameters.json` (template of the working parameters file),
  `devices/simulated_slm.json` (geometry of the Simulated SLM, also used by a real SLM until it has
  reported its own).
- `tests/` — pytest: `python -m pytest tests` with Anaconda base (SLM_venv has no pytest; fake Blink DLL
  in `tests/test_slm.py`). `tests/hardware/` — scripts for the real SLM, not collected by pytest
  (`check_slm_connection.py`, `upload_ramp.py`; both have `--simulate`); run them with SLM_venv.

## Working directory (`~/petanque_dev/SLM/`, outside the repo)
- `settings/settings.json` — startup mode (last used / fixed profile), last and startup profile.
- `parameters/parameters.json` — `objective_focal_length_m`, `telescope_magnification` (created from
  the template if missing; old keys for size / pitch / radius are ignored).
- `profiles/<name>.json` — `board`, `elementary` (`phase_levels`, `wavelength_nm`, `grating` {`x_um`,
  `y_um`}; old top-level `grating` is migrated when read), `corrections`, `pupil`, `aberration`,
  `operation_mode` (`mode`; `static.last_hologram`; `dynamic.trigger_timeout_ms`, `dynamic.holograms`
  as file paths, "none" = empty).
- `devices/devices.json` — per real SLM: `name` (`slm<serial>` at the first connect, editable), `board`,
  `serial`, `wfc`, `lut`, `width_px`, `height_px`, `pixel_pitch_um`, `slm_radius_m`; the Simulated SLM
  is built in, uses no correction, and only its geometry is stored, under `simulated_slm`. Old key
  `factory_lut` is still read.
- `devices/<device>/` — one folder per device, named after its `name` (`board<N>` until it has one,
  `simulated_slm` for the Simulated SLM; renamed with the device): `<wfc>.bmp` (manufacturer wavefront
  correction, loaded as `SLMGUI.factory_lut`), `<lut>.LUT` (controller LUT loaded at connect; none set =
  no LUT loaded, the controller keeps its current one), `holograms/` (created at the first connect).
  WFC / LUT are copied there by Device ▸ Set device parameters… ▸ Load…. Files from the old layout
  (directly in `devices/`) are still read, and copied into the device folder at its next connect.
- `logs/` — temperature CSVs, also written to `temperature_logs/` in the repo; not in simulate mode.

## How to work here
- You may refactor and restructure GUI code on your own judgment, but
  explain the change in a short summary of behaviour, not a line-by-line tour.
- Ask before touching anything that changes what is sent to the hardware.
  Verify refactors of the hologram path bit-for-bit against the previous commit.
- Physics/optics calculations (`gui/panels/arbitrary_arrays/` tweezer_array.py,
  traps.py, rois.py; `gui/hologram_generator.py`; `libslm/zernike.py`; `gui/constants.py`; trap
  equalization) follow the clarity-first rules of data analysis code instead:
  explicit formulas, units in names, small reviewable diffs, explain physics
  changes first.
- Don't rename existing attributes, widgets or methods unless asked; new public
  methods use camelCase.
- `dev/` holds development-only code (e.g. `gpu_benchmarks.py`), not used by the GUI; run it with
  SLM_venv (Anaconda's pyopencl does not import).
- Ignore `docs/old codes - for reference/` and `SLM_venv/` unless asked.

## Known open points (not yet verified on the lab PC)
- Blink return codes: `slm.py` treats 0 as success (Meadowlark's own example treats -1 as failure);
  confirmed in practice for `Load_LUT_file` / `Write_image` / `ImageWriteComplete`, not yet for
  `Load_sequence` / `Select_image`.
- Sequence mode on the real SLM: `Load_sequence` image size taken as bytes per frame; whether
  `Load_sequence` already shows frame 0; whether a timed-out `Select_image` still flips the frame or
  shows an SDK error window. The first trigger wait after Arm completes at once without a trigger
  (observed); `sequence_run.py` absorbs it with an extra uncounted wait on frame 0.
- `Get_pixel_pitch` unit is undocumented (reads 17 → taken as µm; 1–100 µm accepted).
- `Create_SDK` bit depth: the manual gives 8 / 12 per model, the header "8 or 16", and Meadowlark's
  example always passes 12; the code passes 8 (works on this SLM) and frames are uint8 only.
- LUT / WFC are calibrated at 785 nm (`slm6801_at785_33C`) while the profile uses 850 nm.
- Saved `.pkl` holograms do not store the wavelength or pixel pitch they were computed for.
- `wgs` in `hologram_generator.py` computes `target_amplitudes_norm` but never uses it (decision pending).
