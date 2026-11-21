"""SLM and optics parameters, loaded from ``~/petanque_dev/SLM/parameters/parameters.json`` (no Qt).

``param`` starts with the tracked defaults (defaults/parameters/parameters.json); :func:`loadParameters` (called by
SLMGUI at every start) replaces them in place with the working file, creating it from the defaults if missing. Code
reads the values at use time as ``param.<name>``.

The SLM image size ``param.ImgResX`` / ``param.ImgResY`` [px], pixel pitch ``param.pslm`` [m] and SLM radius
``param.max_radius`` [m] (Zernike unit disk) are not in this file: they belong to the device (devices.json; size and
pitch reported by the SLM at connect) and are set with :func:`setImageSize`, :func:`setPixelPitchUm` and
:func:`setSlmRadiusM`.

JSON key                    attribute        meaning
objective_focal_length_m    f                effective focal length of the objective [m]
telescope_magnification     M                magnification of the telescope between SLM and objective
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gui.paths import PARAMETERS_FILE, PARAMETERS_TEMPLATE_FILE
from gui.settings import DeviceDefaultsError, loadDeviceDefaults

JSON_KEYS: dict[str, str] = {
    'f': 'objective_focal_length_m',
    'M': 'telescope_magnification',
}
OBSOLETE_KEYS: frozenset[str] = frozenset({'slm_width_px', 'slm_height_px', 'slm_pixel_pitch_m', 'slm_max_radius_m'})


class ParametersError(RuntimeError):
    """The parameters file is missing or cannot be used."""


@dataclass
class SlmParameters:
    f: float
    M: float
    max_radius: float = 0.0  # device geometry: set from the device (defaults/devices/simulated_slm.json until then)
    ImgResX: int = 0
    ImgResY: int = 0
    pslm: float = 0.0


def _value(name: str, raw: Any) -> float:
    """Validated value for attribute ``name``: a positive finite number."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f'{JSON_KEYS[name]} must be a number, got {raw!r}')
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{JSON_KEYS[name]} must be a positive number, got {raw!r}')
    return value


def _readJson(path: Path) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as fh:
        doc = json.load(fh)
    if not isinstance(doc, dict):
        raise ValueError('expected a JSON object')
    return doc


def readTemplate(path: Path = PARAMETERS_TEMPLATE_FILE) -> SlmParameters:
    """The tracked default parameters; raises ParametersError if the template is missing or incomplete."""
    try:
        doc = _readJson(path)
        return SlmParameters(**{name: _value(name, doc[key]) for name, key in JSON_KEYS.items()})
    except (OSError, ValueError, KeyError) as e:
        raise ParametersError(f'Default parameters template missing or invalid: {path} ({e})') from e


def readParameters(path: Path, defaults: SlmParameters) -> tuple[SlmParameters, list[str]]:
    """Parameters from ``path``; missing or invalid entries take the value from ``defaults``.

    Returns the parameters and a list of warnings (empty if the file was complete and valid). An unreadable file
    gives all defaults and one warning.
    """
    try:
        doc = _readJson(path)
    except (OSError, ValueError) as e:
        return SlmParameters(**{name: getattr(defaults, name) for name in JSON_KEYS}), [f'{path} could not be read ({e}): all default values are used']
    values: dict[str, Any] = {}
    warnings: list[str] = []
    for name, key in JSON_KEYS.items():
        if key not in doc:
            values[name] = getattr(defaults, name)
            warnings.append(f'{key} is missing: default {values[name]!r} is used')
            continue
        try:
            values[name] = _value(name, doc[key])
        except ValueError as e:
            values[name] = getattr(defaults, name)
            warnings.append(f'{e}: default {values[name]!r} is used')
    unknown = sorted(set(doc) - set(JSON_KEYS.values()) - OBSOLETE_KEYS)
    if unknown:
        warnings.append(f'unknown keys ignored: {", ".join(unknown)}')
    return SlmParameters(**values), warnings


def loadParameters(path: Path = PARAMETERS_FILE, template: Path = PARAMETERS_TEMPLATE_FILE) -> list[str]:
    """Load ``path`` into ``param`` (in place), creating it from ``template`` if missing. Returns warnings to show."""
    defaults = readTemplate(template)
    warnings: list[str] = []
    if not path.is_file():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(template, path)
            print(f'Created {path} from the default parameters')
        except OSError as e:
            warnings.append(f'{path} could not be created ({e}): default values are used')
            _apply(defaults)
            return warnings
    loaded, warnings = readParameters(path, defaults)
    _apply(loaded)
    print(f'Loaded parameters from {path}')
    return warnings


def writeParameters(values: dict[str, Any], path: Path = PARAMETERS_FILE) -> None:
    """Validate ``values`` (attribute name -> value, all parameters) and write them to ``path`` with the JSON keys.

    Raises ValueError naming the first invalid value; the file is then not changed. ``param`` is not changed: the
    new values are used from the next start.
    """
    missing = [name for name in JSON_KEYS if name not in values]
    if missing:
        raise ValueError(f'missing parameters: {", ".join(JSON_KEYS[name] for name in missing)}')
    doc = {key: _value(name, values[name]) for name, key in JSON_KEYS.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, indent=2)
        fh.write('\n')


def _apply(values: SlmParameters) -> None:
    for name in JSON_KEYS:
        setattr(param, name, getattr(values, name))


def setImageSize(width_px: int, height_px: int) -> None:
    """Set the SLM image size ``param.ImgResX`` / ``param.ImgResY`` [px]; raises ValueError if not positive integers."""
    for label, value in (('width', width_px), ('height', height_px)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'SLM {label} must be a positive integer, got {value!r}')
    param.ImgResX, param.ImgResY = width_px, height_px


def setPixelPitchUm(pitch_um: float) -> None:
    """Set the SLM pixel pitch ``param.pslm`` [m] from ``pitch_um`` [µm]; raises ValueError if not a positive number."""
    if isinstance(pitch_um, bool) or not isinstance(pitch_um, (int, float)) or not math.isfinite(pitch_um) or pitch_um <= 0:
        raise ValueError(f'SLM pixel pitch must be a positive number of µm, got {pitch_um!r}')
    param.pslm = float(pitch_um) / 1e6


def setSlmRadiusM(radius_m: float) -> None:
    """Set the SLM radius ``param.max_radius`` [m] (Zernike unit disk); raises ValueError if not a positive number."""
    if isinstance(radius_m, bool) or not isinstance(radius_m, (int, float)) or not math.isfinite(radius_m) or radius_m <= 0:
        raise ValueError(f'SLM radius must be a positive number of m, got {radius_m!r}')
    param.max_radius = float(radius_m)


def applyDeviceDefaults() -> None:
    """Set the device geometry in ``param`` to the defaults (defaults/devices/simulated_slm.json)."""
    try:
        defaults = loadDeviceDefaults()
    except DeviceDefaultsError as e:
        raise ParametersError(str(e)) from e
    setImageSize(defaults.width_px, defaults.height_px)
    setPixelPitchUm(defaults.pixel_pitch_um)
    setSlmRadiusM(defaults.slm_radius_m)


param: SlmParameters = readTemplate()
applyDeviceDefaults()
