"""SLM and optics parameters, loaded from ``~/petanque_dev/SLM/parameters/parameters.json`` (no Qt).

``param`` starts with the tracked defaults (profile_templates/parameters.json); :func:`loadParameters` (called by
SLMGUI at every start) replaces them in place with the working file, creating it from the defaults if missing. Code
reads the values at use time as ``param.<name>``.

JSON key                    attribute        meaning
slm_width_px                ImgResX          hologram width [px]
slm_height_px               ImgResY          hologram height [px]
objective_focal_length_m    f                effective focal length of the objective [m]
telescope_magnification     M                magnification of the telescope between SLM and objective
slm_pixel_pitch_m           pslm             SLM pixel pitch [m]
slm_max_radius_m            max_radius       radius used to normalise the Zernike polynomials [m]
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from gui.paths import PARAMETERS_FILE, PARAMETERS_TEMPLATE_FILE

JSON_KEYS: dict[str, str] = {
    'ImgResX': 'slm_width_px',
    'ImgResY': 'slm_height_px',
    'f': 'objective_focal_length_m',
    'M': 'telescope_magnification',
    'pslm': 'slm_pixel_pitch_m',
    'max_radius': 'slm_max_radius_m',
}


class ParametersError(RuntimeError):
    """The parameters file is missing or cannot be used."""


@dataclass
class SlmParameters:
    ImgResX: int
    ImgResY: int
    f: float
    M: float
    pslm: float
    max_radius: float


def _value(name: str, raw: Any) -> int | float:
    """Validated value for attribute ``name``: positive int for the image size, positive finite number otherwise."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f'{JSON_KEYS[name]} must be a number, got {raw!r}')
    if name in ('ImgResX', 'ImgResY'):
        if not isinstance(raw, int) or raw <= 0:
            raise ValueError(f'{JSON_KEYS[name]} must be a positive integer, got {raw!r}')
        return raw
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
        return SlmParameters(**{f.name: getattr(defaults, f.name) for f in fields(SlmParameters)}), [f'{path} could not be read ({e}): all default values are used']
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
    unknown = sorted(set(doc) - set(JSON_KEYS.values()))
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


def _apply(values: SlmParameters) -> None:
    for f in fields(SlmParameters):
        setattr(param, f.name, getattr(values, f.name))


param: SlmParameters = readTemplate()
