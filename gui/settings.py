"""App settings (``settings/settings.json``), named profiles (``profiles/<name>.json``) and the SLM list (``devices/devices.json``)."""

from __future__ import annotations

import copy
import json
import re
import shutil
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from gui.paths import DEFAULT_PROFILE, DEFAULT_STARTUP_MODE, DEVICES_DIR, DEVICES_FILE, PROFILE_DIR, PROFILE_TEMPLATE_FILE, SETTINGS_FILE
from gui.paths import STARTUP_LAST_USED, STARTUP_MODES


def _read_json(path: Path) -> dict[str, Any] | None:
    """Return the JSON object in ``path``, or None if the file is missing or not a valid JSON object."""
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError) as e:
        print(f"Could not read {path}: {e}")
        return None
    if not isinstance(doc, dict):
        print(f"Ignoring {path}: expected a JSON object")
        return None
    return doc


def _write_json(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")


@dataclass
class AppSettings:
    """Application-level choices kept across sessions."""

    startup_mode: str = DEFAULT_STARTUP_MODE
    last_profile: str = DEFAULT_PROFILE

    @classmethod
    def load(cls, path: Path = SETTINGS_FILE) -> AppSettings:
        """Read settings; missing file, unknown keys or invalid values fall back to the defaults."""
        settings = cls()
        doc = _read_json(path) or {}
        for field in fields(cls):
            value = doc.get(field.name)
            if isinstance(value, str) and value:
                setattr(settings, field.name, value)
        if settings.startup_mode not in STARTUP_MODES:
            print(f"Unknown startup mode {settings.startup_mode!r} in {path}; using {DEFAULT_STARTUP_MODE!r}")
            settings.startup_mode = DEFAULT_STARTUP_MODE
        return settings

    def save(self, path: Path = SETTINGS_FILE) -> None:
        _write_json(path, asdict(self))

    def startupProfile(self) -> str:
        """Name of the profile to load when the app starts."""
        if self.startup_mode == STARTUP_LAST_USED:
            return self.last_profile
        return DEFAULT_PROFILE


def profilePath(name: str) -> Path:
    return PROFILE_DIR / f"{name}.json"


def readProfile(name: str) -> dict[str, Any] | None:
    """Parameters stored in profile ``name``, or None if it does not exist yet."""
    return _read_json(profilePath(name))


def writeProfile(name: str, params: dict[str, Any]) -> Path:
    path = profilePath(name)
    _write_json(path, params)
    return path


class ProfileTemplateError(RuntimeError):
    """The default profile template in the repository (profile_templates/default.json) is missing or invalid."""


def loadProfileTemplate() -> dict[str, Any]:
    """Default values of every profile parameter, from the repository template."""
    template = _read_json(PROFILE_TEMPLATE_FILE)
    if template is None:
        raise ProfileTemplateError(f"Default profile template missing or invalid: {PROFILE_TEMPLATE_FILE}")
    return template


def _sameKind(default: Any, value: Any) -> bool:
    """True if ``value`` can replace ``default``: bool for bool, number for number (not bool), str for str."""
    if isinstance(default, bool) or isinstance(value, bool):
        return isinstance(default, bool) and isinstance(value, bool)
    if isinstance(default, (int, float)):
        return isinstance(value, (int, float))
    return isinstance(value, type(default))


def mergeProfile(template: dict[str, Any], profile: dict[str, Any], where: str = "") -> dict[str, Any]:
    """Template values overridden by ``profile``; keys only in the profile (e.g. ``board``) are kept, wrong types ignored."""
    merged = copy.deepcopy(template)
    for key, value in profile.items():
        name = f"{where}{key}"
        if key not in template:
            merged[key] = value
        elif isinstance(template[key], dict):
            if isinstance(value, dict):
                merged[key] = mergeProfile(template[key], value, f"{name}.")
            else:
                print(f"Profile: ignoring {name} = {value!r} (expected a section)")
        elif _sameKind(template[key], value):
            merged[key] = value
        else:
            print(f"Profile: ignoring {name} = {value!r} (expected {type(template[key]).__name__}); using default {template[key]!r}")
    return merged


PROFILE_LOADED: str = "loaded"
PROFILE_MISSING: str = "missing"
PROFILE_UNREADABLE: str = "unreadable"


def loadProfileParams(name: str) -> tuple[dict[str, Any], str]:
    """Full parameters of profile ``name``: the repository template overridden by ``profiles/<name>.json``.

    Returns the parameters and the state of the file: PROFILE_LOADED, PROFILE_MISSING or PROFILE_UNREADABLE
    (exists but is not a valid JSON object; the defaults are used).
    """
    template = loadProfileTemplate()
    profile = readProfile(name)
    if profile is not None:
        return mergeProfile(template, profile), PROFILE_LOADED
    return template, PROFILE_UNREADABLE if profilePath(name).exists() else PROFILE_MISSING


def restoreDefaultProfile() -> Path:
    """Overwrite ``profiles/default.json`` with the repository template; return the written path."""
    loadProfileTemplate()
    path = profilePath(DEFAULT_PROFILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PROFILE_TEMPLATE_FILE, path)
    return path


_PROFILE_NAME_RE = re.compile(r"[\w\- ]+")


def isValidProfileName(name: str) -> bool:
    """Letters, digits, spaces, '-' and '_' only, with no leading/trailing spaces (the name is used as a file name)."""
    return bool(_PROFILE_NAME_RE.fullmatch(name)) and name == name.strip()


def listProfiles() -> list[str]:
    """Names of the profiles saved in ``profiles/`` (sorted)."""
    if not PROFILE_DIR.is_dir():
        return []
    return sorted(p.stem for p in PROFILE_DIR.glob("*.json"))


SIMULATED_BOARD: int = 0  # reserved "board" of the permanent Simulated SLM; real Blink boards are numbered from 1


def parseBoard(value: Any) -> int | None:
    """Board number from a JSON value: SIMULATED_BOARD or a positive int (bools and other types are rejected)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < SIMULATED_BOARD:
        return None
    return value


def parseSerial(value: Any) -> int | None:
    """SLM serial number from a JSON value: a non-negative int (bools and other types are rejected)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


@dataclass
class Device:
    """One SLM: the Blink board it is on and, once known, its serial number. Board SIMULATED_BOARD is the Simulated SLM."""

    board: int
    serial: int | None = None
    factory_lut: str = ""  # name (no extension) of the factory correction .bmp in devices/; "" = none set

    @property
    def simulated(self) -> bool:
        return self.board == SIMULATED_BOARD

    def toDict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"board": self.board}
        if self.serial is not None:
            d["serial"] = self.serial
        d["factory_lut"] = self.factory_lut
        return d


def loadDevices(path: Path = DEVICES_FILE) -> list[Device]:
    """The Simulated SLM (always first), then the devices in ``devices/devices.json`` sorted by board.

    Invalid entries are dropped and the first entry wins per board. The Simulated SLM is permanent: it is never read
    from or written to the file.
    """
    devices: dict[int, Device] = {SIMULATED_BOARD: Device(SIMULATED_BOARD)}
    doc = _read_json(path) or {}
    entries = doc.get("devices", [])
    if not isinstance(entries, list):
        print(f"Ignoring 'devices' in {path}: expected a list")
        entries = []
    for entry in entries:
        board = parseBoard(entry.get("board")) if isinstance(entry, dict) else None
        if board is None or board == SIMULATED_BOARD:
            print(f"Ignoring invalid device entry in {path}: {entry!r}")
            continue
        if board in devices:
            print(f"Ignoring duplicate entry for board {board} in {path}")
            continue
        serial = parseSerial(entry.get("serial"))
        if "serial" in entry and entry["serial"] is not None and serial is None:
            print(f"Ignoring invalid serial {entry['serial']!r} for board {board} in {path}")
        factory_lut = entry.get("factory_lut", "")
        if not isinstance(factory_lut, str):
            print(f"Ignoring invalid factory_lut {factory_lut!r} for board {board} in {path}")
            factory_lut = ""
        devices[board] = Device(board, serial, factory_lut.strip())
    return [devices[b] for b in sorted(devices)]


def saveDevices(devices: list[Device], path: Path = DEVICES_FILE) -> None:
    """Write the real devices (the Simulated SLM is not stored)."""
    unique = {d.board: d for d in reversed(devices) if not d.simulated}
    _write_json(path, {"devices": [unique[b].toDict() for b in sorted(unique)]})


def findDevice(board: int, path: Path = DEVICES_FILE) -> Device | None:
    return next((d for d in loadDevices(path) if d.board == board), None)


def addDevice(board: int, path: Path = DEVICES_FILE) -> bool:
    """Add ``board`` (serial not known yet) to the device list; return False if it was already there."""
    devices = loadDevices(path)
    if any(d.board == board for d in devices):
        return False
    saveDevices(devices + [Device(board)], path)
    return True


def factoryLutPath(name: str, devices_dir: Path | None = None) -> Path:
    """Path of the factory correction file ``name`` (stored without extension in devices.json)."""
    return (DEVICES_DIR if devices_dir is None else devices_dir) / f"{name}.bmp"


def setDeviceFactoryLut(board: int, name: str, path: Path = DEVICES_FILE) -> bool:
    """Record the factory correction file name for the listed device on ``board``; False if the board is not listed."""
    devices = loadDevices(path)
    for device in devices:
        if device.board == board and not device.simulated:
            device.factory_lut = name
            saveDevices(devices, path)
            return True
    return False


def deviceLabel(device: Device) -> str:
    """Name shown in the GUI: 'Simulated SLM' or 'Board N'."""
    return "Simulated SLM" if device.simulated else f"Board {device.board}"


def setDeviceSerial(board: int, serial: int, path: Path = DEVICES_FILE) -> bool:
    """Record ``serial`` for the listed device on ``board``; return False if that board is not in the list."""
    devices = loadDevices(path)
    for device in devices:
        if device.board == board:
            device.serial = serial
            saveDevices(devices, path)
            return True
    return False
