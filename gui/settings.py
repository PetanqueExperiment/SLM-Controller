"""App settings (``settings/settings.json``), named profiles (``profiles/<name>.json``) and the SLM list (``devices/devices.json``)."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from gui.defaults import DEFAULT_PROFILE, DEFAULT_STARTUP_MODE, DEVICES_FILE, PROFILE_DIR, SETTINGS_FILE
from gui.defaults import STARTUP_LAST_USED, STARTUP_MODES


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


_PROFILE_NAME_RE = re.compile(r"[\w\- ]+")


def isValidProfileName(name: str) -> bool:
    """Letters, digits, spaces, '-' and '_' only, with no leading/trailing spaces (the name is used as a file name)."""
    return bool(_PROFILE_NAME_RE.fullmatch(name)) and name == name.strip()


def listProfiles() -> list[str]:
    """Names of the profiles saved in ``profiles/`` (sorted)."""
    if not PROFILE_DIR.is_dir():
        return []
    return sorted(p.stem for p in PROFILE_DIR.glob("*.json"))


def parseBoard(value: Any) -> int | None:
    """Board number from a JSON value: a positive int (bools and other types are rejected)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def parseSerial(value: Any) -> int | None:
    """SLM serial number from a JSON value: a non-negative int (bools and other types are rejected)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


@dataclass
class Device:
    """One SLM in ``devices/devices.json``: the Blink board it is on and, once known, its serial number."""

    board: int
    serial: int | None = None

    def toDict(self) -> dict[str, Any]:
        return {"board": self.board} if self.serial is None else {"board": self.board, "serial": self.serial}


def loadDevices(path: Path = DEVICES_FILE) -> list[Device]:
    """Devices listed in ``devices/devices.json``, sorted by board (invalid entries dropped; first entry wins per board)."""
    doc = _read_json(path) or {}
    entries = doc.get("devices", [])
    if not isinstance(entries, list):
        print(f"Ignoring 'devices' in {path}: expected a list")
        return []
    devices: dict[int, Device] = {}
    for entry in entries:
        board = parseBoard(entry.get("board")) if isinstance(entry, dict) else None
        if board is None:
            print(f"Ignoring invalid device entry in {path}: {entry!r}")
            continue
        if board in devices:
            print(f"Ignoring duplicate entry for board {board} in {path}")
            continue
        serial = parseSerial(entry.get("serial"))
        if "serial" in entry and entry["serial"] is not None and serial is None:
            print(f"Ignoring invalid serial {entry['serial']!r} for board {board} in {path}")
        devices[board] = Device(board, serial)
    return [devices[b] for b in sorted(devices)]


def saveDevices(devices: list[Device], path: Path = DEVICES_FILE) -> None:
    unique = {d.board: d for d in reversed(devices)}
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


def setDeviceSerial(board: int, serial: int, path: Path = DEVICES_FILE) -> bool:
    """Record ``serial`` for the listed device on ``board``; return False if that board is not in the list."""
    devices = loadDevices(path)
    for device in devices:
        if device.board == board:
            device.serial = serial
            saveDevices(devices, path)
            return True
    return False
