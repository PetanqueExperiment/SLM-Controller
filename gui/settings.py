"""App settings (``settings/settings.json``) and named profiles (``profiles/<name>.json``)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from gui.defaults import DEFAULT_PROFILE, DEFAULT_STARTUP_MODE, PROFILE_DIR, SETTINGS_FILE
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
