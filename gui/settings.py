"""App settings (``settings/settings.json``), named profiles (``profiles/<name>.json``) and the SLM list (``devices/devices.json``)."""

from __future__ import annotations

import copy
import json
import re
import shutil
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from gui.paths import DEFAULT_PROFILE, DEFAULT_STARTUP_MODE, DEVICES_DIR, DEVICES_FILE, PROFILE_DIR, PROFILE_TEMPLATE_FILE, SETTINGS_FILE, SIMULATED_SLM_DEFAULTS_FILE
from gui.paths import STARTUP_FIXED, STARTUP_LAST_USED, STARTUP_MODES


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
    startup_profile: str = DEFAULT_PROFILE

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
        """Name of the profile to load when the app starts: the last used one, or the chosen fixed one."""
        if self.startup_mode == STARTUP_LAST_USED:
            return self.last_profile
        if self.startup_mode == STARTUP_FIXED:
            return self.startup_profile
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
    """The default profile template in the repository (defaults/profiles/default.json) is missing or invalid."""


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


OLD_GRATING_SECTION: str = "grating"
ELEMENTARY_SECTION: str = "elementary"
ELEMENTARY_KEYS: tuple[str, ...] = ("phase_levels", "wavelength_nm")


def migrateProfile(profile: dict[str, Any]) -> dict[str, Any]:
    """Profile in the current layout. Old profiles have a top-level "grating" {x_um, y_um, phase_levels,
    wavelength_nm}: phase_levels and wavelength_nm move to "elementary", x_um / y_um to "elementary" ▸ "grating".

    Values already in the new place are kept. ``profile`` itself is not changed.
    """
    if OLD_GRATING_SECTION not in profile:
        return profile
    migrated = copy.deepcopy(profile)
    old = migrated.pop(OLD_GRATING_SECTION)
    if not isinstance(old, dict):
        print(f"Profile: ignoring old {OLD_GRATING_SECTION!r} = {old!r} (expected a section)")
        return migrated
    elementary = migrated.get(ELEMENTARY_SECTION)
    elementary = dict(elementary) if isinstance(elementary, dict) else {}
    grating = elementary.get(OLD_GRATING_SECTION)
    grating = dict(grating) if isinstance(grating, dict) else {}
    for key, value in old.items():
        target = elementary if key in ELEMENTARY_KEYS else grating
        target.setdefault(key, value)
    elementary[OLD_GRATING_SECTION] = grating
    migrated[ELEMENTARY_SECTION] = elementary
    return migrated


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
        return mergeProfile(template, migrateProfile(profile)), PROFILE_LOADED
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
MIN_SLM_PIXEL_PITCH_UM: float = 1.0  # plausible range of a reported pixel pitch (Get_pixel_pitch has no documented unit)
MAX_SLM_PIXEL_PITCH_UM: float = 100.0
MAX_SLM_RADIUS_M: float = 0.1
SIMULATED_SLM_KEY: str = "simulated_slm"
WFC_KEY: str = "wfc"  # devices.json: name of the device's WFC .bmp
OLD_WFC_KEY: str = "factory_lut"  # earlier name of WFC_KEY, still read


def parseSlmRadiusM(value: Any) -> float | None:
    """SLM radius [m] from a JSON value: a number in (0, MAX_SLM_RADIUS_M]."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if 0 < value <= MAX_SLM_RADIUS_M else None


def parsePixelPitchUm(value: Any) -> float | None:
    """Pixel pitch [µm] from a JSON value: a number within MIN_SLM_PIXEL_PITCH_UM … MAX_SLM_PIXEL_PITCH_UM."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if MIN_SLM_PIXEL_PITCH_UM <= value <= MAX_SLM_PIXEL_PITCH_UM else None


def parseImageSize(value: Any) -> int | None:
    """Image width or height from a JSON value: a positive int (bools and other types are rejected)."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


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


class DeviceDefaultsError(RuntimeError):
    """The Simulated SLM defaults in the repository (defaults/devices/simulated_slm.json) are missing or invalid."""


@dataclass(frozen=True)
class DeviceGeometry:
    """Image size [px], pixel pitch [µm] and SLM radius [m] (Zernike unit disk) of an SLM."""

    width_px: int
    height_px: int
    pixel_pitch_um: float
    slm_radius_m: float


_device_defaults_cache: dict[Path, DeviceGeometry] = {}


def loadDeviceDefaults(path: Path = SIMULATED_SLM_DEFAULTS_FILE) -> DeviceGeometry:
    """Default geometry of the Simulated SLM, also used by a real SLM until it has reported its own (read once)."""
    if path in _device_defaults_cache:
        return _device_defaults_cache[path]
    doc = _read_json(path)
    if doc is None:
        raise DeviceDefaultsError(f"Simulated SLM defaults missing or invalid: {path}")
    values = (parseImageSize(doc.get("width_px")), parseImageSize(doc.get("height_px")), parsePixelPitchUm(doc.get("pixel_pitch_um")), parseSlmRadiusM(doc.get("slm_radius_m")))
    if any(v is None for v in values):
        raise DeviceDefaultsError(f"Simulated SLM defaults need width_px, height_px (positive integers), pixel_pitch_um ({MIN_SLM_PIXEL_PITCH_UM:g} … {MAX_SLM_PIXEL_PITCH_UM:g}) and slm_radius_m (0 … {MAX_SLM_RADIUS_M:g}): {path}")
    defaults = DeviceGeometry(*values)
    _device_defaults_cache[path] = defaults
    return defaults


@dataclass
class Device:
    """One SLM: the Blink board it is on and, once known, its serial number. Board SIMULATED_BOARD is the Simulated SLM."""

    board: int
    serial: int | None = None
    factory_lut: str = ""  # name (no extension) of the factory correction (WFC) .bmp in devices/; "" = none set (JSON key "wfc")
    lut: str = ""
    width_px: int | None = None  # image size reported by the SLM at its last connect (Simulated SLM: set by hand)
    height_px: int | None = None
    pixel_pitch_um: float | None = None  # pixel pitch reported by the SLM at its last connect (Simulated SLM: set by hand)
    slm_radius_m: float | None = None  # radius of the Zernike unit disk on the SLM, centred on the SLM (set by hand)
    name: str | None = None  # set to defaultDeviceName(serial) at the first connect; editable

    @property
    def simulated(self) -> bool:
        return self.board == SIMULATED_BOARD

    def imageShape(self) -> tuple[int, int]:
        """(height, width) [px]: the stored size, or the default size if not known yet."""
        if self.width_px is None or self.height_px is None:
            defaults = loadDeviceDefaults()
            return defaults.height_px, defaults.width_px
        return self.height_px, self.width_px

    def pixelPitchUm(self) -> float:
        """Pixel pitch [µm]: the stored one, or the default if not known yet."""
        return loadDeviceDefaults().pixel_pitch_um if self.pixel_pitch_um is None else self.pixel_pitch_um

    def slmRadiusM(self) -> float:
        """SLM radius [m]: the stored one, or the default if not set."""
        return loadDeviceDefaults().slm_radius_m if self.slm_radius_m is None else self.slm_radius_m

    def toDict(self) -> dict[str, Any]:
        d: dict[str, Any] = {} if self.name is None else {"name": self.name}
        d["board"] = self.board
        if self.serial is not None:
            d["serial"] = self.serial
        d[WFC_KEY] = self.factory_lut
        d["lut"] = self.lut
        if self.width_px is not None and self.height_px is not None:
            d["width_px"] = self.width_px
            d["height_px"] = self.height_px
        if self.pixel_pitch_um is not None:
            d["pixel_pitch_um"] = self.pixel_pitch_um
        if self.slm_radius_m is not None:
            d["slm_radius_m"] = self.slm_radius_m
        return d


def _readImageSize(entry: dict[str, Any], what: str, path: Path) -> tuple[int | None, int | None]:
    width, height = parseImageSize(entry.get("width_px")), parseImageSize(entry.get("height_px"))
    if ("width_px" in entry or "height_px" in entry) and (width is None or height is None):
        print(f"Ignoring invalid image size {entry.get('width_px')!r} x {entry.get('height_px')!r} for {what} in {path}")
        return None, None
    return width, height


def _readPixelPitch(entry: dict[str, Any], what: str, path: Path) -> float | None:
    pitch_um = parsePixelPitchUm(entry.get("pixel_pitch_um"))
    if "pixel_pitch_um" in entry and pitch_um is None:
        print(f"Ignoring invalid pixel pitch {entry.get('pixel_pitch_um')!r} µm for {what} in {path}")
    return pitch_um


def _readSlmRadius(entry: dict[str, Any], what: str, path: Path) -> float | None:
    radius_m = parseSlmRadiusM(entry.get("slm_radius_m"))
    if "slm_radius_m" in entry and radius_m is None:
        print(f"Ignoring invalid SLM radius {entry.get('slm_radius_m')!r} m for {what} in {path}")
    return radius_m


def loadDevices(path: Path = DEVICES_FILE) -> list[Device]:
    """The Simulated SLM (always first), then the devices in ``devices/devices.json`` sorted by board.

    Invalid entries are dropped and the first entry wins per board. The Simulated SLM is permanent; only its image
    size is stored in the file (``simulated_slm``).
    """
    doc = _read_json(path) or {}
    simulated = Device(SIMULATED_BOARD)
    entry = doc.get(SIMULATED_SLM_KEY)
    if isinstance(entry, dict):
        simulated.width_px, simulated.height_px = _readImageSize(entry, "the Simulated SLM", path)
        simulated.pixel_pitch_um = _readPixelPitch(entry, "the Simulated SLM", path)
        simulated.slm_radius_m = _readSlmRadius(entry, "the Simulated SLM", path)
    devices: dict[int, Device] = {SIMULATED_BOARD: simulated}
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
        factory_lut = entry.get(WFC_KEY, entry.get(OLD_WFC_KEY, ""))
        if not isinstance(factory_lut, str):
            print(f"Ignoring invalid wfc {factory_lut!r} for board {board} in {path}")
            factory_lut = ""
        lut = entry.get("lut", "")
        if not isinstance(lut, str):
            print(f"Ignoring invalid lut {lut!r} for board {board} in {path}")
            lut = ""
        width, height = _readImageSize(entry, f"board {board}", path)
        name = entry.get("name")
        if name is not None and not isValidDeviceName(name.strip() if isinstance(name, str) else name):
            print(f"Ignoring invalid name {name!r} for board {board} in {path}")
            name = None
        devices[board] = Device(board, serial, factory_lut.strip(), lut.strip(), width, height, _readPixelPitch(entry, f"board {board}", path), _readSlmRadius(entry, f"board {board}", path), None if name is None else name.strip())
    return [devices[b] for b in sorted(devices)]


def saveDevices(devices: list[Device], path: Path = DEVICES_FILE) -> None:
    """Write the real devices, and the image size of the Simulated SLM if it is set."""
    unique = {d.board: d for d in reversed(devices) if not d.simulated}
    doc: dict[str, Any] = {"devices": [unique[b].toDict() for b in sorted(unique)]}
    simulated = next((d for d in devices if d.simulated), None)
    if simulated is not None:
        entry = {key: value for key, value in simulated.toDict().items() if key in ("width_px", "height_px", "pixel_pitch_um", "slm_radius_m")}
        if entry:
            doc[SIMULATED_SLM_KEY] = entry
    _write_json(path, doc)


def findDevice(board: int, path: Path = DEVICES_FILE) -> Device | None:
    return next((d for d in loadDevices(path) if d.board == board), None)


def addDevice(board: int, path: Path = DEVICES_FILE) -> bool:
    """Add ``board`` (serial not known yet) to the device list; return False if it was already there."""
    devices = loadDevices(path)
    if any(d.board == board for d in devices):
        return False
    saveDevices(devices + [Device(board)], path)
    return True


HOLOGRAMS_DIR_NAME: str = "holograms"
SIMULATED_DEVICE_DIR_NAME: str = "simulated_slm"


def deviceDirName(device: Device) -> str:
    """Folder of ``device`` under devices/: its name, 'board<N>' until it has one, 'simulated_slm' for the Simulated SLM."""
    if device.simulated:
        return SIMULATED_DEVICE_DIR_NAME
    return device.name or f"board{device.board}"


def deviceDir(device: Device, devices_dir: Path | None = None) -> Path:
    """devices/<deviceDirName>: holds the device's LUT, WFC and holograms/ folder."""
    return (DEVICES_DIR if devices_dir is None else devices_dir) / deviceDirName(device)


def _deviceFile(device: Device, file_name: str, devices_dir: Path | None) -> Path:
    in_folder = deviceDir(device, devices_dir) / file_name
    old_place = (DEVICES_DIR if devices_dir is None else devices_dir) / file_name
    return old_place if not in_folder.is_file() and old_place.is_file() else in_folder


def deviceWfcPath(device: Device, devices_dir: Path | None = None) -> Path:
    """The device's WFC .bmp: in its folder (or, until copied there, directly in devices/ as before)."""
    return _deviceFile(device, f"{device.factory_lut}.bmp", devices_dir)


def deviceLutPath(device: Device, devices_dir: Path | None = None) -> Path:
    """The device's LUT file: in its folder (or, until copied there, directly in devices/ as before)."""
    return _deviceFile(device, f"{device.lut}.LUT", devices_dir)


def prepareDeviceDir(device: Device, devices_dir: Path | None = None) -> list[str]:
    """Create the device folder and its holograms/ folder; copy a WFC / LUT still stored directly in devices/ into it.

    The old copies are left in place (another device may use them). Returns the names of the copied files.
    """
    root = DEVICES_DIR if devices_dir is None else devices_dir
    folder = deviceDir(device, root)
    (folder / HOLOGRAMS_DIR_NAME).mkdir(parents=True, exist_ok=True)
    copied = []
    for stem, suffix in ((device.factory_lut, ".bmp"), (device.lut, ".LUT")):
        if not stem:
            continue
        old_place, target = root / f"{stem}{suffix}", folder / f"{stem}{suffix}"
        if old_place.is_file() and not target.exists():
            shutil.copy2(old_place, target)
            copied.append(target.name)
    return copied


def factoryLutPath(name: str, devices_dir: Path | None = None) -> Path:
    """Old location of the factory correction file ``name`` (directly in devices/; read until copied to the device folder)."""
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


def lutPath(name: str, devices_dir: Path | None = None) -> Path:
    """Old location of the controller LUT file ``name`` (directly in devices/; read until copied to the device folder)."""
    return (DEVICES_DIR if devices_dir is None else devices_dir) / f"{name}.LUT"


def setDeviceLut(board: int, name: str, path: Path = DEVICES_FILE) -> bool:
    """Record the LUT file name for the listed device on ``board``; False if the board is not listed."""
    devices = loadDevices(path)
    for device in devices:
        if device.board == board and not device.simulated:
            device.lut = name
            saveDevices(devices, path)
            return True
    return False


def deviceLabel(device: Device) -> str:
    """Name shown in the GUI: 'Simulated SLM' or 'Board N'."""
    return "Simulated SLM" if device.simulated else f"Board {device.board}"


def deviceImageShape(board: int | None, path: Path = DEVICES_FILE) -> tuple[int, int]:
    """(height, width) [px] stored for ``board`` (the default size if unknown, not listed, or no board)."""
    device = None if board is None else findDevice(board, path)
    return Device(SIMULATED_BOARD).imageShape() if device is None else device.imageShape()


def setDeviceImageSize(board: int, width_px: int, height_px: int, path: Path = DEVICES_FILE) -> bool:
    """Record the image size of the device on ``board`` (also the Simulated SLM); False if the board is not listed."""
    if parseImageSize(width_px) is None or parseImageSize(height_px) is None:
        raise ValueError(f"image size must be positive integers, got {width_px!r} x {height_px!r}")
    devices = loadDevices(path)
    for device in devices:
        if device.board == board:
            device.width_px, device.height_px = width_px, height_px
            saveDevices(devices, path)
            return True
    return False


def devicePixelPitchUm(board: int | None, path: Path = DEVICES_FILE) -> float:
    """Pixel pitch [µm] stored for ``board`` (the default if unknown, not listed, or no board)."""
    device = None if board is None else findDevice(board, path)
    return Device(SIMULATED_BOARD).pixelPitchUm() if device is None else device.pixelPitchUm()


def setDevicePixelPitchUm(board: int, pitch_um: float, path: Path = DEVICES_FILE) -> bool:
    """Record the pixel pitch [µm] of the device on ``board`` (also the Simulated SLM); False if the board is not listed."""
    value = parsePixelPitchUm(pitch_um)
    if value is None:
        raise ValueError(f"pixel pitch must be {MIN_SLM_PIXEL_PITCH_UM:g} … {MAX_SLM_PIXEL_PITCH_UM:g} µm, got {pitch_um!r}")
    devices = loadDevices(path)
    for device in devices:
        if device.board == board:
            device.pixel_pitch_um = value
            saveDevices(devices, path)
            return True
    return False


def deviceSlmRadiusM(board: int | None, path: Path = DEVICES_FILE) -> float:
    """SLM radius [m] stored for ``board`` (the default if not set, not listed, or no board)."""
    device = None if board is None else findDevice(board, path)
    return Device(SIMULATED_BOARD).slmRadiusM() if device is None else device.slmRadiusM()


def setDeviceSlmRadiusM(board: int, radius_m: float, path: Path = DEVICES_FILE) -> bool:
    """Record the SLM radius [m] of the device on ``board`` (also the Simulated SLM); False if the board is not listed."""
    value = parseSlmRadiusM(radius_m)
    if value is None:
        raise ValueError(f"SLM radius must be in (0, {MAX_SLM_RADIUS_M:g}] m, got {radius_m!r}")
    devices = loadDevices(path)
    for device in devices:
        if device.board == board:
            device.slm_radius_m = value
            saveDevices(devices, path)
            return True
    return False


def defaultDeviceName(serial: int) -> str:
    """Name given to a device at its first connect: 'slm' followed by its serial number."""
    return f"slm{serial}"


def isValidDeviceName(name: str) -> bool:
    """Letters, digits, spaces, '-' and '_' only, no leading/trailing spaces: the name is also the device folder name."""
    return isinstance(name, str) and isValidProfileName(name) and name.lower() != SIMULATED_DEVICE_DIR_NAME


def setDeviceName(board: int, name: str, path: Path = DEVICES_FILE, devices_dir: Path | None = None) -> bool:
    """Record ``name`` for the listed real device on ``board`` and rename its folder; False if not listed.

    Raises ValueError if the name is not a valid folder name or another device has it (also when its folder exists).
    """
    name = name.strip() if isinstance(name, str) else name
    if not isValidDeviceName(name):
        raise ValueError(f"device name {name!r} must be letters, digits, spaces, '-' or '_' (it is also the folder name)")
    devices = loadDevices(path)
    device = next((d for d in devices if d.board == board and not d.simulated), None)
    if device is None:
        return False
    if any(d is not device and not d.simulated and deviceDirName(d).lower() == name.lower() for d in devices):
        raise ValueError(f"another device is already named {name!r}")
    root = path.parent if devices_dir is None else devices_dir
    old_folder, new_folder = deviceDir(device, root), root / name
    renamed = False
    if old_folder.is_dir() and old_folder.name != name:
        if new_folder.exists() and new_folder.resolve() != old_folder.resolve():
            raise ValueError(f"a folder named {name!r} already exists in {root}")
        old_folder.rename(new_folder)
        renamed = True
    device.name = name
    try:
        saveDevices(devices, path)
    except OSError:
        if renamed:
            new_folder.rename(old_folder)
        raise
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
