"""Tests for gui/settings.py: device list, board parsing, profiles and app settings (no Qt, no hardware)."""

import json

import pytest

from gui import settings
from gui.settings import AppSettings, Device, addDevice, findDevice, loadDevices, parseBoard, saveDevices, setDeviceSerial


@pytest.mark.parametrize("value, expected", [(1, 1), (3, 3), (0, 0), (-2, None), (True, None), (1.0, None), ("1", None), (None, None)])
def test_parse_board(value, expected):
    assert parseBoard(value) == expected


SIM = Device(settings.SIMULATED_BOARD)


def test_load_devices_missing_file_has_only_the_simulated_slm(tmp_path):
    assert loadDevices(tmp_path / "devices.json") == [SIM]
    assert SIM.simulated and settings.deviceLabel(SIM) == "Simulated SLM" and settings.deviceLabel(Device(2)) == "Board 2"


def test_add_device_creates_file_and_rejects_duplicates(tmp_path):
    path = tmp_path / "devices" / "devices.json"
    assert addDevice(2, path) is True
    assert addDevice(1, path) is True
    assert addDevice(2, path) is False
    assert loadDevices(path) == [SIM, Device(1), Device(2)]
    assert json.loads(path.read_text(encoding="utf-8")) == {"devices": [{"board": 1, "factory_lut": ""}, {"board": 2, "factory_lut": ""}]}


def test_set_device_serial_and_find(tmp_path):
    path = tmp_path / "devices.json"
    addDevice(1, path)
    assert setDeviceSerial(1, 6801, path) is True
    assert setDeviceSerial(5, 1234, path) is False
    assert findDevice(1, path) == Device(1, 6801)
    assert findDevice(5, path) is None
    assert json.loads(path.read_text(encoding="utf-8")) == {"devices": [{"board": 1, "serial": 6801, "factory_lut": ""}]}


def test_save_devices_sorts_and_deduplicates(tmp_path):
    path = tmp_path / "devices.json"
    saveDevices([Device(3), Device(1, 7), Device(3, 9)], path)
    assert loadDevices(path) == [SIM, Device(1, 7), Device(3)]


def test_load_devices_old_format_without_serial(tmp_path):
    path = tmp_path / "devices.json"
    path.write_text(json.dumps({"devices": [{"board": 1}]}), encoding="utf-8")
    assert loadDevices(path) == [SIM, Device(1, None)]


def test_load_devices_skips_invalid_entries(tmp_path):
    path = tmp_path / "devices.json"
    entries = [{"board": 1, "serial": 6801}, {"board": "x"}, {"board": 0, "serial": 9}, {"board": -1}, 5, {}, {"board": 4, "serial": -1}, {"board": 1, "serial": 2}, {"board": 6, "serial": True}]
    path.write_text(json.dumps({"devices": entries}), encoding="utf-8")
    assert loadDevices(path) == [SIM, Device(1, 6801), Device(4, None), Device(6, None)]


@pytest.mark.parametrize("content", ["not json", "[1, 2]", '{"devices": {"board": 1}}'])
def test_load_devices_bad_file_is_empty(tmp_path, content):
    path = tmp_path / "devices.json"
    path.write_text(content, encoding="utf-8")
    assert loadDevices(path) == [SIM]


def test_simulated_slm_is_permanent_and_never_written(tmp_path):
    path = tmp_path / "devices.json"
    assert addDevice(settings.SIMULATED_BOARD, path) is False
    saveDevices([SIM, Device(2)], path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"devices": [{"board": 2, "factory_lut": ""}]}
    assert findDevice(settings.SIMULATED_BOARD, path) == SIM


@pytest.mark.parametrize("value, expected", [(0, 0), (6801, 6801), (-1, None), (True, None), ("6801", None), (None, None), (1.5, None)])
def test_parse_serial(value, expected):
    assert settings.parseSerial(value) == expected


def test_profile_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "PROFILE_DIR", tmp_path / "profiles")
    assert settings.readProfile("default") is None
    path = settings.writeProfile("default", {"board": 2})
    assert path == tmp_path / "profiles" / "default.json"
    assert settings.readProfile("default") == {"board": 2}


@pytest.mark.parametrize("name, ok", [("default", True), ("exp A-2_b", True), ("", False), (" lead", False), ("trail ", False), ("a/b", False), ("..", False), ("x.json", False), ("a\\b", False)])
def test_is_valid_profile_name(name, ok):
    assert settings.isValidProfileName(name) is ok


def test_list_profiles(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "PROFILE_DIR", tmp_path / "profiles")
    assert settings.listProfiles() == []
    settings.writeProfile("zeta", {})
    settings.writeProfile("alpha", {"board": 1})
    (tmp_path / "profiles" / "notes.txt").write_text("ignored", encoding="utf-8")
    assert settings.listProfiles() == ["alpha", "zeta"]


def test_app_settings_defaults_and_round_trip(tmp_path):
    path = tmp_path / "settings" / "settings.json"
    s = AppSettings.load(path)
    assert (s.startup_mode, s.last_profile) == ("last_used", "default")
    s.last_profile = "experiment_A"
    s.save(path)
    assert AppSettings.load(path).startupProfile() == "experiment_A"


def test_app_settings_unknown_mode_falls_back(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"startup_mode": "bogus", "last_profile": "p"}), encoding="utf-8")
    s = AppSettings.load(path)
    assert s.startup_mode == "last_used"
    assert s.last_profile == "p"


ABERRATION_LABELS = ["defocus", "asti 0", "asti 45", "coma 0", "coma 90", "primary spherical", "trefoil 0", "trefoil 90", "asti 2nd 0", "asti 2nd 45", "coma 2nd 0", "coma 2nd 90", "2nd spherical", "trefoil 2nd 0", "trefoil 2nd 90"]


def test_repository_template_is_complete():
    t = settings.loadProfileTemplate()
    assert set(t) == {"grating", "corrections", "pupil", "aberration"}
    assert set(t["grating"]) == {"x_um", "y_um", "phase_levels", "wavelength_nm"}
    assert set(t["corrections"]) == {"manufacturer", "aberration"} and all(isinstance(v, bool) for v in t["corrections"].values())
    assert set(t["pupil"]) == {"enabled", "radius_mm", "x0_mm", "y0_mm", "move_out_um"} and isinstance(t["pupil"]["radius_mm"], str)
    assert list(t["aberration"]) == ABERRATION_LABELS
    assert "board" not in t


def test_merge_profile_overrides_keeps_extra_and_rejects_wrong_types():
    template = {"grating": {"x_um": 0, "wavelength_nm": 850}, "corrections": {"manufacturer": True}, "pupil": {"radius_mm": "1"}}
    profile = {"board": 2, "grating": {"x_um": 3.5, "wavelength_nm": "fast"}, "corrections": {"manufacturer": 1}, "pupil": "oops"}
    merged = settings.mergeProfile(template, profile)
    assert merged == {"board": 2, "grating": {"x_um": 3.5, "wavelength_nm": 850}, "corrections": {"manufacturer": True}, "pupil": {"radius_mm": "1"}}
    assert template["grating"]["x_um"] == 0


def test_load_profile_params_old_file_gets_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "PROFILE_DIR", tmp_path / "profiles")
    settings.writeProfile("default", {"board": 1})
    params, state = settings.loadProfileParams("default")
    assert state == settings.PROFILE_LOADED and params["board"] == 1 and params["pupil"]["move_out_um"] == 40
    params, state = settings.loadProfileParams("missing")
    assert state == settings.PROFILE_MISSING and "board" not in params and params["grating"]["phase_levels"] == 255
    settings.profilePath("damaged").write_text("{ broken", encoding="utf-8")
    params, state = settings.loadProfileParams("damaged")
    assert state == settings.PROFILE_UNREADABLE and params == settings.loadProfileTemplate()


def test_missing_template_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "PROFILE_TEMPLATE_FILE", tmp_path / "nope.json")
    with pytest.raises(settings.ProfileTemplateError):
        settings.loadProfileParams("default")


def test_restore_default_profile_copies_template(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "PROFILE_DIR", tmp_path / "profiles")
    settings.writeProfile("default", {"grating": {"x_um": 99}, "broken": True})
    path = settings.restoreDefaultProfile()
    assert path == tmp_path / "profiles" / "default.json"
    assert json.loads(path.read_text(encoding="utf-8")) == settings.loadProfileTemplate()


def test_factory_lut_field(tmp_path):
    path = tmp_path / "devices.json"
    path.write_text(json.dumps({"devices": [{"board": 1}, {"board": 2, "factory_lut": " slm6801 "}, {"board": 3, "factory_lut": 5}]}), encoding="utf-8")
    assert [d.factory_lut for d in loadDevices(path)] == ["", "", "slm6801", ""]
    assert settings.setDeviceFactoryLut(1, "head_A", path) is True
    assert settings.setDeviceFactoryLut(settings.SIMULATED_BOARD, "x", path) is False
    assert settings.setDeviceFactoryLut(9, "x", path) is False
    assert findDevice(1, path).factory_lut == "head_A"
    assert json.loads(path.read_text(encoding="utf-8"))["devices"][0] == {"board": 1, "factory_lut": "head_A"}
    assert settings.factoryLutPath("head_A", tmp_path) == tmp_path / "head_A.bmp"
