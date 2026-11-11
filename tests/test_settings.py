"""Tests for gui/settings.py: device list, board parsing, profiles and app settings (no Qt, no hardware)."""

import json

import pytest

from gui import settings
from gui.settings import AppSettings, Device, addDevice, findDevice, loadDevices, parseBoard, saveDevices, setDeviceSerial


@pytest.mark.parametrize("value, expected", [(1, 1), (3, 3), (0, None), (-2, None), (True, None), (1.0, None), ("1", None), (None, None)])
def test_parse_board(value, expected):
    assert parseBoard(value) == expected


def test_load_devices_missing_file_is_empty(tmp_path):
    assert loadDevices(tmp_path / "devices.json") == []


def test_add_device_creates_file_and_rejects_duplicates(tmp_path):
    path = tmp_path / "devices" / "devices.json"
    assert addDevice(2, path) is True
    assert addDevice(1, path) is True
    assert addDevice(2, path) is False
    assert loadDevices(path) == [Device(1), Device(2)]
    assert json.loads(path.read_text(encoding="utf-8")) == {"devices": [{"board": 1}, {"board": 2}]}


def test_set_device_serial_and_find(tmp_path):
    path = tmp_path / "devices.json"
    addDevice(1, path)
    assert setDeviceSerial(1, 6801, path) is True
    assert setDeviceSerial(5, 1234, path) is False
    assert findDevice(1, path) == Device(1, 6801)
    assert findDevice(5, path) is None
    assert json.loads(path.read_text(encoding="utf-8")) == {"devices": [{"board": 1, "serial": 6801}]}


def test_save_devices_sorts_and_deduplicates(tmp_path):
    path = tmp_path / "devices.json"
    saveDevices([Device(3), Device(1, 7), Device(3, 9)], path)
    assert loadDevices(path) == [Device(1, 7), Device(3)]


def test_load_devices_old_format_without_serial(tmp_path):
    path = tmp_path / "devices.json"
    path.write_text(json.dumps({"devices": [{"board": 1}]}), encoding="utf-8")
    assert loadDevices(path) == [Device(1, None)]


def test_load_devices_skips_invalid_entries(tmp_path):
    path = tmp_path / "devices.json"
    entries = [{"board": 1, "serial": 6801}, {"board": "x"}, {"board": 0}, 5, {}, {"board": 4, "serial": -1}, {"board": 1, "serial": 2}, {"board": 6, "serial": True}]
    path.write_text(json.dumps({"devices": entries}), encoding="utf-8")
    assert loadDevices(path) == [Device(1, 6801), Device(4, None), Device(6, None)]


@pytest.mark.parametrize("content", ["not json", "[1, 2]", '{"devices": {"board": 1}}'])
def test_load_devices_bad_file_is_empty(tmp_path, content):
    path = tmp_path / "devices.json"
    path.write_text(content, encoding="utf-8")
    assert loadDevices(path) == []


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
