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
    assert json.loads(path.read_text(encoding="utf-8")) == {"devices": [{"board": 1, "wfc": "", "lut": ""}, {"board": 2, "wfc": "", "lut": ""}]}


def test_set_device_serial_and_find(tmp_path):
    path = tmp_path / "devices.json"
    addDevice(1, path)
    assert setDeviceSerial(1, 6801, path) is True
    assert setDeviceSerial(5, 1234, path) is False
    assert findDevice(1, path) == Device(1, 6801)
    assert findDevice(5, path) is None
    assert json.loads(path.read_text(encoding="utf-8")) == {"devices": [{"board": 1, "serial": 6801, "wfc": "", "lut": ""}]}


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
    assert json.loads(path.read_text(encoding="utf-8")) == {"devices": [{"board": 2, "wfc": "", "lut": ""}]}
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
    assert set(t) == {"elementary", "corrections", "pupil", "aberration", "operation_mode"}
    assert set(t["elementary"]) == {"phase_levels", "wavelength_nm", "grating"} and set(t["elementary"]["grating"]) == {"x_um", "y_um"}
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
    assert state == settings.PROFILE_MISSING and "board" not in params and params["elementary"]["phase_levels"] == 255
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
    assert json.loads(path.read_text(encoding="utf-8"))["devices"][0] == {"board": 1, "wfc": "head_A", "lut": ""}
    assert settings.factoryLutPath("head_A", tmp_path) == tmp_path / "head_A.bmp"


def test_lut_field(tmp_path):
    path = tmp_path / "devices.json"
    path.write_text(json.dumps({"devices": [{"board": 1, "factory_lut": "wfc_A"}, {"board": 2, "lut": " slm6801_at850 "}, {"board": 3, "lut": 7}]}), encoding="utf-8")
    assert [d.lut for d in loadDevices(path)] == ["", "", "slm6801_at850", ""]
    assert settings.setDeviceLut(1, "head_A_lut", path) is True
    assert settings.setDeviceLut(settings.SIMULATED_BOARD, "x", path) is False
    assert settings.setDeviceLut(9, "x", path) is False
    assert findDevice(1, path).lut == "head_A_lut" and findDevice(1, path).factory_lut == "wfc_A"
    assert json.loads(path.read_text(encoding="utf-8"))["devices"][0] == {"board": 1, "wfc": "wfc_A", "lut": "head_A_lut"}
    assert settings.lutPath("head_A_lut", tmp_path) == tmp_path / "head_A_lut.LUT"


def test_template_has_operation_mode_sections():
    t = settings.loadProfileTemplate()
    assert t["operation_mode"] == {"mode": "Static", "static": {"last_hologram": "none"}, "dynamic": {"trigger_timeout_ms": 1000, "holograms": []}}


def test_old_profile_without_operation_mode_gets_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "PROFILE_DIR", tmp_path)
    (tmp_path / "old.json").write_text(json.dumps({"elementary": {"grating": {"x_um": 5}}}), encoding="utf-8")
    params, state = settings.loadProfileParams("old")
    assert state == settings.PROFILE_LOADED and params["elementary"]["grating"]["x_um"] == 5
    assert params["operation_mode"] == settings.loadProfileTemplate()["operation_mode"]
    (tmp_path / "new.json").write_text(json.dumps({"operation_mode": {"static": {"last_hologram": "C:/h.pkl"}, "dynamic": {"future": 1}}}), encoding="utf-8")
    params, _ = settings.loadProfileParams("new")
    assert params["operation_mode"] == {"mode": "Static", "static": {"last_hologram": "C:/h.pkl"}, "dynamic": {"trigger_timeout_ms": 1000, "holograms": [], "future": 1}}
    (tmp_path / "timeout.json").write_text(json.dumps({"operation_mode": {"dynamic": {"trigger_timeout_ms": 250}}}), encoding="utf-8")
    params, _ = settings.loadProfileParams("timeout")
    assert params["operation_mode"]["dynamic"] == {"trigger_timeout_ms": 250, "holograms": []}
    (tmp_path / "frames.json").write_text(json.dumps({"operation_mode": {"mode": "Dynamic", "dynamic": {"holograms": ["C:/a.pkl", "none"]}}}), encoding="utf-8")
    params, _ = settings.loadProfileParams("frames")
    assert params["operation_mode"]["mode"] == "Dynamic" and params["operation_mode"]["dynamic"] == {"trigger_timeout_ms": 1000, "holograms": ["C:/a.pkl", "none"]}


def test_fixed_startup_profile(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"startup_mode": "last_used", "last_profile": "main"}), encoding="utf-8")
    s = AppSettings.load(path)
    assert s.startup_profile == "default" and s.startupProfile() == "main"
    s.startup_mode, s.startup_profile = "fixed", "calibration"
    s.last_profile = "experiment_B"
    s.save(path)
    loaded = AppSettings.load(path)
    assert (loaded.startup_mode, loaded.startup_profile, loaded.last_profile) == ("fixed", "calibration", "experiment_B")
    assert loaded.startupProfile() == "calibration"
    assert json.loads(path.read_text(encoding="utf-8")) == {"startup_mode": "fixed", "last_profile": "experiment_B", "startup_profile": "calibration"}


def test_old_grating_section_is_migrated_to_elementary(tmp_path, monkeypatch):
    old = {"board": 1, "grating": {"x_um": 2.5, "y_um": -1, "phase_levels": 230, "wavelength_nm": 785}, "pupil": {"enabled": True}}
    migrated = settings.migrateProfile(old)
    assert migrated == {"board": 1, "pupil": {"enabled": True}, "elementary": {"phase_levels": 230, "wavelength_nm": 785, "grating": {"x_um": 2.5, "y_um": -1}}}
    assert "grating" in old and settings.migrateProfile(migrated) == migrated
    monkeypatch.setattr(settings, "PROFILE_DIR", tmp_path)
    (tmp_path / "old.json").write_text(json.dumps(old), encoding="utf-8")
    params, state = settings.loadProfileParams("old")
    assert state == settings.PROFILE_LOADED and "grating" not in params
    assert params["elementary"] == {"phase_levels": 230, "wavelength_nm": 785, "grating": {"x_um": 2.5, "y_um": -1}}


def test_migration_keeps_values_already_in_the_new_place():
    both = {"grating": {"x_um": 1, "phase_levels": 200}, "elementary": {"phase_levels": 255, "grating": {"y_um": 3}}}
    assert settings.migrateProfile(both) == {"elementary": {"phase_levels": 255, "grating": {"y_um": 3, "x_um": 1}}}


def test_device_image_size_round_trip(tmp_path):
    path = tmp_path / "devices.json"
    assert settings.addDevice(1, path)
    assert settings.deviceImageShape(1, path) == (1024, 1024)
    assert settings.setDeviceImageSize(1, 1920, 1152, path)
    assert settings.deviceImageShape(1, path) == (1152, 1920)
    assert json.loads(path.read_text(encoding="utf-8"))["devices"][0] == {"board": 1, "wfc": "", "lut": "", "width_px": 1920, "height_px": 1152}
    assert not settings.setDeviceImageSize(7, 10, 10, path)
    assert settings.deviceImageShape(7, path) == settings.deviceImageShape(None, path) == (1024, 1024)
    with pytest.raises(ValueError):
        settings.setDeviceImageSize(1, 0, 10, path)


def test_simulated_slm_size_is_stored_without_a_device_entry(tmp_path):
    path = tmp_path / "devices.json"
    settings.addDevice(1, path)
    assert settings.deviceImageShape(settings.SIMULATED_BOARD, path) == (1024, 1024)
    assert settings.setDeviceImageSize(settings.SIMULATED_BOARD, 800, 600, path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["simulated_slm"] == {"width_px": 800, "height_px": 600} and [d["board"] for d in doc["devices"]] == [1]
    assert settings.deviceImageShape(settings.SIMULATED_BOARD, path) == (600, 800)
    settings.setDeviceLut(1, "x", path)
    assert json.loads(path.read_text(encoding="utf-8"))["simulated_slm"] == {"width_px": 800, "height_px": 600}


def test_old_device_list_without_sizes_and_invalid_sizes(tmp_path):
    path = tmp_path / "devices.json"
    path.write_text(json.dumps({"devices": [{"board": 1, "serial": 6843, "factory_lut": "w", "lut": "l"}, {"board": 2, "width_px": 0, "height_px": 5}]}), encoding="utf-8")
    one, two = settings.findDevice(1, path), settings.findDevice(2, path)
    assert (one.serial, one.width_px, one.height_px) == (6843, None, None) and one.imageShape() == (1024, 1024)
    assert (two.width_px, two.height_px) == (None, None)


def test_device_pixel_pitch_round_trip(tmp_path):
    path = tmp_path / "devices.json"
    settings.addDevice(1, path)
    assert settings.devicePixelPitchUm(1, path) == settings.loadDeviceDefaults().pixel_pitch_um == 17.0
    assert settings.setDevicePixelPitchUm(1, 9.2, path)
    assert settings.devicePixelPitchUm(1, path) == 9.2 and json.loads(path.read_text(encoding="utf-8"))["devices"][0]["pixel_pitch_um"] == 9.2
    assert settings.setDevicePixelPitchUm(settings.SIMULATED_BOARD, 12.5, path)
    assert json.loads(path.read_text(encoding="utf-8"))["simulated_slm"] == {"pixel_pitch_um": 12.5}
    for bad in [0.5, 101, True, "17"]:
        with pytest.raises(ValueError):
            settings.setDevicePixelPitchUm(1, bad, path)
    path.write_text(json.dumps({"devices": [{"board": 1, "pixel_pitch_um": 1000}]}), encoding="utf-8")
    assert settings.findDevice(1, path).pixel_pitch_um is None and settings.devicePixelPitchUm(1, path) == 17.0


def test_wfc_key_and_old_factory_lut_key(tmp_path):
    path = tmp_path / "devices.json"
    path.write_text(json.dumps({"devices": [{"board": 1, "factory_lut": "old_name"}, {"board": 2, "wfc": "new_name"}, {"board": 3, "wfc": "wins", "factory_lut": "loses"}]}), encoding="utf-8")
    assert [settings.findDevice(b, path).factory_lut for b in (1, 2, 3)] == ["old_name", "new_name", "wins"]
    settings.setDeviceLut(1, "x", path)
    written = json.loads(path.read_text(encoding="utf-8"))["devices"]
    assert all("factory_lut" not in d for d in written) and [d["wfc"] for d in written] == ["old_name", "new_name", "wins"]


def test_device_slm_radius_round_trip(tmp_path):
    path = tmp_path / "devices.json"
    settings.addDevice(1, path)
    assert settings.deviceSlmRadiusM(1, path) == settings.loadDeviceDefaults().slm_radius_m == 0.0087
    assert settings.setDeviceSlmRadiusM(1, 0.0065, path)
    assert settings.deviceSlmRadiusM(1, path) == 0.0065 and json.loads(path.read_text(encoding="utf-8"))["devices"][0]["slm_radius_m"] == 0.0065
    assert settings.setDeviceSlmRadiusM(settings.SIMULATED_BOARD, 0.004, path)
    assert json.loads(path.read_text(encoding="utf-8"))["simulated_slm"] == {"slm_radius_m": 0.004}
    for bad in [0, -0.001, 0.2, True, "0.005"]:
        with pytest.raises(ValueError):
            settings.setDeviceSlmRadiusM(1, bad, path)
    path.write_text(json.dumps({"devices": [{"board": 1, "slm_radius_m": -1}]}), encoding="utf-8")
    assert settings.findDevice(1, path).slm_radius_m is None and settings.deviceSlmRadiusM(1, path) == 0.0087


def test_simulated_slm_defaults_file_has_only_geometry():
    doc = json.loads(settings.SIMULATED_SLM_DEFAULTS_FILE.read_text(encoding="utf-8"))
    assert doc == {"width_px": 1024, "height_px": 1024, "pixel_pitch_um": 17.0, "slm_radius_m": 0.0087}
    assert settings.loadDeviceDefaults() == settings.DeviceGeometry(1024, 1024, 17.0, 0.0087)


@pytest.mark.parametrize("doc", [None, {"width_px": 1024, "height_px": 1024, "pixel_pitch_um": 17.0}, {"width_px": 0, "height_px": 1024, "pixel_pitch_um": 17.0, "slm_radius_m": 0.0087}])
def test_missing_or_invalid_simulated_slm_defaults_are_an_error(tmp_path, doc):
    path = tmp_path / "simulated_slm.json"
    if doc is not None:
        path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(settings.DeviceDefaultsError):
        settings.loadDeviceDefaults(path)


def test_device_name(tmp_path):
    path = tmp_path / "devices.json"
    settings.addDevice(1, path)
    assert settings.findDevice(1, path).name is None and "name" not in json.loads(path.read_text(encoding="utf-8"))["devices"][0]
    assert settings.defaultDeviceName(6843) == "slm6843"
    assert settings.setDeviceName(1, " slm6843 ", path)
    entry = json.loads(path.read_text(encoding="utf-8"))["devices"][0]
    assert entry["name"] == "slm6843" and list(entry)[:2] == ["name", "board"]
    assert settings.setDeviceName(1, "tweezer SLM", path) and settings.findDevice(1, path).name == "tweezer SLM"
    assert not settings.setDeviceName(settings.SIMULATED_BOARD, "x", path) and not settings.setDeviceName(5, "x", path)
    for bad in ["", "   ", None, "a/b", "simulated_slm"]:
        with pytest.raises(ValueError):
            settings.setDeviceName(1, bad, path)
    path.write_text(json.dumps({"devices": [{"board": 1, "name": 5}, {"board": 2, "name": "  "}]}), encoding="utf-8")
    assert settings.findDevice(1, path).name is None and settings.findDevice(2, path).name is None


def test_device_entry_key_order(tmp_path):
    path = tmp_path / "devices.json"
    settings.saveDevices([settings.Device(1, 6843, "w", "l", 1024, 1024, 17.0, 0.0087, "slm6843")], path)
    assert list(json.loads(path.read_text(encoding="utf-8"))["devices"][0]) == ["name", "board", "serial", "wfc", "lut", "width_px", "height_px", "pixel_pitch_um", "slm_radius_m"]


def test_device_folder_names(tmp_path):
    assert settings.deviceDirName(settings.Device(1)) == "board1"
    assert settings.deviceDirName(settings.Device(1, name="slm6843")) == "slm6843"
    assert settings.deviceDirName(settings.Device(settings.SIMULATED_BOARD)) == "simulated_slm"
    assert settings.deviceDir(settings.Device(2, name="x"), tmp_path) == tmp_path / "x"


def test_prepare_device_dir_creates_holograms_and_copies_old_files(tmp_path):
    (tmp_path / "w.bmp").write_bytes(b"wfc")
    (tmp_path / "l.LUT").write_text("0 0\n", encoding="ascii")
    device = settings.Device(1, factory_lut="w", lut="l", name="slm6843")
    assert settings.deviceWfcPath(device, tmp_path) == tmp_path / "w.bmp"
    assert sorted(settings.prepareDeviceDir(device, tmp_path)) == ["l.LUT", "w.bmp"]
    folder = tmp_path / "slm6843"
    assert (folder / "holograms").is_dir() and (folder / "w.bmp").read_bytes() == b"wfc" and (tmp_path / "w.bmp").exists()
    assert settings.deviceWfcPath(device, tmp_path) == folder / "w.bmp" and settings.deviceLutPath(device, tmp_path) == folder / "l.LUT"
    assert settings.prepareDeviceDir(device, tmp_path) == []
    assert settings.deviceLutPath(settings.Device(1, lut="missing", name="slm6843"), tmp_path) == folder / "missing.LUT"


def test_renaming_a_device_renames_its_folder(tmp_path):
    path = tmp_path / "devices.json"
    settings.addDevice(1, path)
    settings.addDevice(2, path)
    (tmp_path / "board1" / "holograms").mkdir(parents=True)
    (tmp_path / "board1" / "l.LUT").write_text("0 0\n", encoding="ascii")
    assert settings.setDeviceName(1, "slm6843", path)
    assert not (tmp_path / "board1").exists() and (tmp_path / "slm6843" / "l.LUT").exists()
    assert settings.setDeviceName(1, "tweezer SLM", path)
    assert (tmp_path / "tweezer SLM" / "holograms").is_dir() and not (tmp_path / "slm6843").exists()
    with pytest.raises(ValueError, match="already named"):
        settings.setDeviceName(2, "Tweezer SLM", path)
    (tmp_path / "board2").mkdir()
    (tmp_path / "taken").mkdir()
    with pytest.raises(ValueError, match="already exists"):
        settings.setDeviceName(2, "taken", path)
    assert (tmp_path / "board2").is_dir() and settings.findDevice(2, path).name is None
