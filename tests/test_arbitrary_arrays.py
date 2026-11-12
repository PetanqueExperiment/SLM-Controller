"""Tests for the non-UI modules of gui/panels/arbitrary_arrays: rois.json files, trap arrays, hologram files."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gui.parameters import param
from gui.constants import positionToStepPerPixel, positionToStepPerPixelForZ
from gui import hologram_handler as hologram_files
from gui.panels.arbitrary_arrays import rois
from gui.panels.arbitrary_arrays.traps import trapArrays
from gui.panels.arbitrary_arrays.tweezer_array import coords_npy_to_traps_pos, rectangular_lattice_coords

CX, CY = 0.5 * param.ImgResX, 0.5 * param.ImgResY


def test_pure_modules_do_not_load_qt_or_gpu():
    code = "import sys; import gui.panels.arbitrary_arrays.rois, gui.panels.arbitrary_arrays.traps, gui.hologram_handler; print('libslm.gpu' in sys.modules, 'PyQt5.QtWidgets' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent, check=True).stdout.split()
    assert out == ["False", "False"]


def test_resolve_rois_path(tmp_path):
    (tmp_path / "rois.json").write_text("{}", encoding="utf-8")
    (tmp_path / "lattice.json").write_text("{}", encoding="utf-8")
    d = str(tmp_path)
    assert rois.resolveRoisPath("", d) == str(tmp_path / "rois.json")
    assert rois.resolveRoisPath("lattice", d) == str(tmp_path / "lattice.json")
    assert rois.resolveRoisPath("lattice.json", d) == str(tmp_path / "lattice.json")
    assert rois.resolveRoisPath(str(tmp_path / "lattice.json"), "elsewhere") == str(tmp_path / "lattice.json")
    assert rois.resolveRoisPath("missing", d) is None


def test_lattice_round_trip_through_rois_json(tmp_path):
    arr = rectangular_lattice_coords(2, 3, 10.0, 8.0, sep_z_um=0.0, rotation_deg_z=0.0, center_gap_um=0.0)
    centers, side = rois.latticeCenters(arr, 10.0, 8.0)
    assert len(centers) == 6 and side == 8
    path = tmp_path / "rois.json"
    rois.writeRoisJson(str(path), centers, side, 30.0)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["format_version"] == 2 and doc["roi_sizes"] == [8] and doc["angle_deg"] == 30.0
    coords_m = rois.readTrapCoordsM(str(path))
    assert coords_m.shape == (6, 3) and np.all(coords_m[:, 2] == 0)
    np.testing.assert_allclose(coords_m[:, :2], arr[:, :2], atol=1e-10)
    np.testing.assert_array_equal(rois.readCentersPx(str(path)), np.asarray(centers))


def test_centers_to_coords_matches_saved_file(tmp_path):
    arr = rectangular_lattice_coords(3, 4, 13.7, 9.1, sep_z_um=5.0, rotation_deg_z=17.0, center_gap_um=4.0)
    centers, side = rois.latticeCenters(arr, 13.7, 9.1)
    path = tmp_path / "rois.json"
    rois.writeRoisJson(str(path), centers, side, 17.0)
    np.testing.assert_array_equal(rois.centersToCoordsM(centers), rois.readTrapCoordsM(str(path)))
    assert rois.centersToCoordsM([[1], "x"]) is None


def test_traps_pos_to_centers_px_inverts_centers_to_coords():
    centers = [[100.25, 200.5], [640.0, 512.0], [900.125, 30.75]]
    traps_pos = coords_npy_to_traps_pos(rois.centersToCoordsM(centers))
    np.testing.assert_allclose(rois.trapsPosToCentersPx(traps_pos), np.asarray(centers), atol=1e-9)


@pytest.mark.parametrize("content", ['{"centers": []}', '{"centers": [[1], "x"]}', "{}"])
def test_read_trap_coords_without_valid_centers(tmp_path, content):
    path = tmp_path / "rois.json"
    path.write_text(content, encoding="utf-8")
    assert rois.readTrapCoordsM(str(path)) is None


def test_pasted_centers_px_and_um():
    coords = np.array([[10.0, 20.0], [13.0, 24.0]])
    centers, side = rois.pastedCenters(coords, in_um=False)
    assert centers == [[10.0, 20.0], [13.0, 24.0]] and side == 5
    centers_um, _ = rois.pastedCenters(coords, in_um=True)
    assert centers_um[0] == [round(10.0 + CX, 4), round(20.0 + CY, 4)]
    assert rois.pastedCenters(coords[:1], in_um=False)[1] == 3


def test_trap_arrays_k_vectors_scale_with_wavelength():
    traps_pos = coords_npy_to_traps_pos(np.array([[10e-6, -5e-6, 1e-6], [0.0, 0.0, 0.0]]))
    np.random.seed(0)
    a = trapArrays(traps_pos, 850e-9, "1.5")
    b = trapArrays(traps_pos, 425e-9, "1.5")
    assert a["N_traps"] == 2
    np.testing.assert_allclose(a["traps_k"][:, 0:2], traps_pos[:, 1:3] * positionToStepPerPixel(850e-9))
    np.testing.assert_allclose(a["traps_k"][:, 2], traps_pos[:, 3] * positionToStepPerPixelForZ(850e-9))
    np.testing.assert_allclose(b["traps_k"], 2 * a["traps_k"])
    assert np.all(a["weight_traps"] == 1) and np.all((0 <= a["theta_traps"]) & (a["theta_traps"] < 2 * np.pi))


@pytest.mark.parametrize("text, axes_mm, angle_deg", [("1.5", (1.5, 1.5), 0.0), ("1, 2", (1.0, 2.0), 0.0), ("1, 2, 90", (1.0, 2.0), 90.0), ("", (100.0, 100.0), 100.0)])
def test_trap_arrays_pupil_text(text, axes_mm, angle_deg):
    pupil = trapArrays(coords_npy_to_traps_pos(np.zeros((3, 3))), 850e-9, text)["pupil_traps"]
    short_mm, long_mm = axes_mm
    np.testing.assert_allclose(pupil["pupil_shortaxis"], short_mm / (param.pslm * 1e3))
    np.testing.assert_allclose(pupil["pupil_longaxis"], long_mm / (param.pslm * 1e3))
    np.testing.assert_allclose(pupil["pupil_angle"], np.deg2rad(angle_deg))
    assert all(v.shape == (3,) for v in pupil.values())


def test_hologram_pkl_round_trip(tmp_path):
    traps_dict = {"traps_pos": np.arange(8.0).reshape(2, 4), "phasePattern": np.ones((4, 4)), "pupil_traps": {"pupil_angle": np.zeros(2)}}
    path = str(tmp_path / "holo.pkl")
    hologram_files.writeTrapsDict(path, traps_dict)
    loaded = hologram_files.readTrapsDict(path)
    assert sorted(loaded) == sorted(traps_dict)
    np.testing.assert_array_equal(loaded["traps_pos"], traps_dict["traps_pos"])


def test_legacy_pattern_files(tmp_path):
    npy = tmp_path / "old.npy"
    np.save(npy, np.ones((600, 800)))
    assert hologram_files.readLegacyPattern(str(npy)).shape == (600, 792)
    bmp = tmp_path / "old.bmp"
    Image.fromarray(np.full((4, 4), 128, dtype=np.uint8)).save(bmp)
    np.testing.assert_allclose(hologram_files.readLegacyPattern(str(bmp)), np.full((4, 4), 128 * 2 * np.pi / 256))
    assert hologram_files.readLegacyPattern(str(tmp_path / "x.txt")) is None
