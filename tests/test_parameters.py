"""Tests for gui/parameters.py: defaults template, loading parameters.json into ``param`` (no Qt)."""

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from gui import parameters
from gui.parameters import JSON_KEYS, ParametersError, SlmParameters, loadParameters, param, readTemplate

REPO = Path(__file__).resolve().parents[1]
PREVIOUS_VALUES = {'ImgResX': 1024, 'ImgResY': 1024, 'f': 23e-3, 'M': 350 / 150, 'pslm': 17.0e-6, 'max_radius': 17.4e-3 / 2}


@pytest.fixture(autouse=True)
def restore_param():
    saved = asdict(param)
    yield
    for name, value in saved.items():
        setattr(param, name, value)


def test_template_and_device_defaults_give_the_previous_values_exactly():
    template = asdict(readTemplate())
    assert {name: template[name] for name in JSON_KEYS} == {name: PREVIOUS_VALUES[name] for name in JSON_KEYS}
    parameters.applyDeviceDefaults()
    assert asdict(param) == PREVIOUS_VALUES


def test_missing_file_is_created_from_template(tmp_path):
    path = tmp_path / 'parameters' / 'parameters.json'
    assert loadParameters(path) == []
    assert path.is_file()
    assert json.loads(path.read_text(encoding='utf-8')) == json.loads(parameters.PARAMETERS_TEMPLATE_FILE.read_text(encoding='utf-8'))
    assert asdict(param) == PREVIOUS_VALUES


def test_values_are_loaded_in_place(tmp_path):
    path = tmp_path / 'parameters.json'
    doc = {key: PREVIOUS_VALUES[name] for name, key in JSON_KEYS.items()}
    doc['objective_focal_length_m'] = 0.02
    path.write_text(json.dumps(doc), encoding='utf-8')
    same_object = param
    assert loadParameters(path) == []
    assert same_object is parameters.param and param.f == 0.02 and param.M == 350 / 150


def test_missing_and_invalid_values_use_defaults(tmp_path):
    path = tmp_path / 'parameters.json'
    path.write_text(json.dumps({'objective_focal_length_m': -1, 'telescope_magnification': 3.0, 'extra': 1}), encoding='utf-8')
    warnings = loadParameters(path)
    assert param.M == 3.0
    assert param.f == 23e-3
    text = ' '.join(warnings)
    assert 'objective_focal_length_m' in text and 'extra' in text


def test_unreadable_file_uses_all_defaults(tmp_path):
    path = tmp_path / 'parameters.json'
    path.write_text('{not json', encoding='utf-8')
    param.f = 1.0
    warnings = loadParameters(path)
    assert len(warnings) == 1 and 'could not be read' in warnings[0]
    assert asdict(param) == PREVIOUS_VALUES


def test_missing_template_is_an_error(tmp_path):
    with pytest.raises(ParametersError):
        readTemplate(tmp_path / 'missing.json')


def test_old_module_is_gone_and_loader_has_no_qt():
    assert not (REPO / 'libslm' / 'parameters.py').exists()
    code = "import sys; import gui.parameters; print('PyQt5.QtWidgets' in sys.modules)"
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=REPO, check=True)
    assert out.stdout.strip() == 'False'


def test_param_is_a_slm_parameters():
    assert isinstance(param, SlmParameters)


def test_write_parameters_round_trip(tmp_path):
    path = tmp_path / 'parameters' / 'parameters.json'
    values = dict(PREVIOUS_VALUES, f=0.02)
    parameters.writeParameters(values, path)
    assert json.loads(path.read_text(encoding='utf-8')) == {key: values[name] for name, key in JSON_KEYS.items()}
    assert 'slm_width_px' not in path.read_text(encoding='utf-8')
    assert loadParameters(path) == [] and param.f == 0.02


@pytest.mark.parametrize('name, bad', [('f', -1.0), ('M', float('nan')), ('M', 'x')])
def test_write_parameters_rejects_invalid_values_and_keeps_the_file(tmp_path, name, bad):
    path = tmp_path / 'parameters.json'
    parameters.writeParameters(dict(PREVIOUS_VALUES), path)
    before = path.read_text(encoding='utf-8')
    with pytest.raises(ValueError, match=JSON_KEYS[name]):
        parameters.writeParameters(dict(PREVIOUS_VALUES, **{name: bad}), path)
    assert path.read_text(encoding='utf-8') == before


def test_image_size_is_not_in_the_parameters_file():
    assert set(JSON_KEYS) == {'f', 'M'}
    template = json.loads(parameters.PARAMETERS_TEMPLATE_FILE.read_text(encoding='utf-8'))
    assert not {'slm_width_px', 'slm_height_px', 'slm_pixel_pitch_m', 'slm_max_radius_m'} & set(template)


def test_old_file_with_image_size_loads_without_warning_and_keeps_the_size(tmp_path):
    path = tmp_path / 'parameters.json'
    doc = {key: PREVIOUS_VALUES[name] for name, key in JSON_KEYS.items()}
    doc.update(slm_width_px=512, slm_height_px=256, slm_pixel_pitch_m=9.2e-6, slm_max_radius_m=0.005)
    path.write_text(json.dumps(doc), encoding='utf-8')
    parameters.setImageSize(1920, 1152)
    parameters.setPixelPitchUm(9.2)
    assert loadParameters(path) == []
    assert (param.ImgResX, param.ImgResY) == (1920, 1152) and param.pslm == 9.2 / 1e6


def test_set_image_size():
    parameters.setImageSize(800, 600)
    assert (param.ImgResX, param.ImgResY) == (800, 600)
    for bad in [(0, 600), (800.0, 600), (True, 600), (800, -1)]:
        with pytest.raises(ValueError):
            parameters.setImageSize(*bad)
    assert (param.ImgResX, param.ImgResY) == (800, 600)


def test_set_pixel_pitch_um_gives_the_exact_previous_value():
    parameters.setPixelPitchUm(17.0)
    assert param.pslm == 1.7e-05 == PREVIOUS_VALUES['pslm']
    for bad in [0, -1.0, float('nan'), True, '17']:
        with pytest.raises(ValueError):
            parameters.setPixelPitchUm(bad)
    assert param.pslm == 1.7e-05


def test_set_slm_radius_m_default_is_the_previous_value():
    parameters.applyDeviceDefaults()
    assert param.max_radius == PREVIOUS_VALUES['max_radius'] == 0.0087
    parameters.setSlmRadiusM(0.005)
    assert param.max_radius == 0.005
    for bad in [0, -1.0, float('inf'), True, '1']:
        with pytest.raises(ValueError):
            parameters.setSlmRadiusM(bad)
    assert param.max_radius == 0.005
