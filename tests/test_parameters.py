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


def test_template_has_the_previous_values_exactly():
    assert asdict(readTemplate()) == PREVIOUS_VALUES


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
    doc['slm_width_px'] = 512
    path.write_text(json.dumps(doc), encoding='utf-8')
    same_object = param
    assert loadParameters(path) == []
    assert same_object is parameters.param and param.f == 0.02 and param.ImgResX == 512 and param.M == 350 / 150


def test_missing_and_invalid_values_use_defaults(tmp_path):
    path = tmp_path / 'parameters.json'
    path.write_text(json.dumps({'slm_width_px': 10.5, 'objective_focal_length_m': -1, 'telescope_magnification': 3.0, 'extra': 1}), encoding='utf-8')
    warnings = loadParameters(path)
    assert param.M == 3.0
    assert param.ImgResX == 1024 and param.f == 23e-3 and param.pslm == 17.0e-6
    text = ' '.join(warnings)
    assert 'slm_width_px' in text and 'objective_focal_length_m' in text and 'slm_pixel_pitch_m is missing' in text and 'extra' in text


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
