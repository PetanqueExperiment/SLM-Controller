"""Tests for gui/temperature_log.py: daily CSV files written to several folders at once."""

from datetime import datetime

import pytest

from gui import temperature_log
from gui.defaults import LOGS_DIR
from gui.temperature_log import DailyTemperatureLog


def rows(path):
    return path.read_text(encoding="utf-8").splitlines()


def test_default_folders_are_project_and_logs_dir():
    assert temperature_log.LOG_DIRS == (temperature_log.PROJECT_LOG_DIR, LOGS_DIR)
    assert temperature_log.PROJECT_LOG_DIR.name == "temperature_logs"


def test_same_rows_in_every_folder(tmp_path):
    dirs = [tmp_path / "project", tmp_path / "logs"]
    log = DailyTemperatureLog(dirs)
    log.append(33.123, datetime(2026, 11, 11, 12, 0, 0))
    log.append(33.5, datetime(2026, 11, 11, 12, 0, 1))
    log.close()
    expected = ["timestamp,temperature_C", "2026-11-11T12:00:00,33.12", "2026-11-11T12:00:01,33.50"]
    for d in dirs:
        assert rows(d / "2026-11-11.csv") == expected


def test_new_file_each_day_and_header_once(tmp_path):
    log = DailyTemperatureLog([tmp_path])
    log.append(33.0, datetime(2026, 11, 11, 23, 59, 59))
    log.append(34.0, datetime(2026, 11, 12, 0, 0, 0))
    log.close()
    log = DailyTemperatureLog([tmp_path])
    log.append(35.0, datetime(2026, 11, 12, 0, 0, 1))
    log.close()
    assert rows(tmp_path / "2026-11-11.csv") == ["timestamp,temperature_C", "2026-11-11T23:59:59,33.00"]
    assert rows(tmp_path / "2026-11-12.csv") == ["timestamp,temperature_C", "2026-11-12T00:00:00,34.00", "2026-11-12T00:00:01,35.00"]


def test_one_failing_folder_does_not_stop_the_other(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("a file, not a folder", encoding="utf-8")
    good = tmp_path / "good"
    log = DailyTemperatureLog([blocked, good])
    with pytest.raises(OSError, match="blocked"):
        log.append(33.0, datetime(2026, 11, 11, 12, 0, 0))
    log.close()
    assert rows(good / "2026-11-11.csv") == ["timestamp,temperature_C", "2026-11-11T12:00:00,33.00"]
