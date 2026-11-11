"""Append SLM temperature samples to one CSV file per local calendar day, in several log folders at once."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import IO, Sequence

from gui.paths import LOGS_DIR

PROJECT_LOG_DIR: Path = Path(__file__).resolve().parent.parent / "temperature_logs"
LOG_DIRS: tuple[Path, ...] = (PROJECT_LOG_DIR, LOGS_DIR)


class _DailyCsv:
    """``<log_dir>/YYYY-MM-DD.csv`` with rows ``timestamp,temperature_C``; a new file is opened when the day changes."""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir: Path = Path(log_dir)
        self._day: str | None = None
        self._file: IO[str] | None = None

    def append(self, stamp: str, day: str, temperature_c: float) -> None:
        if self._file is None or self._day != day:
            self.close()
            self.log_dir.mkdir(parents=True, exist_ok=True)
            path = self.log_dir / f"{day}.csv"
            is_new = (not path.exists()) or path.stat().st_size == 0
            self._file = path.open("a", encoding="utf-8", newline="")
            self._day = day
            if is_new:
                self._file.write("timestamp,temperature_C\n")
        self._file.write(f"{stamp},{float(temperature_c):.2f}\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is None:
            return
        try:
            self._file.close()
        finally:
            self._file = None
            self._day = None


class DailyTemperatureLog:
    """Write each sample to ``YYYY-MM-DD.csv`` in every folder of ``log_dirs`` (default: the project folder and LOGS_DIR).

    The folders are independent: if one cannot be written, the others still get the sample and an OSError naming
    the failed folder(s) is raised afterwards.
    """

    def __init__(self, log_dirs: Sequence[Path] = LOG_DIRS) -> None:
        self._logs: list[_DailyCsv] = [_DailyCsv(d) for d in log_dirs]

    @property
    def log_dirs(self) -> list[Path]:
        return [log.log_dir for log in self._logs]

    def append(self, temperature_c: float, when: datetime | None = None) -> None:
        when = when or datetime.now()
        day = when.strftime("%Y-%m-%d")
        stamp = when.strftime("%Y-%m-%dT%H:%M:%S")
        errors: list[str] = []
        for log in self._logs:
            try:
                log.append(stamp, day, temperature_c)
            except OSError as e:
                log.close()
                errors.append(f"{log.log_dir}: {e}")
        if errors:
            raise OSError("temperature log not written to " + "; ".join(errors))

    def close(self) -> None:
        for log in self._logs:
            log.close()
