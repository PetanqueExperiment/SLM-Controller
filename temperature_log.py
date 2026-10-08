"""Append SLM temperature samples to one CSV file per local calendar day."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "temperature_logs"


class DailyTemperatureLog:
    """Write `temperature_logs/YYYY-MM-DD.csv` rows: timestamp, temperature_C."""

    def __init__(self, log_dir: Path = LOG_DIR):
        self.log_dir = Path(log_dir)
        self._day: str | None = None
        self._file = None

    def append(self, temperature_c: float, when: datetime | None = None) -> None:
        when = when or datetime.now()
        day = when.strftime("%Y-%m-%d")
        if self._file is None or self._day != day:
            self.close()
            self.log_dir.mkdir(parents=True, exist_ok=True)
            path = self.log_dir / f"{day}.csv"
            is_new = (not path.exists()) or path.stat().st_size == 0
            self._file = path.open("a", encoding="utf-8", newline="")
            self._day = day
            if is_new:
                self._file.write("timestamp,temperature_C\n")
        stamp = when.strftime("%Y-%m-%dT%H:%M:%S")
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
