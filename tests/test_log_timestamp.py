"""Tests for gui/log_timestamp.py: date and time before every log line."""

import re

from gui.log_timestamp import LineTimestamper


def test_print_pieces_get_one_stamp_per_line():
    stamper = LineTimestamper(now=lambda: "2026.11.21 14:05:09")
    out = "".join(stamper.stamp(piece) for piece in ["Armed: 3 frames", "\n", "a\nb\n", "", "partial", " line\n"])
    assert out == "2026.11.21 14:05:09 Armed: 3 frames\n2026.11.21 14:05:09 a\n2026.11.21 14:05:09 b\n2026.11.21 14:05:09 partial line\n"


def test_default_format():
    out = LineTimestamper().stamp("x\n")
    assert re.fullmatch(r"\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2} x\n", out)
