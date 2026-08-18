"""
Pytest suite for the market session clock.

Fixed timestamps with known answers — a Tuesday mid-morning is Open, a
Saturday is Closed, and a UTC timestamp must be judged in Eastern time, not
its own. The honesty property: the Open label always carries the holiday
caveat, because holidays are not modelled and the label must say so itself.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svp import marketclock as mc          # noqa: E402

ET = mc.ET


class TestStatus:
    def test_tuesday_midmorning_is_open(self):
        s = mc.status(datetime(2026, 8, 18, 10, 30, tzinfo=ET))
        assert s.label == "Open" and s.is_open

    def test_saturday_is_closed(self):
        s = mc.status(datetime(2026, 8, 22, 11, 0, tzinfo=ET))
        assert s.label == "Closed" and not s.is_open
        assert "Weekend" in s.note

    def test_early_weekday_is_premarket(self):
        s = mc.status(datetime(2026, 8, 18, 7, 0, tzinfo=ET))
        assert s.label == "Pre-market" and not s.is_open

    def test_evening_is_afterhours(self):
        s = mc.status(datetime(2026, 8, 18, 17, 30, tzinfo=ET))
        assert s.label == "After-hours" and not s.is_open

    def test_late_night_is_closed(self):
        s = mc.status(datetime(2026, 8, 18, 23, 0, tzinfo=ET))
        assert s.label == "Closed"

    def test_boundaries(self):
        assert mc.status(datetime(2026, 8, 18, 9, 30, tzinfo=ET)).label == "Open"
        assert mc.status(datetime(2026, 8, 18, 16, 0, tzinfo=ET)).label == "After-hours"

    def test_utc_input_is_judged_in_eastern(self):
        # 18:00 UTC in August == 14:00 ET → Open, even though 18:00 > 16:00.
        s = mc.status(datetime(2026, 8, 18, 18, 0, tzinfo=ZoneInfo("UTC")))
        assert s.label == "Open"

    def test_open_label_admits_holiday_blindness(self):
        s = mc.status(datetime(2026, 8, 18, 12, 0, tzinfo=ET))
        assert "Holidays not modelled" in s.note
