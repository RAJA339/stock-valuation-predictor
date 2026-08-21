"""
Pytest suite for bad-print detection.

Two failure directions, and the second is the one that matters more. Missing a
corrupt bar lets it set the range for everything downstream — that is the bug
this module was written for. But flagging a *real* move as bad data is worse:
markets gap, halt and limit-move, and a detector that erases genuine history
would quietly rewrite the record while looking like it was helping. Most of
these tests are therefore about what must NOT be flagged.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svp.analytics import dataquality as DQ          # noqa: E402


def _frame(close, tz=None, seed=0):
    close = np.asarray(close, dtype=float)
    idx = pd.date_range("2024-01-02", periods=len(close), freq="B", tz=tz)
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Open": close, "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Volume": rng.uniform(1e6, 5e6, len(close)),
    }, index=idx)


def _walk(n=260, seed=1, drift=0.0005, vol=0.015, start=50.0):
    rng = np.random.default_rng(seed)
    return start * np.exp(np.cumsum(rng.normal(drift, vol, n)))


class TestDetection:
    def test_a_spike_high_is_caught(self):
        df = _frame(_walk())
        df.iloc[120, df.columns.get_loc("High")] = 900.0
        rep = DQ.inspect(df)
        assert rep.n_flagged == 1
        assert not rep.is_clean

    def test_the_last_bar_is_not_exempt(self):
        """
        The case that defeated the previous fix: a corrupt *final* bar. Code
        that forces the latest price into view re-broke the axis on exactly
        this input, so it gets its own test.
        """
        df = _frame(_walk())
        df.iloc[-1, df.columns.get_loc("High")] = 900.0
        df.iloc[-1, df.columns.get_loc("Close")] = 850.0
        assert DQ.inspect(df).n_flagged >= 1
        clean, _ = DQ.clean(df)
        assert float(clean["High"].max()) < 200

    def test_impossible_ohlc_is_caught_without_statistics(self):
        df = _frame(_walk())
        df.iloc[50, df.columns.get_loc("High")] = 1.0     # high below low
        rep = DQ.inspect(df)
        assert rep.n_flagged >= 1
        assert any("impossible" in r for r in rep.reasons.values())

    def test_non_positive_prices_are_caught(self):
        df = _frame(_walk())
        df.iloc[70, df.columns.get_loc("Close")] = 0.0
        assert DQ.inspect(df).n_flagged >= 1

    def test_cleaning_removes_the_bar_and_fixes_the_range(self):
        df = _frame(_walk())
        df.iloc[120, df.columns.get_loc("High")] = 900.0
        before = float(df["High"].max())
        clean, rep = DQ.clean(df)
        assert len(clean) == len(df) - rep.n_flagged
        assert float(clean["High"].max()) < before / 4

    def test_report_names_dates_and_reasons(self):
        df = _frame(_walk())
        df.iloc[120, df.columns.get_loc("High")] = 900.0
        rep = DQ.inspect(df)
        assert rep.flagged_dates
        assert rep.reasons
        assert "excluded" in rep.note()


class TestNoFalsePositives:
    """A detector that erases real history is worse than the bug it fixes."""

    def test_a_clean_random_walk_is_untouched(self):
        for seed in range(6):
            assert DQ.inspect(_frame(_walk(seed=seed))).is_clean

    def test_a_genuine_twenty_percent_gap_survives(self):
        c = _walk(seed=2)
        c[150:] *= 1.20                       # a real repricing, then continues
        assert DQ.inspect(_frame(c)).is_clean

    def test_a_genuine_crash_survives(self):
        c = _walk(seed=3)
        c[150:] *= 0.65                       # −35% and stays there
        assert DQ.inspect(_frame(c)).is_clean

    def test_a_strong_trend_is_not_flagged_at_its_extremes(self):
        """A stock that triples must not have its recent highs erased."""
        c = _walk(n=400, seed=4, drift=0.004, vol=0.012)
        rep = DQ.inspect(_frame(c))
        assert rep.is_clean, rep.flagged_dates

    def test_high_volatility_alone_is_not_bad_data(self):
        assert DQ.inspect(_frame(_walk(seed=5, vol=0.05))).is_clean

    def test_clean_frames_are_returned_unchanged(self):
        df = _frame(_walk(seed=7))
        out, rep = DQ.clean(df)
        assert rep.is_clean
        assert out is df


class TestEdges:
    def test_short_history_is_never_flagged(self):
        """Too little context to judge — say nothing rather than guess."""
        assert DQ.inspect(_frame(_walk(n=15))).is_clean

    def test_missing_columns_is_an_empty_report(self):
        df = _frame(_walk()).drop(columns=["High"])
        assert DQ.flag(df) is None
        assert DQ.inspect(df).n_bars == 0

    def test_empty_frame(self):
        assert DQ.flag(pd.DataFrame()) is None
        out, rep = DQ.clean(pd.DataFrame())
        assert rep.is_clean

    def test_timezone_aware_frames_work(self):
        df = _frame(_walk(), tz="America/New_York")
        df.iloc[100, df.columns.get_loc("High")] = 900.0
        clean, rep = DQ.clean(df)
        assert rep.n_flagged == 1
        assert clean.index.tz is not None

    def test_a_flat_series_does_not_divide_by_zero(self):
        df = _frame(np.full(120, 50.0))
        assert DQ.inspect(df).is_clean

    def test_note_is_empty_when_clean(self):
        assert DQ.inspect(_frame(_walk(seed=9))).note() == ""
