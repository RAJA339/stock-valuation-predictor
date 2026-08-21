"""
Pytest suite for the volume profile and the AMD study.

The load-bearing test in this file is the null: fed a random walk, the AMD
study must report that follow-through is indistinguishable from a coin flip.
A pattern language that finds an edge in noise is worse than useless, and it
is the easy direction to get wrong — so it is checked alongside a constructed
positive control that the detector *must* see, proving the null result is
honesty rather than a broken detector.

The profile is checked against hand-placed volume: if 80% of the volume trades
in one narrow shelf, the POC must land on that shelf and the value area must
be narrower than the range.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svp.analytics import amd as AMD, volume_profile as VP     # noqa: E402


def _bars(close, volume=None, spread=0.004, start="2022-01-03"):
    close = np.asarray(close, dtype=float)
    idx = pd.date_range(start, periods=len(close), freq="B")
    if volume is None:
        volume = np.full(len(close), 1e6)
    return pd.DataFrame({
        "Open": close,
        "High": close * (1 + spread),
        "Low": close * (1 - spread),
        "Close": close,
        "Volume": np.asarray(volume, dtype=float),
    }, index=idx)


# ── Volume profile ───────────────────────────────────────────────────────────
class TestVolumeProfile:
    def test_poc_lands_where_the_volume_traded(self):
        """80% of volume on a tight shelf at 100 — the POC must be there."""
        rng = np.random.default_rng(0)
        shelf = np.full(160, 100.0) + rng.normal(0, 0.05, 160)
        tails = np.concatenate([np.full(20, 90.0), np.full(20, 110.0)])
        close = np.concatenate([shelf, tails])
        vol = np.concatenate([np.full(160, 5e6), np.full(40, 2e5)])
        p = VP.profile(_bars(close, vol, spread=0.001))
        assert p is not None
        assert p.poc == pytest.approx(100.0, abs=1.0)

    def test_value_area_is_narrower_than_the_range(self):
        rng = np.random.default_rng(1)
        p = VP.profile(_bars(100 + rng.normal(0, 2, 300)))
        assert p.val < p.poc < p.vah
        assert (p.vah - p.val) < (p.high - p.low)

    def test_value_area_holds_about_seventy_percent(self):
        rng = np.random.default_rng(2)
        p = VP.profile(_bars(100 + rng.normal(0, 2, 400)))
        inside = sum(v for m, v in zip(p.bin_prices, p.bin_volumes)
                     if p.val <= m <= p.vah)
        assert 0.65 <= inside / p.total_volume <= 0.90

    def test_volume_is_spread_across_the_bar_not_dumped_at_the_close(self):
        """
        A single wide bar must light up every bin it spans. The old close-only
        approach put all of it in one bin — the bug this module replaces.
        """
        df = _bars(np.full(40, 100.0), spread=0.10)      # each bar spans 90–110
        p = VP.profile(df, bins=20)
        occupied = sum(1 for v in p.bin_volumes if v > 0)
        assert occupied >= 15

    def test_total_volume_is_conserved(self):
        rng = np.random.default_rng(3)
        vol = rng.uniform(1e5, 9e5, 200)
        p = VP.profile(_bars(100 + rng.normal(0, 3, 200), vol))
        assert p.total_volume == pytest.approx(vol.sum(), rel=1e-6)

    def test_position_and_reading_track_the_price(self):
        rng = np.random.default_rng(4)
        p = VP.profile(_bars(100 + rng.normal(0, 2, 300)))
        assert p.position_of(p.poc) == "inside value"
        assert p.position_of(p.vah + 10) == "above value"
        assert p.position_of(p.val - 10) == "below value"
        assert "above the value area" in p.reading(p.vah + 10)

    def test_reading_never_predicts(self):
        rng = np.random.default_rng(5)
        p = VP.profile(_bars(100 + rng.normal(0, 2, 250)))
        for px in (p.val - 5, p.poc, p.vah + 5):
            low = p.reading(px).lower()
            assert "will " not in low and "buy" not in low and "sell" not in low

    def test_short_history_is_none(self):
        assert VP.profile(_bars(np.full(5, 100.0))) is None

    def test_missing_volume_is_none(self):
        df = _bars(np.full(50, 100.0)).drop(columns=["Volume"])
        assert VP.profile(df) is None

    def test_zero_volume_is_none(self):
        assert VP.profile(_bars(np.full(50, 100.0), np.zeros(50))) is None

    def test_flat_price_is_none_not_a_divide_by_zero(self):
        df = _bars(np.full(50, 100.0), spread=0.0)
        assert VP.profile(df) is None


# ── AMD ──────────────────────────────────────────────────────────────────────
class TestAMD:
    def _random_walk(self, n=1200, seed=7):
        rng = np.random.default_rng(seed)
        return 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))

    def test_random_walk_reads_as_a_coin_flip(self):
        """
        The null. On a random walk the framework must not find an edge — if it
        did, the detector would be manufacturing one.
        """
        s = AMD.detect(_bars(self._random_walk()))
        assert s is not None
        if s.n_events >= AMD.MIN_EVENTS:
            assert not s.is_better_than_chance
            assert "coin flip" in s.verdict or "below chance" in s.verdict

    def test_constructed_amd_is_detected_and_scores_high(self):
        """
        Positive control: tapes that coil flat, sweep the low and reclaim it,
        then rally. The detector must find these and score their
        follow-through — the proof that the null result above is honesty
        rather than blindness.

        The coil is deliberately *flat*. An earlier version of this fixture
        drifted upward inside the coil, so a later coil bar exceeded the
        trailing range high and registered as a bearish sweep before the real
        one — the detector was right and the fixture was wrong.
        """
        rows, level = [], 100.0
        for _ in range(30):
            for _ in range(30):                      # flat coil
                rows.append((level, level * 1.0015, level * 0.9985, level))
            # Sweep the floor and close back inside it.
            rows.append((level, level * 1.0010, level * 0.970, level * 1.0005))
            for k in range(1, 13):                   # distribute upward
                px = level * (1 + 0.008 * k)
                rows.append((px, px * 1.002, px * 0.998, px))
            level = rows[-1][3]

        arr = np.array(rows, dtype=float)
        df = pd.DataFrame(
            {"Open": arr[:, 0], "High": arr[:, 1], "Low": arr[:, 2],
             "Close": arr[:, 3], "Volume": np.full(len(arr), 1e6)},
            index=pd.date_range("2015-01-01", periods=len(arr), freq="B"))

        s = AMD.detect(df, acc_bars=20, manip_window=6, horizon=10)
        assert s.n_events >= AMD.MIN_EVENTS
        assert all(e.direction == "bullish" for e in s.events)
        assert s.rate > 0.8
        assert s.is_better_than_chance
        assert "above chance" in s.verdict

    def test_events_do_not_overlap(self):
        s = AMD.detect(_bars(self._random_walk(seed=11)))
        dates = [e.sweep_date for e in s.events]
        assert len(dates) == len(set(dates))
        assert dates == sorted(dates)

    def test_sweep_direction_matches_the_break(self):
        s = AMD.detect(_bars(self._random_walk(seed=3)))
        for e in s.events:
            if e.direction == "bullish":
                assert e.sweep_extreme < e.acc_low       # undercut the floor
            else:
                assert e.sweep_extreme > e.acc_high      # exceeded the ceiling

    def test_follow_through_flag_matches_the_return(self):
        s = AMD.detect(_bars(self._random_walk(seed=5)))
        for e in s.events:
            expected = (e.forward_return_pct > 0 if e.direction == "bullish"
                        else e.forward_return_pct < 0)
            assert e.followed_through == expected

    def test_small_sample_quotes_no_rate(self):
        rng = np.random.default_rng(9)
        s = AMD.detect(_bars(100 * np.exp(np.cumsum(rng.normal(0, .01, 150)))))
        if s is not None and s.n_events < AMD.MIN_EVENTS:
            assert "too few" in s.verdict.lower()

    def test_current_phase_reports_a_known_coil(self):
        """A tape that ends inside a tight range must read as accumulation."""
        rng = np.random.default_rng(2)
        body = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300)))
        coil = np.full(40, body[-1]) + rng.normal(0, 0.02, 40)
        s = AMD.detect(_bars(np.concatenate([body, coil]), spread=0.0005))
        assert s.current_phase == "Accumulation"
        assert "contracted range" in s.current_note

    def test_current_note_never_forecasts(self):
        for seed in (1, 4, 8):
            s = AMD.detect(_bars(self._random_walk(seed=seed)))
            low = s.current_note.lower()
            assert "will rise" not in low and "will fall" not in low
            assert "guarantee" not in low

    def test_short_history_is_none(self):
        assert AMD.detect(_bars(self._random_walk(n=50))) is None

    def test_missing_columns_is_none(self):
        df = _bars(self._random_walk()).drop(columns=["High"])
        assert AMD.detect(df) is None
