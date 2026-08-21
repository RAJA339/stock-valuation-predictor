"""
Pytest suite for fair value gaps and their inversions.

Detection is checked against hand-built bars where the gap's edges are known
to the cent — a gap finder that is a few cents out is worse than none, since
every level downstream inherits the error. The lifecycle rules get their own
tests because they are where the framework is easy to fudge: a wick through
a gap is a test, only a *close* through inverts it.

As with AMD, the load-bearing test is the null: on a random walk, retested
inverted gaps must read as a coin flip.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svp.analytics import fvg as F          # noqa: E402


def _frame(rows, start="2022-01-03"):
    """rows: iterable of (open, high, low, close)."""
    arr = np.array(rows, dtype=float)
    return pd.DataFrame(
        {"Open": arr[:, 0], "High": arr[:, 1], "Low": arr[:, 2],
         "Close": arr[:, 3], "Volume": np.full(len(arr), 1e6)},
        index=pd.date_range(start, periods=len(arr), freq="B"))


def _flat(n, px=100.0, halfspan=0.5):
    return [(px, px + halfspan, px - halfspan, px)] * n


def _walk(n=600, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    rows = []
    for c in close:
        rows.append((c, c * 1.006, c * 0.994, c))
    return _frame(rows)


# ── Detection ────────────────────────────────────────────────────────────────
class TestFindGaps:
    def test_bullish_gap_edges_are_exact(self):
        # bar1 high 101, bar3 low 104 → gap band is exactly 101–104.
        rows = [(100, 101, 99, 100), (103, 106, 102, 105), (105, 107, 104, 106)]
        [g] = F.find_gaps(_frame(rows))
        assert g.direction == "bullish"
        assert g.bottom == pytest.approx(101.0)
        assert g.top == pytest.approx(104.0)
        assert g.role == "support"
        assert g.label == "Bullish FVG"

    def test_bearish_gap_edges_are_exact(self):
        # bar1 low 104, bar3 high 101 → band 101–104.
        rows = [(105, 106, 104, 105), (102, 103, 99, 100), (100, 101, 98, 99)]
        [g] = F.find_gaps(_frame(rows))
        assert g.direction == "bearish"
        assert g.bottom == pytest.approx(101.0)
        assert g.top == pytest.approx(104.0)
        assert g.role == "resistance"

    def test_overlapping_bars_are_not_a_gap(self):
        rows = [(100, 103, 99, 102), (101, 104, 100, 103), (102, 105, 101, 104)]
        assert F.find_gaps(_frame(rows)) == []

    def test_gaps_thinner_than_the_floor_are_ignored(self):
        """A one-cent band on a $100 stock is noise, not an imbalance."""
        rows = [(100, 100.00, 99, 100), (100, 101, 99.9, 100.5),
                (100, 101, 100.01, 100.5)]
        assert F.find_gaps(_frame(rows)) == []

    def test_missing_columns_is_empty(self):
        df = _frame(_flat(10)).drop(columns=["High"])
        assert F.find_gaps(df) == []


# ── Inversion lifecycle ──────────────────────────────────────────────────────
class TestInversion:
    """
    Fixtures here contain more than one gap on purpose: stepping a flat tape
    up into a setup *is* an imbalance, and the detector is right to see it.
    Each test therefore picks out the gap it built rather than assuming it is
    the only one — asserting a count would be testing the fixture, not the
    lifecycle rule.
    """

    @staticmethod
    def _band(gaps, bottom, top):
        for g in gaps:
            if (abs(g.bottom - bottom) < 1e-6) and (abs(g.top - top) < 1e-6):
                return g
        raise AssertionError(f"no gap at {bottom}-{top} in "
                             f"{[(x.bottom, x.top) for x in gaps]}")

    def _bullish_then(self, tail):
        # The intended gap is 101–104 (bar1 high 101, bar3 low 104).
        rows = (_flat(30, 100.0)
                + [(100, 101, 99, 100), (103, 106, 102, 105), (105, 107, 104, 106)]
                + tail)
        return _frame(rows)

    def test_a_wick_through_does_not_invert(self):
        """Trading into the band is a test; only a close through flips it."""
        tail = [(105, 106, 100.5, 105)] * 30      # dips into 101–104, closes above
        s = F.detect(self._bullish_then(tail))
        g = self._band(s.gaps, 101.0, 104.0)
        assert not g.inverted
        assert g.role == "support"

    def test_a_close_below_inverts_it_to_resistance(self):
        tail = [(102, 103, 99, 100.0)] + _flat(40, 100.0)
        s = F.detect(self._bullish_then(tail))
        g = self._band(s.gaps, 101.0, 104.0)
        assert g.inverted
        assert g.role == "resistance"
        assert g.label == "Bearish IFVG"
        assert g.inverted_date

    def test_bearish_gap_inverts_to_support_on_a_close_above(self):
        # Flat at 105 so the step into the setup makes no extra gap below it;
        # the intended bearish band is 101–104.
        rows = (_flat(30, 105.0)
                + [(105, 106, 104, 105), (102, 103, 99, 100), (100, 101, 98, 99)]
                + [(104, 106, 103, 105.0)] + _flat(40, 105.0))
        s = F.detect(_frame(rows))
        g = self._band(s.gaps, 101.0, 104.0)
        assert g.direction == "bearish"
        assert g.inverted
        assert g.role == "support"
        assert g.label == "Bullish IFVG"

    def test_open_and_inverted_partition_the_gaps(self):
        s = F.detect(_walk(seed=4))
        assert len(s.open_gaps) + len(s.inverted_gaps) == len(s.gaps)


# ── Measurement ──────────────────────────────────────────────────────────────
class TestRetestMeasurement:
    def test_random_walk_reads_as_a_coin_flip(self):
        """The null — an inverted gap must not manufacture an edge in noise."""
        s = F.detect(_walk(n=1500, seed=12))
        assert s is not None
        if s.n_retests >= F.MIN_RETESTS:
            assert not s.is_better_than_chance
            assert "coin flip" in s.verdict or "below chance" in s.verdict

    def test_respected_flag_matches_the_direction_moved(self):
        s = F.detect(_walk(n=1200, seed=6))
        for r in s.retests:
            expected = (r.forward_return_pct < 0 if r.role == "resistance"
                        else r.forward_return_pct > 0)
            assert r.respected == expected

    def test_retests_do_not_overlap_within_a_zone(self):
        s = F.detect(_walk(n=1200, seed=8), horizon=5)
        by_zone: dict = {}
        for r in s.retests:
            by_zone.setdefault((r.zone_bottom, r.zone_top), []).append(r.date)
        for dates in by_zone.values():
            assert len(dates) == len(set(dates))

    def test_small_sample_quotes_no_rate(self):
        s = F.detect(_walk(n=200, seed=2))
        if s.n_retests < F.MIN_RETESTS:
            assert "too few" in s.verdict.lower()

    def test_short_history_is_none(self):
        assert F.detect(_frame(_flat(20))) is None

    def test_no_gaps_is_an_empty_study_not_a_crash(self):
        s = F.detect(_frame(_flat(120)))
        assert s is not None
        assert s.gaps == [] and s.n_retests == 0


# ── UI-facing helpers ────────────────────────────────────────────────────────
class TestHelpers:
    def test_nearest_helpers_sort_by_distance(self):
        s = F.detect(_walk(seed=5))
        px = 100.0
        near = F.active_ifvgs(s, px, n=3)
        gaps_d = [abs(g.mid - px) for g in near]
        assert gaps_d == sorted(gaps_d)

    def test_zone_levels_emit_two_lines_per_gap(self):
        s = F.detect(_walk(seed=7))
        gaps = s.gaps[:3]
        lines = F.zone_levels(gaps)
        assert len(lines) == 2 * len(gaps)
        assert all("price" in ln and "title" in ln for ln in lines)

    def test_summary_note_never_forecasts(self):
        s = F.detect(_walk(seed=9))
        note = F.summary_note(s, 100.0).lower()
        assert "will " not in note and "buy" not in note and "sell" not in note

    def test_summary_note_handles_no_gaps(self):
        s = F.detect(_frame(_flat(120)))
        assert "no fair value gaps" in F.summary_note(s, 100.0).lower()

    def test_gap_frame_columns(self):
        s = F.detect(_walk(seed=1))
        df = F.gap_frame(s.gaps[:5])
        assert list(df.columns) == ["Zone", "Band", "Formed",
                                    "Closed through", "Acts as"]
