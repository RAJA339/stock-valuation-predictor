"""
Pytest suite for the microstructure metrics.

The property that matters most here is honesty under a null: fed a pure random
walk, the sweep and round-number studies must report "no better than chance",
because the whole point of the module is to separate real structure from the
patterns the eye invents. Several tests therefore assert that a signal is *not*
claimed where none exists — the harder direction to get right, and the one a
signal-service would skip.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svp.analytics import microstructure as M          # noqa: E402


def _daily(open_, close, high=None, low=None, n_index=None):
    n = len(close)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.DataFrame({
        "Open": open_,
        "High": high if high is not None else np.maximum(open_, close) * 1.001,
        "Low": low if low is not None else np.minimum(open_, close) * 0.999,
        "Close": close,
    }, index=idx)


# ── Overnight vs intraday ────────────────────────────────────────────────────
class TestOvernightSplit:
    def _series(self, on_bp, id_bp, n=300, seed=1):
        rng = np.random.default_rng(seed)
        prev, opens, closes = 100.0, [], []
        for _ in range(n):
            o = prev * (1 + on_bp / 1e4 + rng.normal(0, 0.002))
            c = o * (1 + id_bp / 1e4 + rng.normal(0, 0.002))
            opens.append(o)
            closes.append(c)
            prev = c
        return _daily(np.array(opens), np.array(closes))

    def test_detects_overnight_drift(self):
        s = M.overnight_intraday_split(self._series(on_bp=10, id_bp=-6))
        assert s.overnight_mean_bp > 0 > s.intraday_mean_bp
        assert s.intraday_is_negative
        assert "overnight" in s.verdict.lower()

    def test_the_two_legs_reconstruct_close_to_close(self):
        """The decomposition is an identity, not an approximation."""
        s = M.overnight_intraday_split(self._series(on_bp=5, id_bp=3))
        recombined = (1 + s.overnight_total / 100) * (1 + s.intraday_total / 100) - 1
        assert recombined * 100 == pytest.approx(s.total, rel=1e-9)

    def test_short_history_returns_none(self):
        assert M.overnight_intraday_split(self._series(on_bp=1, id_bp=1, n=30)) is None

    def test_missing_columns_returns_none(self):
        assert M.overnight_intraday_split(pd.DataFrame({"Close": range(100)})) is None

    def test_empty_returns_none(self):
        assert M.overnight_intraday_split(pd.DataFrame()) is None

    def test_share_is_none_on_a_decliner(self):
        """The overnight share of an up-move is not defined when the name fell."""
        s = M.overnight_intraday_split(self._series(on_bp=-5, id_bp=-5))
        assert s.overnight_share is None

    def test_intraday_outperformance_is_named(self):
        s = M.overnight_intraday_split(self._series(on_bp=1, id_bp=12))
        assert "intraday" in s.verdict.lower()


# ── Time-of-day profile ──────────────────────────────────────────────────────
class TestTimeOfDay:
    def _intraday(self, sessions=4):
        rows, idx = [], []
        for d in range(sessions):
            times = pd.date_range(f"2023-05-0{d+1} 09:30", periods=13, freq="30min")
            for i, t in enumerate(times):
                # U-shape: heavy volume at the ends, light in the middle.
                edge = min(i, len(times) - 1 - i)
                rows.append({"Close": 100 + 0.1 * i, "Volume": 1000 - 120 * edge})
                idx.append(t)
        return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))

    def test_profile_buckets_the_session(self):
        p = M.time_of_day_profile(self._intraday())
        assert p is not None
        assert len(p.labels) == 13
        assert p.n_sessions == 4

    def test_volume_shares_are_normalised_per_session(self):
        p = M.time_of_day_profile(self._intraday())
        # Each session's shares sum to 100, so the average across buckets does too.
        assert sum(p.volume_share) == pytest.approx(100.0, abs=1e-6)

    def test_u_shape_puts_the_peak_at_an_edge(self):
        p = M.time_of_day_profile(self._intraday())
        peak = int(np.argmax(p.volume_share))
        assert peak in (0, len(p.volume_share) - 1)

    def test_single_session_returns_none(self):
        one = self._intraday(sessions=1)
        assert M.time_of_day_profile(one) is None

    def test_non_datetime_index_returns_none(self):
        df = pd.DataFrame({"Close": range(50), "Volume": range(50)})
        assert M.time_of_day_profile(df) is None

    def test_empty_returns_none(self):
        assert M.time_of_day_profile(pd.DataFrame()) is None


# ── Liquidity sweeps ─────────────────────────────────────────────────────────
class TestSweeps:
    def _walk(self, n=800, seed=3):
        rng = np.random.default_rng(seed)
        close = 100 + np.cumsum(rng.normal(0, 1, n))
        high = close + rng.random(n)
        low = close - rng.random(n)
        return _daily(close, close, high=high, low=low)

    def test_random_walk_reversal_is_indistinguishable_from_chance(self):
        """
        The null. On a random walk the reversal rate must not clear the
        confidence bound — if it did, the study would be manufacturing signal.
        """
        out = M.liquidity_sweeps(self._walk())
        for key in ("downside", "upside"):
            st = out[key]
            if st.n_events >= 20:
                assert not st.is_better_than_chance
                assert st.ci_low <= 0.50 <= st.ci_high or "coin flip" in st.verdict

    def test_a_constructed_mean_reverter_reads_above_chance(self):
        """
        Build a series that genuinely bounces after piercing its low, and the
        study should see it. This is the positive control for the null above.

        Events are placed on a fixed 30-bar cadence — wider than the 20-bar
        lookback — so the pierce from one event has rolled out of the prior-low
        window before the next, leaving each sweep to undercut a flat base of
        100. A tighter cadence would let an earlier pierce sit inside the
        lookback, so the next "pierce" would not actually be a new low and no
        event would register.
        """
        n = 900
        close = np.full(n, 100.0)
        low = close.copy()
        high = close + 0.5
        for i in range(25, n - 6, 30):
            low[i] = 100.0 - 2.0               # pierce the flat prior low
            close[i] = 100.5                   # close back above it
            for h in range(1, 6):
                close[i + h] = 100.5 + h * 0.4  # then rise over the horizon
            # Return to the base well before the next event so lows stay flat.
            for j in range(i + 6, min(i + 30, n)):
                close[j] = 100.0
        df = _daily(close, close, high=high, low=low)
        st = M.liquidity_sweeps(df, lookback=20, horizon=5)["downside"]
        assert st.n_events >= 20
        assert st.reversal_rate > 0.6
        assert st.is_better_than_chance

    def test_short_history_returns_none(self):
        assert M.liquidity_sweeps(self._walk(n=10)) is None

    def test_missing_columns_returns_none(self):
        assert M.liquidity_sweeps(pd.DataFrame({"Close": range(200)})) is None

    def test_verdict_language_matches_the_interval(self):
        out = M.liquidity_sweeps(self._walk())
        for st in out.values():
            if st.n_events < 20:
                assert "Too few" in st.verdict
            elif st.ci_low > 0.50:
                assert "above chance" in st.verdict
            elif st.ci_high < 0.50:
                assert "continue" in st.verdict
            else:
                assert "coin flip" in st.verdict


# ── Round-number magnetism ───────────────────────────────────────────────────
class TestRoundNumbers:
    def test_uniform_prices_show_no_magnetism(self):
        """Evenly spread closes must not read as clustered."""
        rng = np.random.default_rng(2)
        close = rng.uniform(100, 110, 2000)
        study = M.round_number_magnetism(_daily(close, close))
        assert study is not None
        assert not study.is_magnetic
        assert abs(study.excess) < 3.0

    def test_clustered_prices_show_magnetism(self):
        """Closes snapped toward round dollars should exceed the baseline."""
        rng = np.random.default_rng(4)
        base = rng.uniform(100, 110, 2000)
        rounded = np.round(base)
        # 70% sit exactly on the dollar, 30% scattered.
        mask = rng.random(2000) < 0.70
        close = np.where(mask, rounded, base)
        study = M.round_number_magnetism(_daily(close, close))
        assert study.excess > 3.0
        assert study.is_magnetic

    def test_short_history_returns_none(self):
        assert M.round_number_magnetism(_daily(np.arange(50.0), np.arange(50.0))) is None

    def test_missing_close_returns_none(self):
        assert M.round_number_magnetism(pd.DataFrame({"Open": range(200)})) is None
