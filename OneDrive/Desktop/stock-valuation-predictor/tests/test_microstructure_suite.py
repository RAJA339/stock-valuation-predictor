"""
Pytest suite for market structure, candlestick patterns, regime state and GEX.

Each module is pinned to a property it cannot fake. Structure must tell BOS
from MSS on a tape built to do both in a known order. The pattern engine must
*reject* three green candles that fail the volume or wick filters, since a
detector that accepts everything is what makes these patterns useless. The
regime classifier must call a synthetic trend a trend and a mean-reverting
series consolidation. Gamma must be symmetric between calls and puts at the
same strike, and net exposure must flip sign with the dealer convention.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svp.analytics import (                                   # noqa: E402
    amd as AMD, candles as CD, fvg as FV, regime_state as RS,
    structure as ST, volume_profile as VP,
)
from svp.models import gex as GEX                             # noqa: E402


def _ohlcv(rows, start="2021-01-04"):
    arr = np.array(rows, dtype=float)
    return pd.DataFrame(
        {"Open": arr[:, 0], "High": arr[:, 1], "Low": arr[:, 2],
         "Close": arr[:, 3],
         "Volume": arr[:, 4] if arr.shape[1] > 4 else np.full(len(arr), 1e6)},
        index=pd.date_range(start, periods=len(arr), freq="B"))


def _walk(n=400, seed=0, vol=0.012, drift=0.0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    return _ohlcv([(x, x * 1.008, x * 0.992, x, 1e6) for x in c])


# ── Timezone-aware data ──────────────────────────────────────────────────────
class TestTimezoneAwareInput:
    """
    Production data is tz-aware; the offline data CI falls back to is not.

    That gap shipped a real bug: the Structure Desk re-parsed an event's
    display date into a tz-naive Timestamp and compared it against a tz-aware
    price index, which raises. Every analytics module is therefore exercised
    against a tz-aware frame here — the shape yfinance actually returns for
    equities — so the suite stops testing a world the app never runs in.
    """

    @staticmethod
    def _aware(n=400, seed=3, tz="America/New_York"):
        rng = np.random.default_rng(seed)
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
        idx = pd.date_range("2023-01-03", periods=n, freq="B", tz=tz)
        return pd.DataFrame(
            {"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c,
             "Volume": np.full(n, 1e6)}, index=idx)

    def test_structure_events_carry_a_comparable_timestamp(self):
        """The exact failure: event stamps must compare against the index."""
        df = self._aware()
        sm = ST.analyse(df)
        assert sm is not None and sm.events
        x0, x1 = df.index[100], df.index[-1]
        for e in sm.events:
            assert e.ts is not None, "event lost its index label"
            # Would raise TypeError if ts were tz-naive.
            _ = x0 <= e.ts <= x1
        assert any(x0 <= e.ts <= x1 for e in sm.events)

    def test_event_ts_matches_the_frame_index(self):
        df = self._aware()
        sm = ST.analyse(df)
        for e in sm.events:
            assert e.ts in df.index

    def test_every_analytics_module_accepts_tz_aware_frames(self):
        df = self._aware(n=600, seed=5)
        assert ST.analyse(df) is not None
        assert CD.detect(df) is not None
        assert CD.anchored_vwap(df) is not None
        assert RS.classify(df) is not None
        assert VP.profile(df) is not None
        assert FV.detect(df) is not None
        assert AMD.detect(df) is not None

    def test_utc_and_naive_give_the_same_structure(self):
        """Timezone is presentation; the structure read must not depend on it."""
        aware = self._aware(seed=7)
        naive = aware.copy()
        naive.index = naive.index.tz_localize(None)
        a, n = ST.analyse(aware), ST.analyse(naive)
        assert [e.kind for e in a.events] == [e.kind for e in n.events]
        assert a.bias == n.bias


# ── P1: market structure ─────────────────────────────────────────────────────
class TestStructure:
    def test_swings_record_when_they_became_knowable(self):
        """A pivot is confirmed k bars later — never on the bar itself."""
        sw = ST.find_swings(_walk(seed=1), k=3)
        assert sw
        assert all(s.confirmed_idx == s.idx + 3 for s in sw)

    def test_first_break_is_a_shift_then_continuation_is_a_bos(self):
        """
        Built to break upward twice from a neutral start: the first break can
        only be MSS (no prior bullish bias), the next in the same direction
        must be BOS.
        """
        # Pivots need a genuine peak: a run of *identical* highs has no single
        # maximum, and find_swings is right to refuse to invent one, so every
        # pause here rises to a distinct high and falls away from it.
        def peak(base, height, n=7):
            """A small hump whose centre bar is the unique local high."""
            out = []
            for j in range(n):
                off = height * (1 - abs(j - n // 2) / (n // 2 + 0.5))
                px = base + off
                out.append((px, px + 0.3, px - 0.3, px, 1e6))
            return out

        rows = []
        for j in range(12):                       # base range, slight variation
            rows += [(100 + j * 0.01, 100.4 + j * 0.01, 99.6, 100, 1e6)]
        rows += peak(99.0, 1.5)                   # pivot high near 100.5
        for px in np.linspace(99, 112, 14):       # break up → MSS
            rows += [(px, px + 1, px - 1, px + 0.8, 3e6)]
        rows += peak(110.0, 3.0)                  # new pivot high near 113
        for px in np.linspace(112, 128, 14):      # break it again → BOS
            rows += [(px, px + 1, px - 1, px + 0.8, 3e6)]
        rows += [(128 + j * 0.01, 128.5, 127.5, 128, 1e6) for j in range(20)]

        sm = ST.analyse(_ohlcv(rows), k=3)
        assert sm is not None
        ups = [e for e in sm.events if e.direction == "bullish"]
        assert ups, "no bullish break detected"
        assert ups[0].kind == "MSS"
        assert any(e.kind == "BOS" for e in ups), "no continuation recognised"
        assert sm.bias == "bullish"

    def test_order_block_is_the_last_opposing_candle(self):
        sm = ST.analyse(_walk(seed=4), k=3)
        for b in sm.blocks:
            assert b.bottom < b.top
            assert b.kind in ("demand", "supply")

    def test_mitigation_is_tracked_and_counted(self):
        sm = ST.analyse(_walk(seed=5, n=600), k=3)
        for b in sm.blocks:
            if b.mitigated:
                assert b.touches >= 1
                assert b.mitigation_date
                assert "tested" in b.status
            else:
                assert b.status == "untested" and b.touches == 0

    def test_untested_blocks_are_a_subset_sorted_by_distance(self):
        sm = ST.analyse(_walk(seed=6, n=600), k=3)
        near = sm.untested_blocks(100.0, n=3)
        assert all(not b.mitigated for b in near)
        d = [abs(b.mid - 100.0) for b in near]
        assert d == sorted(d)

    def test_volume_expansion_is_measured_not_assumed(self):
        sm = ST.analyse(_walk(seed=7), k=3)
        for b in sm.blocks:
            assert b.impulse_volume_ratio > 0
            assert b.strength in ("high", "expanded", "ordinary")

    def test_reading_never_forecasts(self):
        sm = ST.analyse(_walk(seed=8))
        low = sm.reading(100.0).lower()
        assert "will " not in low and "buy" not in low and "sell" not in low

    def test_short_history_is_none(self):
        assert ST.analyse(_walk(n=20)) is None

    def test_missing_columns_is_none(self):
        assert ST.analyse(_walk().drop(columns=["Open"])) is None


# ── P2: candlestick patterns ─────────────────────────────────────────────────
class TestCandlePatterns:
    def _soldiers(self, volume=3e6, upper_wick=0.0):
        """Three textbook soldiers on rising volume, appended to a quiet base."""
        rows = [(100, 100.6, 99.4, 100, 1e6)] * 30
        px = 100.0
        for _ in range(3):
            o = px
            c = px + 3.0
            h = c + upper_wick
            lo = o - 0.05
            rows.append((o, h, lo, c, volume))
            px = c - 0.4              # next opens inside the previous body
        return _ohlcv(rows)

    def test_clean_soldiers_are_detected(self):
        ev = CD.detect(self._soldiers())
        assert ev, "clean three white soldiers not detected"
        e = ev[-1]
        assert e.kind == "three_white_soldiers"
        assert e.total_move_pct > 0
        assert min(e.volume_ratios) >= CD.VOLUME_MULTIPLE

    def test_low_volume_soldiers_are_rejected(self):
        """The filter that makes the pattern mean anything."""
        assert CD.detect(self._soldiers(volume=0.8e6)) == []

    def test_long_upper_wicks_are_rejected(self):
        """Closing far from the high is a fade, not a soldier."""
        assert CD.detect(self._soldiers(upper_wick=6.0)) == []

    def test_volume_filter_can_be_relaxed_explicitly(self):
        ev = CD.detect(self._soldiers(volume=0.8e6), require_volume=False)
        assert ev, "relaxing the volume filter should admit the pattern"

    def test_crows_mirror_soldiers(self):
        rows = [(100, 100.6, 99.4, 100, 1e6)] * 30
        px = 100.0
        for _ in range(3):
            o, c = px, px - 3.0
            rows.append((o, o + 0.05, c, c, 3e6))
            px = c + 0.4
        ev = CD.detect(_ohlcv(rows))
        assert ev and ev[-1].kind == "three_black_crows"
        assert ev[-1].direction == "bearish"

    def test_random_walk_yields_few_or_no_patterns(self):
        """With all filters on, noise should rarely qualify."""
        ev = CD.detect(_walk(n=500, seed=11))
        assert len(ev) <= 6

    def test_patterns_do_not_overlap(self):
        ev = CD.detect(_walk(n=800, seed=12), require_volume=False)
        for a, b in zip(ev, ev[1:]):
            assert b.start_idx > a.end_idx

    def test_short_history_is_none(self):
        assert CD.detect(_walk(n=10)) is None


class TestAnchoredVWAP:
    def test_vwap_of_a_flat_tape_is_that_price(self):
        df = _ohlcv([(100, 100, 100, 100, 1e6)] * 50)
        av = CD.anchored_vwap(df, anchor_idx=0)
        assert av.current == pytest.approx(100.0)
        assert av.upper1 .iloc[-1] == pytest.approx(100.0, abs=1e-6)

    def test_bands_widen_with_dispersion(self):
        calm = CD.anchored_vwap(_walk(seed=3, vol=0.004))
        wild = CD.anchored_vwap(_walk(seed=3, vol=0.03))
        calm_w = float(calm.upper2.iloc[-1] - calm.lower2.iloc[-1])
        wild_w = float(wild.upper2.iloc[-1] - wild.lower2.iloc[-1])
        assert wild_w > calm_w

    def test_bands_are_ordered(self):
        av = CD.anchored_vwap(_walk(seed=9))
        assert (av.lower2.iloc[-1] <= av.lower1.iloc[-1]
                <= av.vwap.iloc[-1] <= av.upper1.iloc[-1] <= av.upper2.iloc[-1])

    def test_anchor_moves_the_average(self):
        df = _walk(seed=10, drift=0.004, n=300)
        early = CD.anchored_vwap(df, anchor_idx=0).current
        late = CD.anchored_vwap(df, anchor_idx=250).current
        assert late > early          # anchored later in an uptrend

    def test_volume_weighting_actually_applies(self):
        """A heavy bar must pull the average toward its price."""
        rows = [(100, 100, 100, 100, 1e6)] * 10 + [(120, 120, 120, 120, 5e7)]
        av = CD.anchored_vwap(_ohlcv(rows))
        assert av.current > 112

    def test_empty_is_none(self):
        assert CD.anchored_vwap(pd.DataFrame()) is None


# ── P3: regime state ─────────────────────────────────────────────────────────
class TestHurstRS:
    def test_random_walk_is_near_half(self):
        h = RS.hurst_rs(_walk(n=1500, seed=2)["Close"])
        assert h is not None
        assert 0.38 <= h.exponent <= 0.62

    def test_persistent_series_reads_above_half(self):
        rng = np.random.default_rng(4)
        r = np.zeros(2000)
        for i in range(1, 2000):
            r[i] = 0.65 * r[i - 1] + rng.normal(0, 0.01)
        h = RS.hurst_rs(pd.Series(100 * np.exp(np.cumsum(r))))
        assert h.exponent > 0.55 and h.is_trending

    def test_mean_reverting_series_reads_below_half(self):
        rng = np.random.default_rng(5)
        x = [np.log(100.0)]
        for _ in range(2000):
            x.append(x[-1] + 0.5 * (np.log(100.0) - x[-1]) + rng.normal(0, .01))
        h = RS.hurst_rs(pd.Series(np.exp(x)))
        assert h.exponent < 0.45 and h.is_mean_reverting

    def test_fit_quality_is_reported(self):
        h = RS.hurst_rs(_walk(n=1200, seed=6)["Close"])
        assert 0.0 <= h.r_squared <= 1.0
        assert h.n_windows >= 4

    def test_short_series_is_none(self):
        assert RS.hurst_rs(pd.Series(np.arange(20.0))) is None


class TestRegimeClassifier:
    def test_strong_uptrend_classifies_bullish(self):
        rng = np.random.default_rng(7)
        c = 100 * np.exp(np.cumsum(0.006 + rng.normal(0, 0.004, 400)))
        st = RS.classify(_ohlcv([(x, x * 1.01, x * 0.99, x, 1e6) for x in c]))
        assert st is not None
        assert st.direction == "bullish"
        assert st.state in ("Strong Bull Trend", "Weak Bull")

    def test_strong_downtrend_classifies_bearish(self):
        rng = np.random.default_rng(8)
        c = 100 * np.exp(np.cumsum(-0.006 + rng.normal(0, 0.004, 400)))
        st = RS.classify(_ohlcv([(x, x * 1.01, x * 0.99, x, 1e6) for x in c]))
        assert st.direction == "bearish"

    def test_mean_reverting_tape_reads_sideways(self):
        rng = np.random.default_rng(9)
        x = [np.log(100.0)]
        for _ in range(500):
            x.append(x[-1] + 0.5 * (np.log(100.0) - x[-1]) + rng.normal(0, .01))
        c = np.exp(x)
        st = RS.classify(_ohlcv([(v, v * 1.01, v * 0.99, v, 1e6) for v in c]))
        assert st.state == "Sideways Consolidation"

    def test_state_is_one_of_the_five(self):
        for seed in range(5):
            st = RS.classify(_walk(n=300, seed=seed))
            assert st.state in ("Strong Bull Trend", "Weak Bull",
                                "Sideways Consolidation", "Weak Bear",
                                "Strong Bear Trend")

    def test_both_estimators_are_reported(self):
        st = RS.classify(_walk(n=600, seed=3))
        assert st.hurst_rs is not None and st.hurst_diff is not None
        assert isinstance(st.estimators_agree, bool)

    def test_narrative_mentions_adx_and_hurst(self):
        st = RS.classify(_walk(n=400, seed=2))
        assert "ADX" in st.narrative and "H ≈" in st.narrative

    def test_short_history_is_none(self):
        assert RS.classify(_walk(n=40)) is None


# ── P4: HVN ──────────────────────────────────────────────────────────────────
class TestHighVolumeNodes:
    def test_hvn_sits_where_volume_concentrated(self):
        rng = np.random.default_rng(1)
        shelf = np.full(200, 100.0) + rng.normal(0, 0.05, 200)
        tail = np.linspace(101, 110, 40)
        close = np.concatenate([shelf, tail])
        vol = np.concatenate([np.full(200, 5e6), np.full(40, 1e5)])
        df = _ohlcv([(c, c * 1.001, c * 0.999, c, v)
                     for c, v in zip(close, vol)])
        p = VP.profile(df)
        assert p.hvn, "no high-volume nodes found"
        assert min(abs(np.array(p.hvn) - 100.0)) < 1.5

    def test_hvn_and_lvn_do_not_overlap(self):
        p = VP.profile(_walk(n=400, seed=2))
        assert set(p.hvn).isdisjoint(set(p.lvn))

    def test_poc_is_among_the_high_volume_nodes(self):
        p = VP.profile(_walk(n=500, seed=3))
        if p.hvn:
            assert min(abs(np.array(p.hvn) - p.poc)) < (p.high - p.low)


# ── P5: gamma exposure ───────────────────────────────────────────────────────
class _Chain:
    def __init__(self, calls, puts, spot=100.0, expiry="2026-01-16"):
        self.calls, self.puts = calls, puts
        self.spot, self.expiry, self.source = spot, expiry, "test"


def _leg(strikes, oi, iv=0.25):
    return pd.DataFrame({"strike": strikes,
                         "openInterest": oi,
                         "impliedVolatility": [iv] * len(strikes)})


class TestGEX:
    def test_gamma_is_symmetric_between_calls_and_puts(self):
        """A textbook identity: same strike, same gamma, either side."""
        g = GEX.bs_gamma(100, 100, 0.25, 0.3)
        assert g is not None and g > 0

    def test_gamma_peaks_at_the_money(self):
        atm = GEX.bs_gamma(100, 100, 0.25, 0.3)
        otm = GEX.bs_gamma(100, 140, 0.25, 0.3)
        assert atm > otm

    def test_bad_inputs_return_none_not_zero(self):
        assert GEX.bs_gamma(0, 100, 0.25, 0.3) is None
        assert GEX.bs_gamma(100, 100, 0, 0.3) is None
        assert GEX.bs_gamma(100, 100, 0.25, 0) is None

    def test_call_heavy_chain_is_positive_gamma(self):
        ks = [90, 95, 100, 105, 110]
        chain = _Chain(_leg(ks, [5000] * 5), _leg(ks, [10] * 5))
        p = GEX.compute(chain, days_to_expiry=30)
        assert p is not None and p.is_positive_gamma
        assert "dampens" in p.regime

    def test_put_heavy_chain_is_negative_gamma(self):
        ks = [90, 95, 100, 105, 110]
        chain = _Chain(_leg(ks, [10] * 5), _leg(ks, [5000] * 5))
        p = GEX.compute(chain, days_to_expiry=30)
        assert not p.is_positive_gamma
        assert "amplifies" in p.regime

    def test_oi_walls_find_the_heaviest_strikes(self):
        ks = [90, 95, 100, 105, 110]
        chain = _Chain(_leg(ks, [10, 10, 10, 9000, 10]),
                       _leg(ks, [8000, 10, 10, 10, 10]))
        p = GEX.compute(chain, days_to_expiry=30)
        assert p.call_wall == 105
        assert p.put_wall == 90

    def test_zero_gamma_lies_within_the_strike_range(self):
        ks = [90, 95, 100, 105, 110]
        chain = _Chain(_leg(ks, [100, 200, 3000, 4000, 5000]),
                       _leg(ks, [5000, 4000, 3000, 200, 100]))
        p = GEX.compute(chain, days_to_expiry=30)
        if p.zero_gamma is not None:
            assert 90 <= p.zero_gamma <= 110

    def test_unusable_strikes_are_dropped_and_counted(self):
        calls = pd.DataFrame({"strike": [100, 105, 110],
                              "openInterest": [500, None, 0],
                              "impliedVolatility": [0.3, 0.3, 0.3]})
        chain = _Chain(calls, _leg([100, 105, 110, 115], [100] * 4))
        p = GEX.compute(chain, days_to_expiry=30)
        assert p.dropped >= 2

    def test_percentage_iv_is_normalised(self):
        ks = [95, 100, 105, 110]
        pct = GEX.compute(_Chain(_leg(ks, [1000] * 4, iv=25.0),
                                 _leg(ks, [10] * 4, iv=25.0)), 30)
        frac = GEX.compute(_Chain(_leg(ks, [1000] * 4, iv=0.25),
                                  _leg(ks, [10] * 4, iv=0.25)), 30)
        assert pct.total_gex == pytest.approx(frac.total_gex, rel=1e-6)

    def test_too_few_strikes_is_none(self):
        chain = _Chain(_leg([100], [10]), _leg([100], [10]))
        assert GEX.compute(chain, days_to_expiry=30) is None

    def test_none_chain_is_none(self):
        assert GEX.compute(None, days_to_expiry=30) is None

    def test_reading_states_the_dealer_assumption(self):
        ks = [90, 95, 100, 105, 110]
        p = GEX.compute(_Chain(_leg(ks, [1000] * 5), _leg(ks, [500] * 5)), 30)
        assert "not published" in p.reading()
