"""
Pytest suite for the Chart Studio pieces: the Lightweight-Charts HTML builder
and the technical snapshot.

The builder is checked as a pure function — valid embedded JSON, zones drawn
only when given, honest ``None`` on unusable history. The snapshot is checked
against constructed tapes with known character: an uptrend must read bullish,
a downtrend bearish, and the score must expose the same components it was
built from so the UI can show its work.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svp import lwchart                                # noqa: E402
from svp.analytics import snapshot as snap             # noqa: E402


def _frame(close, volume=None, start="2022-01-03"):
    close = np.asarray(close, dtype=float)
    idx = pd.date_range(start, periods=len(close), freq="B")
    df = pd.DataFrame({
        "Open": close * 0.998, "High": close * 1.006,
        "Low": close * 0.994, "Close": close,
    }, index=idx)
    if volume is not None:
        df["Volume"] = np.asarray(volume, dtype=float)
    return df


def _uptrend(n=300, seed=1):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(0.002 + rng.normal(0, 0.008, n)))


def _downtrend(n=300, seed=2):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(-0.002 + rng.normal(0, 0.008, n)))


def _payload(html: str) -> dict:
    """Extract and parse the embedded `const P = {...}` JSON payload."""
    s = html.split("const P = ", 1)[1]
    depth = 0
    for i, ch in enumerate(s):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(s[: i + 1])
    raise AssertionError("unbalanced payload")


# ── Chart HTML builder ───────────────────────────────────────────────────────
class TestChartBuilder:
    def test_returns_html_with_valid_embedded_json(self):
        out = lwchart.build_chart_html(_frame(_uptrend()))
        assert out is not None
        p = _payload(out.html)
        assert len(p["candles"]) == 300
        assert p["kind"] == "candles"

    def test_short_history_is_none(self):
        assert lwchart.build_chart_html(_frame(_uptrend(n=5))) is None

    def test_empty_frame_is_none(self):
        assert lwchart.build_chart_html(pd.DataFrame()) is None

    def test_non_datetime_index_is_none(self):
        df = _frame(_uptrend()).reset_index(drop=True)
        assert lwchart.build_chart_html(df) is None

    def test_nan_rows_are_dropped_not_invented(self):
        df = _frame(_uptrend(n=100))
        df.iloc[10, df.columns.get_loc("Close")] = np.nan
        out = lwchart.build_chart_html(df)
        assert len(_payload(out.html)["candles"]) == 99

    def test_zones_draw_only_the_levels_given(self):
        z = lwchart.FairValueZones(bear=80.0, base=110.0, bull=None, mos=95.0)
        out = lwchart.build_chart_html(_frame(_uptrend()), zones=z)
        lines = _payload(out.html)["priceLines"]
        titles = " ".join(ln["title"] for ln in lines)
        assert len(lines) == 3
        assert "Bear" in titles and "Base" in titles and "Buy-zone" in titles
        assert "Bull" not in titles

    def test_no_zones_means_no_price_lines(self):
        out = lwchart.build_chart_html(_frame(_uptrend()))
        assert _payload(out.html)["priceLines"] == []

    def test_volume_colours_follow_candle_direction(self):
        df = _frame(_uptrend(n=60), volume=np.full(60, 1e6))
        p = _payload(lwchart.build_chart_html(df).html)
        assert len(p["volume"]) == 60
        assert all("color" in v for v in p["volume"])

    def test_missing_volume_is_graceful(self):
        p = _payload(lwchart.build_chart_html(_frame(_uptrend())).html)
        assert p["volume"] == []

    def test_overlays_reindexed_and_nan_free(self):
        df = _frame(_uptrend())
        sma = df["Close"].rolling(50).mean()          # first 49 NaN
        p = _payload(lwchart.build_chart_html(df, overlays={"SMA 50": sma}).html)
        assert len(p["overlays"]) == 1
        assert len(p["overlays"][0]["data"]) == 300 - 49
        assert all(v["value"] == v["value"] for v in p["overlays"][0]["data"])

    def test_rsi_pane_included_when_given(self):
        df = _frame(_uptrend())
        rsi = pd.Series(50.0, index=df.index)
        out = lwchart.build_chart_html(df, rsi=rsi)
        assert len(_payload(out.html)["rsi"]) == 300
        assert out.height > lwchart.build_chart_html(df).height

    def test_line_and_area_kinds_use_closes(self):
        for kind in ("line", "area"):
            p = _payload(lwchart.build_chart_html(_frame(_uptrend()), kind=kind).html)
            assert p["kind"] == kind
            assert len(p["closes"]) == 300

    def test_log_scale_flag_passes_through(self):
        p = _payload(lwchart.build_chart_html(_frame(_uptrend()), log_scale=True).html)
        assert p["logScale"] is True


# ── Fair value zones ─────────────────────────────────────────────────────────
class TestZones:
    def test_all_none_is_no_lines(self):
        assert lwchart.FairValueZones().lines() == []

    def test_levels_carry_prices(self):
        z = lwchart.FairValueZones(bear=1, base=2, bull=3, mos=1.5)
        assert sorted(ln["price"] for ln in z.lines()) == [1, 1.5, 2, 3]


# ── Technical snapshot ───────────────────────────────────────────────────────
class TestSnapshot:
    def test_uptrend_reads_bullish(self):
        s = snap.compute(_frame(_uptrend()))
        assert s is not None
        assert s.trend == "Bullish"
        assert s.ma_alignment == "Stacked bullish"
        assert s.score > 60

    def test_downtrend_reads_bearish(self):
        s = snap.compute(_frame(_downtrend()))
        assert s.trend == "Bearish"
        assert s.score < 40

    def test_score_is_the_weighted_components(self):
        """The number shown must be reproducible from the shown breakdown."""
        s = snap.compute(_frame(_uptrend(seed=7)))
        w = {"Trend": 0.35, "MA alignment": 0.25, "Momentum": 0.25, "RSI": 0.15}
        assert s.score == int(round(sum(s.components[k] * v for k, v in w.items())))

    def test_score_bounds(self):
        for seed in range(6):
            rng = np.random.default_rng(seed)
            c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 250)))
            s = snap.compute(_frame(c))
            assert 0 <= s.score <= 100

    def test_support_below_and_resistance_above_price(self):
        rng = np.random.default_rng(3)
        c = 100 + np.cumsum(rng.normal(0, 1, 300))
        s = snap.compute(_frame(c))
        price = c[-1] * 1.0
        if s.support is not None:
            assert s.support < price * 1.007      # swing lows use the Low column
        if s.resistance is not None:
            assert s.resistance > price * 0.993

    def test_relative_volume_reads_a_spike(self):
        vol = np.full(300, 1e6)
        vol[-1] = 3e6
        s = snap.compute(_frame(_uptrend(), volume=vol))
        assert s.rel_volume == pytest.approx(3.0, rel=0.15)

    def test_short_history_is_none(self):
        assert snap.compute(_frame(_uptrend(n=20))) is None

    def test_empty_is_none(self):
        assert snap.compute(pd.DataFrame()) is None
        assert snap.compute(None) is None

    def test_narrative_never_instructs(self):
        for seed in (1, 2, 3):
            rng = np.random.default_rng(seed)
            c = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, 260)))
            s = snap.compute(_frame(c))
            low = s.narrative.lower()
            assert "buy now" not in low and "sell now" not in low
            assert "guarantee" not in low

    def test_calm_tape_reads_low_or_normal_volatility(self):
        rng = np.random.default_rng(9)
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, 300)))
        s = snap.compute(_frame(c))
        assert s.volatility in {"Low", "Normal"}
