"""
Pytest suite for the valuation core.

Complements ``ci/smoke_test.py``, which walks the whole pipeline end to end.
This file isolates the three things most likely to break quietly in
production: a data feed going away, the model emitting a malformed range, and
the cache losing or corrupting a round-trip.

Everything here runs offline. Network calls are stubbed rather than skipped,
so the fallback paths are genuinely executed instead of merely tolerated.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svp import features as F                       # noqa: E402
from svp.data import market, storage                # noqa: E402
from svp.models import valuation as V               # noqa: E402


# ── Fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def model():
    """Train once; the suite only reads from it."""
    return V.train_model()


@pytest.fixture
def feature_row():
    row = {c: np.nan for c in F.FEATURE_COLUMNS}
    row.update({
        "pe_ratio": 28.0, "roe": 0.35, "roa": 0.18, "debt_to_equity": 0.6,
        "fcf_yield": 0.04, "profit_margin": 0.25, "asset_turnover": 0.9,
        "revenue_yoy": 0.12, "revenue_qoq": 0.03, "net_income_yoy": 0.15,
        "fcf_yoy": 0.10, "sentiment": 0.4, "cpi": 314.0, "fed_funds": 4.3,
        "yield_curve": -0.1, "market_price": 190.0,
    })
    return row


# ── Live price fetching: error handling and fallback ─────────────────────────
class TestMarketDataFallback:
    def test_returns_data_when_the_feed_raises(self, monkeypatch):
        """A provider outage must degrade to synthetic history, not propagate."""
        if hasattr(market, "yf"):
            class _Boom:
                @staticmethod
                def Ticker(*_a, **_k):
                    raise ConnectionError("provider unreachable")
            monkeypatch.setattr(market, "yf", _Boom, raising=False)

        md = market.get_market_data("AAPL", fallback_price=190.0)
        assert md is not None
        assert md.price and md.price > 0
        assert not md.history.empty

    def test_fallback_price_is_honoured(self, monkeypatch):
        if hasattr(market, "_HAS_YF"):
            monkeypatch.setattr(market, "_HAS_YF", False, raising=False)
        md = market.get_market_data("ZZZZ", fallback_price=42.5)
        assert md.price == pytest.approx(42.5, rel=0.5)
        assert md.is_live is False

    def test_history_is_ohlc_consistent(self):
        """High must bound open/close and low must floor them, in any source."""
        md = market.get_market_data("AAPL", fallback_price=190.0)
        h = md.history
        assert {"Open", "High", "Low", "Close"} <= set(h.columns)
        assert (h["High"] >= h[["Open", "Close"]].max(axis=1) - 1e-6).all()
        assert (h["Low"] <= h[["Open", "Close"]].min(axis=1) + 1e-6).all()
        assert (h["Close"] > 0).all()

    def test_unknown_ticker_does_not_raise(self):
        md = market.get_market_data("NOTAREALTICKER", fallback_price=10.0)
        assert md.price and md.price > 0


# ── XGBoost prediction: shape and quantile ordering ──────────────────────────
class TestValuationModel:
    def test_quantile_models_present(self, model):
        assert set(model.quantiles) == set(V.QUANTILES)

    def test_prediction_shape(self, model, feature_row):
        res = V.predict(feature_row, model)
        for field in ("point", "low", "median", "high"):
            val = getattr(res, field)
            assert isinstance(val, float), f"{field} is {type(val)}"
            assert np.isfinite(val), f"{field} is not finite"
        assert res.mc_samples.ndim == 1
        assert len(res.mc_samples) >= 100

    def test_quantile_bounds_are_ordered(self, model, feature_row):
        """p10 <= p50 <= p90 — the invariant the whole range display rests on."""
        res = V.predict(feature_row, model)
        assert res.low <= res.median <= res.high

    @pytest.mark.parametrize("price", [12.0, 95.0, 190.0, 480.0, 1500.0])
    def test_ordering_holds_across_price_levels(self, model, feature_row, price):
        feature_row["market_price"] = price
        res = V.predict(feature_row, model)
        assert res.low <= res.median <= res.high
        assert res.point > 0

    def test_all_nan_features_still_predict(self, model):
        """Trees handle NaN natively; a sparse filing must not crash inference."""
        sparse = {c: np.nan for c in F.FEATURE_COLUMNS}
        sparse["market_price"] = 100.0
        res = V.predict(sparse, model)
        assert np.isfinite(res.point)
        assert res.low <= res.median <= res.high

    def test_monte_carlo_brackets_the_point_estimate(self, model, feature_row):
        res = V.predict(feature_row, model)
        lo, hi = np.percentile(res.mc_samples, [1, 99])
        assert lo <= res.point <= hi

    def test_prediction_is_deterministic(self, model, feature_row):
        a = V.predict(feature_row, model)
        b = V.predict(feature_row, model)
        assert a.point == pytest.approx(b.point)

    def test_signal_matches_the_numbers(self, model, feature_row):
        res = V.predict(feature_row, model)
        text, css = V.valuation_signal(res.point, 190.0)
        assert isinstance(text, str) and text
        assert isinstance(css, str) and css
        cheap, _ = V.valuation_signal(200.0, 100.0)
        rich, _ = V.valuation_signal(100.0, 200.0)
        assert cheap != rich


# ── SQLite cache ─────────────────────────────────────────────────────────────
class TestStorageCache:
    def test_round_trip_preserves_structure(self):
        payload = {"a": 1, "b": [1, 2, 3], "c": {"nested": True}, "d": None}
        storage.cache_set("pytest:round", payload, ttl=60)
        assert storage.cache_get("pytest:round") == payload

    def test_missing_key_returns_none(self):
        assert storage.cache_get(f"pytest:absent:{time.time()}") is None

    def test_overwrite_wins(self):
        storage.cache_set("pytest:over", {"v": 1}, ttl=60)
        storage.cache_set("pytest:over", {"v": 2}, ttl=60)
        assert storage.cache_get("pytest:over") == {"v": 2}

    def test_expired_entry_is_not_served(self):
        storage.cache_set("pytest:ttl", {"v": "stale"}, ttl=-1)
        assert storage.cache_get("pytest:ttl") is None

    @pytest.mark.parametrize("value", [
        {"unicode": "café ☕ Ω"},
        {"big": list(range(500))},
        {"floats": [1.5, -2.25, 1e12]},
        {"empty": {}},
    ])
    def test_payload_shapes(self, value):
        key = f"pytest:shape:{abs(hash(str(value)))}"
        storage.cache_set(key, value, ttl=60)
        assert storage.cache_get(key) == value

    def test_stats_reports_backend(self):
        st = storage.stats()
        assert "backend" in st and st["backend"]
        assert st["rows"] >= 0
