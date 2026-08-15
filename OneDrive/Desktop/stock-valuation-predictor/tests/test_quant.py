"""
Pytest suite for the advanced quant analytics.

Each estimator is checked against a series with a *known* property — a random
walk must read as a random walk, a mean-reverting series as mean-reverting, a
persistent one as trending — because the whole value of these tools is that they
report the truth about the series rather than inventing structure. The
persistence fixtures are built from the returns' autocorrelation, not from drift:
a constant drift with iid noise is still a random walk in its differences, so
testing "trending" with drift would pass a broken estimator.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svp.analytics import quant as Q          # noqa: E402


def _random_walk(n=2000, vol=0.01, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, vol, n))))


def _persistent(n=4000, phi=0.6, seed=1):
    """Returns with strong positive autocorrelation → Hurst > 0.5 (momentum)."""
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = phi * r[i - 1] + rng.normal(0, 0.01)
    return pd.Series(100 * np.exp(np.cumsum(r)))


def _mean_reverting(n=3000, seed=2):
    """Ornstein-Uhlenbeck around a level → Hurst < 0.5."""
    rng = np.random.default_rng(seed)
    x = [np.log(100.0)]
    for _ in range(n):
        x.append(x[-1] + 0.4 * (np.log(100.0) - x[-1]) + rng.normal(0, 0.01))
    return pd.Series(np.exp(x))


# ── Volatility ───────────────────────────────────────────────────────────────
class TestVolatility:
    def test_realized_matches_hand_computation(self):
        s = _random_walk(seed=5)
        r = np.diff(np.log(s.to_numpy()))
        expected = float(np.std(r, ddof=1)) * np.sqrt(Q.CRYPTO_YEAR)
        assert Q.volatility(s).realized == pytest.approx(expected, rel=1e-9)

    def test_higher_vol_series_reads_higher(self):
        lo = Q.volatility(_random_walk(vol=0.005, seed=1)).realized
        hi = Q.volatility(_random_walk(vol=0.03, seed=1)).realized
        assert hi > lo

    def test_crypto_annualiser_exceeds_equity(self):
        """365 vs 252 — a crypto series is scaled up relative to an equity one."""
        s = _random_walk(seed=3)
        assert Q.volatility(s, annualiser=Q.CRYPTO_YEAR).realized > \
            Q.volatility(s, annualiser=Q.EQUITY_YEAR).realized

    def test_ewma_reacts_to_a_late_spike(self):
        rng = np.random.default_rng(9)
        calm = rng.normal(0, 0.005, 400)
        spike = rng.normal(0, 0.05, 40)
        s = pd.Series(100 * np.exp(np.cumsum(np.concatenate([calm, spike]))))
        vp = Q.volatility(s)
        # EWMA weights the recent spike, so it should exceed the equal-weighted
        # realized figure that is still diluting it with the long calm stretch.
        assert vp.ewma > vp.realized

    def test_short_series_is_none(self):
        vp = Q.volatility(pd.Series([100.0, 101.0]))
        assert vp.realized is None and vp.ewma is None


# ── Regime ───────────────────────────────────────────────────────────────────
class TestVolRegime:
    def test_late_spike_reads_stressed(self):
        rng = np.random.default_rng(4)
        s = pd.Series(100 * np.exp(np.cumsum(np.concatenate([
            rng.normal(0, 0.008, 300), rng.normal(0, 0.05, 30)]))))
        reg = Q.vol_regime(s)
        assert reg is not None
        assert reg.label in {"Elevated", "Stressed"}
        assert reg.percentile > 60

    def test_a_series_that_ends_calm_is_not_stressed(self):
        """
        High vol early, calm late — the *recent* window is quiet, so the regime
        must read low. A plain random walk can end in a high-vol window by
        chance, which is why the fixture makes the ending deterministically calm.
        """
        rng = np.random.default_rng(6)
        s = pd.Series(100 * np.exp(np.cumsum(np.concatenate([
            rng.normal(0, 0.04, 250), rng.normal(0, 0.004, 120)]))))
        reg = Q.vol_regime(s)
        assert reg.label in {"Calm", "Normal"}
        assert reg.percentile < 60

    def test_short_series_is_none(self):
        assert Q.vol_regime(_random_walk(n=20)) is None

    def test_percentile_is_bounded(self):
        reg = Q.vol_regime(_random_walk(seed=7))
        assert 0.0 <= reg.percentile <= 100.0


# ── Hurst ────────────────────────────────────────────────────────────────────
class TestHurst:
    def test_random_walk_is_near_half(self):
        h = Q.hurst_exponent(_random_walk(seed=0)).exponent
        assert 0.42 <= h <= 0.58

    def test_persistent_series_trends(self):
        res = Q.hurst_exponent(_persistent())
        assert res.exponent > 0.55
        assert res.is_trending
        assert "rend" in res.reading

    def test_mean_reverting_series_reverts(self):
        res = Q.hurst_exponent(_mean_reverting())
        assert res.exponent < 0.45
        assert res.is_mean_reverting
        assert "evert" in res.reading

    def test_exponent_is_clamped_to_unit_interval(self):
        for seed in range(5):
            h = Q.hurst_exponent(_persistent(seed=seed)).exponent
            assert 0.0 <= h <= 1.0

    def test_short_series_is_none(self):
        res = Q.hurst_exponent(_random_walk(n=30))
        assert res.exponent is None
        assert "Not enough" in res.reading

    def test_random_walk_is_neither_trend_nor_revert(self):
        res = Q.hurst_exponent(_random_walk(seed=11))
        assert not (res.is_trending and res.is_mean_reverting)
