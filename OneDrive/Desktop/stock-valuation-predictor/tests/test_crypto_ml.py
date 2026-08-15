"""
Pytest suite for the crypto directional model.

The single most important test is the null: on a random walk the model must NOT
claim an edge. That is the property that makes the whole thing safe to ship — a
directional model that fabricates skill on noise is worse than none. The
threshold logic (beats-baseline, verdict wording) is unit-tested directly on the
dataclass so those assertions are deterministic rather than riding on a trained
model's exact fold splits.

The feature builder is checked for the thing that silently invalidates every
backtest: lookahead. Every feature must be computable from bars at or before its
own row.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svp.models import crypto_ml as ML          # noqa: E402


def _frame(close, seed=0):
    rng = np.random.default_rng(seed)
    close = np.asarray(close, dtype=float)
    n = len(close)
    return pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99,
         "Close": close, "Volume": rng.integers(1_000_000, 5_000_000, n)},
        index=pd.date_range("2021-01-01", periods=n, freq="D"),
    )


def _random_walk(n=1200, vol=0.02, seed=0):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(0, vol, n)))


# ── The null: no edge on noise ───────────────────────────────────────────────
class TestNoEdgeOnNoise:
    def test_random_walk_does_not_beat_baseline(self):
        """The load-bearing guarantee of the whole module."""
        f = ML.evaluate(_frame(_random_walk(seed=0)))
        assert f is not None
        assert not f.beats_baseline

    def test_random_walk_hit_rate_is_near_half(self):
        f = ML.evaluate(_frame(_random_walk(seed=1)))
        assert 0.42 <= f.hit_rate <= 0.58

    def test_random_walk_verdict_admits_no_edge(self):
        f = ML.evaluate(_frame(_random_walk(seed=2)))
        assert any(w in f.verdict for w in ("no measurable edge", "no better",
                                            "coin flip", "worse than"))

    def test_no_actionable_call_without_an_edge(self):
        f = ML.evaluate(_frame(_random_walk(seed=3)))
        assert f.call == "No actionable call"

    @pytest.mark.parametrize("seed", range(4))
    def test_null_holds_across_seeds(self, seed):
        """A single lucky fold must not flip the verdict; check several draws."""
        f = ML.evaluate(_frame(_random_walk(seed=seed)))
        assert not f.beats_baseline


# ── Determinism & output shape ───────────────────────────────────────────────
class TestEvaluate:
    def test_is_deterministic(self):
        df = _frame(_random_walk(seed=5))
        a, b = ML.evaluate(df), ML.evaluate(df)
        assert a.hit_rate == b.hit_rate and a.n_oos == b.n_oos

    def test_prob_up_is_a_probability(self):
        f = ML.evaluate(_frame(_random_walk(seed=6)))
        assert f.prob_up is None or 0.0 <= f.prob_up <= 1.0

    def test_short_history_returns_none(self):
        assert ML.evaluate(_frame(_random_walk(n=100))) is None

    def test_empty_frame_returns_none(self):
        assert ML.evaluate(pd.DataFrame()) is None

    def test_ci_brackets_the_hit_rate(self):
        f = ML.evaluate(_frame(_random_walk(seed=7)))
        assert f.ci_low <= f.hit_rate <= f.ci_high


# ── Feature builder: no lookahead ────────────────────────────────────────────
class TestFeatures:
    def test_expected_columns_present(self):
        feats = ML.build_features(_frame(_random_walk(n=300)))
        for col in ("mom_1", "mom_5", "vol_14", "rsi_14", "vol_z", "range"):
            assert col in feats.columns

    def test_no_lookahead(self):
        """
        A feature at row t must not change when future rows are altered. Perturb
        the tail and confirm an early row's features are untouched — the classic
        backtest-invalidating leak.
        """
        base = _frame(_random_walk(n=400, seed=8))
        feats_full = ML.build_features(base)

        truncated = ML.build_features(base.iloc[:200])
        row = 150
        a = feats_full.iloc[row].dropna()
        b = truncated.iloc[row].dropna()
        common = a.index.intersection(b.index)
        assert len(common) > 3
        assert np.allclose(a[common].to_numpy(), b[common].to_numpy(), equal_nan=True)

    def test_missing_volume_is_tolerated(self):
        df = _frame(_random_walk(n=300)).drop(columns="Volume")
        feats = ML.build_features(df)
        assert not feats.empty
        assert "vol_z" not in feats.columns

    def test_empty_input(self):
        assert ML.build_features(pd.DataFrame()).empty


# ── Threshold logic (deterministic, model-free) ──────────────────────────────
class TestForecastLogic:
    def _fc(self, **kw):
        base = dict(n_oos=200, hit_rate=0.55, ci_low=0.52, ci_high=0.58,
                    baseline_rate=0.50, horizon=3, prob_up=0.7)
        base.update(kw)
        return ML.CryptoForecast(**base)

    def test_beats_baseline_needs_the_interval_above_it(self):
        assert self._fc(ci_low=0.53, baseline_rate=0.50).beats_baseline
        assert not self._fc(ci_low=0.49, baseline_rate=0.50).beats_baseline

    def test_small_sample_never_beats_baseline(self):
        assert not self._fc(n_oos=20, ci_low=0.9, baseline_rate=0.5).beats_baseline

    def test_a_strong_baseline_denies_a_weak_model(self):
        """55% accuracy is no achievement if 'always up' also scored 55%."""
        fc = self._fc(hit_rate=0.55, ci_low=0.53, baseline_rate=0.60)
        assert not fc.beats_baseline
        assert "no measurable edge" in fc.verdict or "no better" in fc.verdict

    def test_verdict_flags_a_sub_chance_model(self):
        assert "worse than a coin flip" in self._fc(
            hit_rate=0.44, ci_low=0.40, ci_high=0.48, baseline_rate=0.50).verdict

    def test_verdict_reports_a_real_edge(self):
        assert "edge over the naive baseline" in self._fc(
            hit_rate=0.60, ci_low=0.55, ci_high=0.65, baseline_rate=0.50).verdict

    def test_call_is_gated_by_beating_the_baseline(self):
        assert self._fc(ci_low=0.55, baseline_rate=0.50, prob_up=0.8).call == "Up"
        assert self._fc(ci_low=0.55, baseline_rate=0.50, prob_up=0.2).call == "Down"
        assert self._fc(ci_low=0.49, baseline_rate=0.50).call == "No actionable call"

    def test_edge_points(self):
        assert self._fc(hit_rate=0.57, baseline_rate=0.50).edge_pts == pytest.approx(7.0)
