"""
Pytest suite for the institutional-quant pipeline pieces that are actually
buildable here: fractional differentiation, VPIN via bulk-volume classification,
and a Gaussian-HMM regime over frac-diff features.

Each is checked against a property it must have by construction — frac-diff at
d=1 reproduces a first difference and trades memory for stationarity monotonically;
VPIN rises with one-sided flow and needs many buckets to roll over; the HMM labels
states by volatility so the names mean something. The point is not that these are
exotic — it is that they are correct, which the exotic-but-undeployable
alternatives (TFT, Neural ODE) could never be shown to be on this much data.
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

from svp.analytics import fracdiff as FD, flow as FL       # noqa: E402
from svp.models import crypto_regime as CR                 # noqa: E402


def _rw(n=2000, vol=0.02, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, vol, n))))


# ── Fractional differentiation ───────────────────────────────────────────────
class TestFracDiff:
    def test_first_weight_is_one(self):
        assert FD.ffd_weights(0.4)[-1] == pytest.approx(1.0)

    def test_weights_decay_below_threshold(self):
        w = FD.ffd_weights(0.4, thresh=1e-3)
        assert abs(w[0]) >= 1e-3          # oldest retained weight is above threshold
        assert len(FD.ffd_weights(0.4, thresh=1e-5)) > len(w)

    def test_d_zero_is_identity(self):
        s = _rw()
        assert FD.frac_diff(s, 0.0).equals(s)

    def test_d_one_matches_first_difference(self):
        s = np.log(_rw(seed=3))
        fd1 = FD.frac_diff(s, 1.0).dropna()
        diff1 = s.diff().dropna()
        common = fd1.index.intersection(diff1.index)
        assert np.corrcoef(fd1[common], diff1[common])[0, 1] > 0.999

    def test_memory_falls_as_d_rises(self):
        """The core trade-off: more differencing keeps less of the level's memory."""
        s = np.log(_rw(seed=1))
        mems = []
        for d in (0.2, 0.5, 0.9):
            fdv = FD.frac_diff(s, d).dropna()
            aligned = s.reindex(fdv.index)
            mems.append(abs(np.corrcoef(aligned, fdv)[0, 1]))
        assert mems[0] > mems[1] > mems[2]

    def test_min_d_reduces_autocorrelation(self):
        fit = FD.min_d(np.log(_rw(seed=2)))
        assert fit is not None
        assert abs(fit.acf1_after) < abs(fit.acf1_before)
        assert 0.0 <= fit.memory_retained <= 1.0

    def test_min_d_on_short_series_is_none(self):
        assert FD.min_d(_rw(n=40)) is None

    def test_window_wider_than_series_is_all_nan(self):
        s = _rw(n=5)
        assert FD.frac_diff(s, 0.3, thresh=1e-12).isna().all()


# ── VPIN ─────────────────────────────────────────────────────────────────────
class TestVPIN:
    def _frame(self, close, vol):
        return pd.DataFrame({"Close": np.asarray(close, float),
                             "Volume": np.asarray(vol, float)})

    def test_balanced_flow_is_not_toxic(self):
        rng = np.random.default_rng(1)
        n = 4000
        r = FL.vpin_bvc(self._frame(100 + np.cumsum(rng.normal(0, 0.5, n)),
                                    rng.uniform(1e5, 2e5, n)))
        assert r is not None
        assert not r.is_toxic
        assert 0.0 <= r.value <= 1.0

    def test_one_sided_flow_scores_higher_than_balanced_flow(self):
        """
        The load-bearing property: sustained one-directional flow (every bar the
        same sign) is maximally toxic and must produce a higher VPIN than
        symmetric two-sided noise. Comparing two series is the clean test;
        comparing a percentile inside one tail-heavy series is not.
        """
        rng = np.random.default_rng(2)
        n = 4000
        vol = rng.uniform(1e5, 2e5, n)

        balanced = 100 + np.cumsum(rng.normal(0, 0.5, n))
        one_sided = 100 + np.cumsum(np.abs(rng.normal(0.5, 0.1, n)))   # always up

        v_bal = FL.vpin_bvc(self._frame(balanced, vol)).value
        v_tox = FL.vpin_bvc(self._frame(one_sided, vol)).value
        assert v_tox > v_bal
        assert v_tox > 0.8        # near-perfectly one-sided ⇒ imbalance near 1

    def test_percentile_is_bounded(self):
        rng = np.random.default_rng(4)
        r = FL.vpin_bvc(self._frame(100 + np.cumsum(rng.normal(0, 0.5, 3000)),
                                    rng.uniform(1e5, 2e5, 3000)))
        assert 0.0 <= r.percentile <= 100.0

    def test_short_history_is_none(self):
        assert FL.vpin_bvc(self._frame([1, 2, 3], [1, 1, 1])) is None

    def test_missing_volume_is_none(self):
        assert FL.vpin_bvc(pd.DataFrame({"Close": range(500)})) is None

    def test_zero_volume_is_none(self):
        assert FL.vpin_bvc(self._frame(np.arange(600.0), np.zeros(600))) is None


# ── Gaussian HMM regime ──────────────────────────────────────────────────────
class TestRegime:
    def _multi_regime(self, seed=0):
        rng = np.random.default_rng(seed)
        calm = np.cumsum(rng.normal(0, 0.005, 400))
        trend = calm[-1] + np.cumsum(0.01 + rng.normal(0, 0.006, 400))
        stress = trend[-1] + np.cumsum(rng.normal(0, 0.05, 400))
        c = 100 * np.exp(np.concatenate([calm, trend, stress]))
        return pd.DataFrame({"Close": c},
                            index=pd.date_range("2021-01-01", periods=len(c), freq="D"))

    @pytest.mark.skipif(not CR._HMM_OK, reason="hmmlearn not installed")
    def test_detect_returns_a_labelled_state(self):
        r = CR.detect(self._multi_regime())
        assert r is not None
        assert r.label in {"Calm", "Trending", "Stressed"}
        assert 0.0 <= r.confidence <= 1.0
        assert 0.0 <= r.persistence <= 1.0

    @pytest.mark.skipif(not CR._HMM_OK, reason="hmmlearn not installed")
    def test_uses_a_fractional_difference_order(self):
        r = CR.detect(self._multi_regime())
        assert 0.0 <= r.d_used <= 1.0

    @pytest.mark.skipif(not CR._HMM_OK, reason="hmmlearn not installed")
    def test_reading_mentions_the_posterior(self):
        r = CR.detect(self._multi_regime())
        assert "posterior" in r.reading

    @pytest.mark.skipif(not CR._HMM_OK, reason="hmmlearn not installed")
    def test_is_deterministic(self):
        df = self._multi_regime(seed=5)
        assert CR.detect(df).label == CR.detect(df).label

    def test_short_history_is_none(self):
        assert CR.detect(pd.DataFrame({"Close": range(50)})) is None

    def test_missing_close_is_none(self):
        assert CR.detect(pd.DataFrame({"Open": range(500)})) is None

    def test_label_ordering_is_by_volatility(self):
        """Calm gets the lowest-vol state, Stressed the highest — names must map."""
        labels = CR._label_states(np.array([0.05, 0.01, 0.20]))
        assert labels[1] == "Calm"       # index 1 has the smallest vol mean
        assert labels[2] == "Stressed"   # index 2 has the largest
