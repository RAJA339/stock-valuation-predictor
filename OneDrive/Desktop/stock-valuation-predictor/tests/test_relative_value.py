"""
Pytest suite for the real-data training set and the relative-value model.

These tests encode the P0-1 and P0-2 acceptance criteria as executable checks,
because the defect they replace was not a crash — it was a confident number
with nothing behind it, and that class of failure is invisible unless the
invariants are asserted. The most important test in the file is the simplest:
no feature may be a function of price.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svp.models import dataset as DS, relative_value as RV      # noqa: E402


def _fundamentals(margin=0.15, growth=0.10, rev=1e10, seed=0):
    rng = np.random.default_rng(seed)
    ni = rev * margin
    ocf = ni + rev * 0.05
    capex = rev * 0.04
    assets = rev * 1.5
    return {
        "revenue": rev, "revenue_prior": rev / (1 + growth),
        "net_income": ni, "net_income_prior": ni * 0.9,
        "assets": assets, "equity": assets * 0.45,
        "long_term_debt": assets * 0.2, "op_cash_flow": ocf, "capex": capex,
        "free_cash_flow": ocf - capex, "fcf_prior": (ocf - capex) * 0.9,
        "gross_profit": rev * 0.5, "ebit": ni * 1.3,
        "interest_expense": assets * 0.01,
        "current_assets": assets * 0.4, "current_liabilities": assets * 0.25,
        "cash": assets * 0.1, "total_debt": assets * 0.2,
        "market_cap": rev * 3.0, "shares": 1e8,
        "as_of": "2026-01-01", "_rng": rng,
    }


class TestNoPriceLeakage:
    """P0-1 acceptance criterion 1, as a test rather than a promise."""

    PRICE_WORDS = ("price", "market_cap", "marketcap", "ev_", "enterprise",
                   "pe_", "yield", "beta", "volatility", "multiple")

    def test_no_feature_name_suggests_price(self):
        for name in DS.FUNDAMENTAL_FEATURES:
            assert not any(w in name.lower() for w in self.PRICE_WORDS), name

    def test_the_three_removed_features_are_gone(self):
        for dead in ("market_price", "fcf_yield", "pe_ratio"):
            assert dead not in DS.FUNDAMENTAL_FEATURES

    def test_features_do_not_move_when_only_price_moves(self):
        """
        The decisive check. Two companies identical in every filing quantity
        but priced differently must produce identical features — that is what
        "no price leakage" means operationally.
        """
        cheap = _fundamentals()
        rich = dict(cheap, market_cap=cheap["market_cap"] * 10)
        assert (DS.features_from_fundamentals(cheap)
                == DS.features_from_fundamentals(rich))

    def test_but_the_target_does_move_with_price(self):
        """The target is the market multiple — it *must* respond to price."""
        cheap = _fundamentals()
        y_cheap = DS.target_from_market(
            DS.enterprise_value(cheap["market_cap"], cheap["total_debt"],
                                cheap["cash"]), cheap["revenue"])
        y_rich = DS.target_from_market(
            DS.enterprise_value(cheap["market_cap"] * 4, cheap["total_debt"],
                                cheap["cash"]), cheap["revenue"])
        assert y_rich > y_cheap


class TestTarget:
    def test_target_is_scale_free(self):
        """A $9 stock and a $900 stock with the same multiple are one problem."""
        small = DS.target_from_market(2e9, 1e9)
        large = DS.target_from_market(2e12, 1e12)
        assert small == pytest.approx(large)

    def test_target_is_a_log_multiple(self):
        assert DS.target_from_market(3e9, 1e9) == pytest.approx(math.log(3.0))

    def test_absurd_multiples_are_refused(self):
        assert DS.target_from_market(1e15, 1e9) is None        # 1,000,000x
        assert DS.target_from_market(1e6, 1e9) is None         # 0.001x

    def test_missing_or_negative_revenue_is_none(self):
        assert DS.target_from_market(1e9, 0) is None
        assert DS.target_from_market(1e9, -5e8) is None
        assert DS.target_from_market(None, 1e9) is None

    def test_enterprise_value_arithmetic(self):
        assert DS.enterprise_value(100, 40, 15) == pytest.approx(125)
        assert DS.enterprise_value(0, 10, 5) is None


class TestFeatureConstruction:
    def test_growth_from_a_negative_base_is_undefined(self):
        """
        Earnings from −100 to −50 is not "+50% growth". A number there would
        train the model on a quantity with no interpretation.
        """
        f = _fundamentals()
        f["net_income_prior"] = -1e9
        assert math.isnan(DS.features_from_fundamentals(f)["net_income_yoy"])

    def test_zero_denominators_give_nan_not_infinity(self):
        f = _fundamentals()
        f["equity"] = 0.0
        f["revenue"] = 0.0
        out = DS.features_from_fundamentals(f)
        assert all(not math.isinf(v) for v in out.values())
        assert math.isnan(out["roe"])

    def test_every_declared_feature_is_produced(self):
        out = DS.features_from_fundamentals(_fundamentals())
        assert set(out) == set(DS.FUNDAMENTAL_FEATURES)

    def test_accruals_follow_sloan(self):
        f = _fundamentals()
        expected = (f["net_income"] - f["op_cash_flow"]) / f["assets"]
        assert DS.features_from_fundamentals(f)["accruals"] == \
            pytest.approx(expected)

    def test_log_assets_replaces_market_cap_as_the_size_control(self):
        out = DS.features_from_fundamentals(_fundamentals(rev=1e10))
        assert out["log_assets"] == pytest.approx(math.log(1.5e10))


class TestBuild:
    def _fetch(self, seed=0):
        rng = np.random.default_rng(seed)

        def fetch(ticker):
            m = rng.uniform(-0.3, 0.3)
            g = rng.uniform(-0.2, 0.4)
            f = _fundamentals(margin=m, growth=g, rev=10 ** rng.uniform(8, 11))
            f["market_cap"] = f["revenue"] * float(np.exp(rng.normal(1.0, 0.5)))
            return f
        return fetch

    def test_build_produces_features_and_target(self):
        df, rep = DS.build([f"T{i}" for i in range(60)], self._fetch(), pause=0)
        assert rep.built > 40
        for c in DS.FUNDAMENTAL_FEATURES + [DS.TARGET]:
            assert c in df.columns

    def test_failed_fetches_are_counted_not_crashed(self):
        def flaky(t):
            if t.endswith("3"):
                raise RuntimeError("network")
            return None
        df, rep = DS.build([f"T{i}" for i in range(10)], flaky, pause=0)
        assert df.empty
        assert rep.built == 0 and len(rep.skipped) == 10
        assert "Skipped" in rep.note()

    def test_winsorize_tames_extremes_without_dropping_rows(self):
        df, _ = DS.build([f"T{i}" for i in range(80)], self._fetch(1), pause=0)
        df.loc[df.index[0], "roe"] = 5000.0
        out = DS.winsorize(df)
        assert len(out) == len(df)
        assert out["roe"].max() < 5000.0


class TestInversion:
    def test_value_per_share_is_never_negative(self):
        """P0-2: equity floors at zero; a negative price is an artefact."""
        v = DS.implied_value_per_share(math.log(0.2), revenue=1e9,
                                       total_debt=5e9, cash=0, shares=1e8)
        assert v is None

    def test_inversion_round_trips(self):
        ev_sales, rev, debt, cash, sh = 3.0, 1e10, 2e9, 1e9, 5e8
        v = DS.implied_value_per_share(math.log(ev_sales), rev, debt, cash, sh)
        assert v == pytest.approx(((ev_sales * rev) - debt + cash) / sh)

    def test_zero_shares_is_none(self):
        assert DS.implied_value_per_share(1.0, 1e9, 0, 0, 0) is None


class TestModel:
    @staticmethod
    def _dataset(n=320, seed=3):
        rng = np.random.default_rng(seed)

        def fetch(t):
            m = rng.uniform(-0.35, 0.35)
            g = rng.uniform(-0.25, 0.45)
            f = _fundamentals(margin=m, growth=g, rev=10 ** rng.uniform(8, 11.5))
            mult = float(np.clip(np.exp(0.6 + 3.0 * m + 1.8 * g
                                        + rng.normal(0, 0.3)), 0.1, 60))
            f["market_cap"] = mult * f["revenue"] - f["total_debt"] + f["cash"]
            return f
        df, _ = DS.build([f"T{i}" for i in range(n)], fetch, pause=0)
        return DS.winsorize(df)

    def test_model_learns_a_real_relationship(self):
        m = RV.train(self._dataset())
        assert m is not None
        assert m.r2 > 0.4, f"R2 {m.r2}"

    def test_base_value_is_a_ratio_not_a_dollar_figure(self):
        """
        P0-2 criterion 1. The old model's mean prediction was $341.50 — a
        dollars-per-share average that could only come from memorising price
        level. A log-multiple mean sits near zero.
        """
        df = self._dataset()
        assert abs(float(df[DS.TARGET].mean())) < 3.0

    def test_too_little_data_returns_none_rather_than_a_model(self):
        assert RV.train(self._dataset(n=20)) is None
        assert RV.train(pd.DataFrame()) is None

    def test_prediction_band_is_ordered_and_non_negative(self):
        m = RV.train(self._dataset())
        df = self._dataset(n=60, seed=9)
        for _, row in df.head(20).iterrows():
            r = RV.predict({c: row[c] for c in m.feature_cols}, m,
                           revenue=row.get("revenue", 1e10) or 1e10,
                           total_debt=1e9, cash=5e8, shares=1e8,
                           market_cap=8e9)
            if r is None:
                continue
            assert r.low >= 0.0
            if r.high is not None:
                assert r.high >= r.low

    def test_signal_is_computed_in_multiple_space(self):
        m = RV.train(self._dataset())
        row = self._dataset(n=40, seed=11).iloc[0]
        r = RV.predict({c: row[c] for c in m.feature_cols}, m, revenue=1e10,
                       total_debt=1e9, cash=5e8, shares=1e8, market_cap=2e10)
        assert r.actual_multiple is not None
        assert r.signal_pct == pytest.approx(
            (r.implied_multiple / r.actual_multiple - 1) * 100, rel=1e-6)

    def test_target_description_is_rendered_for_the_ui(self):
        m = RV.train(self._dataset())
        d = m.target_description
        assert "enterprise value" in d and "not an absolute worth" in d


class TestCalibrationSummary:
    def test_a_well_behaved_model_is_not_flagged(self):
        s = RV.summarise_calibration([2, -3, 5, -1, 4, 0.5])
        assert not s.is_broken
        assert "median |signal|" in s.verdict()

    def test_wild_disagreement_is_flagged_as_the_model_not_the_market(self):
        """The old model's signals were −96.9% and +139%; that is this case."""
        s = RV.summarise_calibration([96.9, 139.0, 120.0, -88.0, 150.0])
        assert s.is_broken
        assert "the likelier explanation is the model" in s.verdict()

    def test_threshold_is_forty_percent(self):
        assert RV.CalibrationSummary.THRESHOLD == 40.0
        assert RV.summarise_calibration([41] * 9).is_broken
        assert not RV.summarise_calibration([39] * 9).is_broken

    def test_empty_input_is_none(self):
        assert RV.summarise_calibration([]) is None
        assert RV.summarise_calibration([None, float("nan")]) is None
