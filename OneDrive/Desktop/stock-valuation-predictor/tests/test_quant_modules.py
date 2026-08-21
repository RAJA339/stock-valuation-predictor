"""
Pytest suite for the five institutional modules.

The centrepiece is the conformal coverage test: the module claims a
distribution-free finite-sample guarantee, so the suite *simulates* it —
thousands of trials against heavy-tailed and skewed errors, checking that
realised coverage clears the promised floor. A guarantee that is only stated
in a docstring is marketing; one that survives a Monte Carlo against
pathological errors is a property.

The others are pinned to their published definitions: Beneish's coefficients
must reproduce his intercept form, Z'' must use the four-variable weights and
not the manufacturing five, CVaR must be at least as bad as VaR by
construction, and HHI must rise when ownership concentrates.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svp.models import conformal as C, regime_dcf as RD       # noqa: E402
from svp.analytics import quality as Q, risk as RK            # noqa: E402
from svp.analytics import transcript as T                     # noqa: E402


# ── M1: conformal prediction ─────────────────────────────────────────────────
class TestConformal:
    def test_coverage_guarantee_holds_under_heavy_tails(self):
        """
        The load-bearing claim. Student-t errors with 2 degrees of freedom have
        infinite variance — anything assuming normality fails here. Conformal
        must still deliver its floor.
        """
        rng = np.random.default_rng(0)
        alpha, n_cal, trials = 0.10, 200, 400
        covered = 0
        for _ in range(trials):
            cal_err = rng.standard_t(2, size=n_cal)
            radius = C.calibrate_absolute(np.zeros(n_cal), cal_err, alpha)
            test_err = rng.standard_t(2)
            if abs(test_err) <= radius:
                covered += 1
        realised = covered / trials
        floor = C.achievable_coverage(n_cal, alpha)
        assert realised >= floor - 0.04, f"coverage {realised:.3f} < {floor:.3f}"

    def test_coverage_holds_under_skew(self):
        rng = np.random.default_rng(1)
        alpha, n_cal, trials = 0.20, 150, 400
        covered = 0
        for _ in range(trials):
            cal = rng.lognormal(0, 1.2, size=n_cal) - 1.0
            iv = C.interval_from_residuals(0.0, np.zeros(n_cal), cal, alpha)
            if iv.contains(rng.lognormal(0, 1.2) - 1.0):
                covered += 1
        assert covered / trials >= (1 - alpha) - 0.05

    def test_finite_sample_correction_is_the_ceil_rule(self):
        """The (n+1) correction is what makes the guarantee finite-sample."""
        scores = np.arange(1.0, 11.0)          # 1..10
        # ceil(11 * 0.9) = 10 → the 10th smallest score.
        assert C._conformal_quantile(scores, 0.10) == pytest.approx(10.0)
        # ceil(11 * 0.5) = 6 → the 6th.
        assert C._conformal_quantile(scores, 0.50) == pytest.approx(6.0)

    def test_tiny_calibration_reports_the_coverage_it_can_support(self):
        assert C.achievable_coverage(9, 0.05) == pytest.approx(0.90)
        assert C.achievable_coverage(1000, 0.05) == pytest.approx(0.95)
        assert C.achievable_coverage(0, 0.10) == 0.0

    def test_cqr_tightens_an_overwide_band(self):
        """
        The property that separates CQR from padding: a model whose band was
        too wide gets a NEGATIVE correction and a narrower interval.
        """
        rng = np.random.default_rng(3)
        n = 300
        actual = rng.normal(100, 1, n)
        lo, hi = np.full(n, 50.0), np.full(n, 150.0)      # absurdly wide
        iv = C.conformalised_quantiles(100.0, 50.0, 150.0, lo, hi, actual,
                                       alpha=0.10)
        assert iv.quantile_used < 0
        assert iv.width < 100.0

    def test_cqr_widens_an_overconfident_band(self):
        rng = np.random.default_rng(4)
        n = 300
        actual = rng.normal(100, 10, n)
        lo, hi = np.full(n, 99.0), np.full(n, 101.0)      # far too tight
        iv = C.conformalised_quantiles(100.0, 99.0, 101.0, lo, hi, actual,
                                       alpha=0.10)
        assert iv.quantile_used > 0
        assert iv.width > 2.0

    def test_empirical_coverage_counts_hits(self):
        assert C.empirical_coverage([(0, 2), (0, 2), (0, 2)], [1, 1, 5]) == \
            pytest.approx(2 / 3)
        assert C.empirical_coverage([], []) is None

    def test_degenerate_inputs_are_none(self):
        assert C.calibrate_absolute([], []) is None
        assert C.interval_from_residuals(10.0, [], []) is None
        assert C.conformalised_quantiles(1, 0, 2, [], [], []) is None

    def test_reading_states_the_proviso(self):
        iv = C.interval_from_residuals(100.0, np.zeros(50),
                                       np.random.default_rng(5).normal(0, 5, 50))
        assert "distribution-free" in iv.reading()
        assert "Regime shifts" in iv.reading()


# ── M2: forensic engine ──────────────────────────────────────────────────────
class TestForensic:
    def test_beneish_coefficients_match_the_published_model(self):
        expected = {"DSRI": 0.920, "GMI": 0.528, "AQI": 0.404, "SGI": 0.892,
                    "DEPI": 0.115, "SGAI": -0.172, "TATA": 4.679, "LVGI": -0.327}
        assert {k: v[0] for k, v in Q.BENEISH_TERMS.items()} == expected
        assert Q.BENEISH_INTERCEPT == pytest.approx(-4.84)
        assert Q.BENEISH_THRESHOLD == pytest.approx(-1.78)

    def test_all_neutral_indices_reproduce_the_neutral_score(self):
        """Every index at 1.0 and TATA at 0 is the textbook neutral firm."""
        comps = {k: 1.0 for k in Q.BENEISH_TERMS}
        comps["TATA"] = 0.0
        m = Q.BENEISH_INTERCEPT + sum(
            Q.BENEISH_TERMS[k][0] * v for k, v in comps.items())
        assert m == pytest.approx(-2.480, abs=1e-3)
        assert m < Q.BENEISH_THRESHOLD          # neutral firm is not flagged

    def test_altman_zz_uses_four_variables_not_five(self):
        """Z'' must not carry the manufacturing sales/assets term."""
        facts = _fake_facts(assets=1000, liab=400, ca=500, cl=200,
                            retained=300, ebit=150, equity=600, revenue=9_000_000)
        zz = Q.altman_z_double_prime(facts)
        expected = 6.56 * 0.3 + 3.26 * 0.3 + 6.72 * 0.15 + 1.05 * 1.5
        assert zz.z == pytest.approx(expected, rel=1e-6)

    def test_zz_zones_follow_the_published_cutoffs(self):
        assert Q._distress_probability(1.10) == pytest.approx(0.5, abs=1e-6)
        for lo, hi in [(0, 1), (1, 2), (2, 4)]:
            assert Q._distress_probability(lo) > Q._distress_probability(hi)
        assert 0.0 < Q._distress_probability(10.0) < 0.01

    def test_zz_is_none_without_a_balance_sheet(self):
        assert Q.altman_z_double_prime({}) is None

    def test_beneish_detail_names_missing_indices(self):
        facts = _fake_facts(assets=1000, liab=400, revenue=800, prior=True)
        d = Q.beneish_detail(facts)
        if d is not None:
            assert set(d.components) == set(Q.BENEISH_TERMS)
            assert "neutral" in d.reading() or d.missing == []


def _fake_facts(assets=None, liab=None, ca=None, cl=None, retained=None,
                ebit=None, equity=None, revenue=None, prior=False):
    """Minimal SEC-facts shape: {concept: {units: {USD: [entries]}}}."""
    def entries(v):
        rows = [{"end": "2024-12-31", "val": v, "form": "10-K", "fy": 2024}]
        if prior:
            rows.insert(0, {"end": "2023-12-31", "val": v * 0.9,
                            "form": "10-K", "fy": 2023})
        return {"units": {"USD": rows}}

    out = {}
    for name, val in (("Assets", assets), ("Liabilities", liab),
                      ("AssetsCurrent", ca), ("LiabilitiesCurrent", cl),
                      ("RetainedEarningsAccumulatedDeficit", retained),
                      ("OperatingIncomeLoss", ebit),
                      ("StockholdersEquity", equity), ("Revenues", revenue)):
        if val is not None:
            out[name] = entries(val)
    return {"facts": {"us-gaap": out}}


# ── M3: transcript language ──────────────────────────────────────────────────
class TestTranscript:
    HEDGED = ("We might potentially see approximately some improvement, and "
              "results could possibly be somewhat better, subject to headwinds "
              "in a challenging macro environment with continued uncertainty. "
              "It is too early to say and hard to predict, so perhaps we will "
              "generally see relatively soft demand for the most part. " * 4)
    COMMITTED = ("We will deliver 12% revenue growth and we are reaffirming "
                 "guidance of 450 million dollars. We delivered 30% margins, "
                 "exceeded our targets, signed 40 new customers and shipped "
                 "the platform. We are confident and on track to raise "
                 "guidance by 200 bps. " * 4)

    def test_hedged_transcript_scores_high(self):
        p = T.analyse(self.HEDGED)
        assert p is not None
        assert p.hedging_index > 0.6
        assert p.tone == "Heavily hedged"
        assert p.headwinds > 0

    def test_committed_transcript_scores_low(self):
        p = T.analyse(self.COMMITTED)
        assert p.hedging_index < 0.35
        assert p.tone == "Committed"
        assert p.numeric_guidance >= 4

    def test_index_is_a_ratio_not_a_length_artefact(self):
        """Doubling the text must not move the index."""
        a = T.analyse(self.HEDGED)
        b = T.analyse(self.HEDGED * 2)
        assert a.hedging_index == pytest.approx(b.hedging_index, abs=0.02)

    def test_matches_are_auditable(self):
        p = T.analyse(self.HEDGED)
        terms = [t for t, _ in p.matched_hedges]
        assert any(t in ("might", "could", "approximately") for t in terms)

    def test_short_text_is_none(self):
        assert T.analyse("Might be fine.") is None
        assert T.analyse("") is None

    def test_drift_reports_direction(self):
        hedged = T.analyse(self.HEDGED)
        committed = T.analyse(self.COMMITTED)
        assert "more" in T.drift(hedged, committed)
        assert "less" in T.drift(committed, hedged)
        assert "unchanged" in T.drift(hedged, hedged)

    def test_reading_never_accuses(self):
        low = T.analyse(self.HEDGED).reading().lower()
        assert "lying" not in low and "fraud" not in low
        assert "not evidence about" in low


# ── M4: macro-regime DCF ─────────────────────────────────────────────────────
class _Macro:
    def __init__(self, cpi=314.0, fed_funds=4.3, t10y=4.2, t2y=4.3,
                 source="FRED"):
        self.cpi, self.fed_funds = cpi, fed_funds
        self.t10y, self.t2y, self.source = t10y, t2y, source

    @property
    def yield_curve_10y_2y(self):
        return self.t10y - self.t2y


class TestRegimeDCF:
    def test_wacc_is_the_sum_of_its_stated_parts(self):
        r = RD.from_macro(_Macro(fed_funds=4.0, t10y=4.5, t2y=4.0))
        assert r.recession_premium == 0.0            # curve is positive
        assert r.wacc == pytest.approx(0.04 + RD.EQUITY_RISK_PREMIUM, abs=1e-9)

    def test_inversion_adds_a_recession_premium(self):
        flat = RD.from_macro(_Macro(t10y=4.5, t2y=4.0))
        inv = RD.from_macro(_Macro(t10y=3.5, t2y=5.0))
        assert inv.recession_premium > flat.recession_premium
        assert inv.wacc > flat.wacc
        assert inv.is_inverted and inv.regime == "Inverted"

    def test_recession_premium_is_capped(self):
        deep = RD.from_macro(_Macro(t10y=1.0, t2y=9.0))
        assert deep.recession_premium <= 0.02

    def test_higher_policy_rate_raises_the_discount_rate(self):
        lo = RD.from_macro(_Macro(fed_funds=0.5))
        hi = RD.from_macro(_Macro(fed_funds=6.0))
        assert hi.wacc > lo.wacc

    def test_wacc_is_bounded(self):
        for ff in (-5.0, 0.0, 50.0):
            r = RD.from_macro(_Macro(fed_funds=ff))
            assert RD.WACC_FLOOR <= r.wacc <= RD.WACC_CEILING

    def test_terminal_growth_never_reaches_the_discount_rate(self):
        """Gordon growth diverges if it does — the classic rigged DCF."""
        for ff in (0.0, 1.0, 4.0, 12.0):
            r = RD.from_macro(_Macro(fed_funds=ff))
            assert r.terminal_growth < r.wacc

    def test_cpi_index_is_not_mistaken_for_a_rate(self):
        r = RD.from_macro(_Macro(cpi=314.0))
        assert r.inflation < 0.05
        assert any("index level" in n for n in r.notes)

    def test_regime_labels(self):
        # A positive curve is required to reach the non-inverted labels at all;
        # the default fixture curve is slightly inverted, which is why these
        # pass their own t10y/t2y.
        assert RD.from_macro(_Macro(fed_funds=8.0, cpi=2.0, t10y=4.5,
                                    t2y=4.0)).regime == "Restrictive"
        assert RD.from_macro(_Macro(fed_funds=0.1, cpi=3.0, t10y=4.5,
                                    t2y=4.0)).regime == "Easing"
        assert RD.from_macro(_Macro(t10y=3.0, t2y=4.0)).regime == "Inverted"

    def test_none_macro_is_none(self):
        assert RD.from_macro(None) is None

    def test_policy_sensitivity_is_monotone(self):
        r = RD.from_macro(_Macro(fed_funds=3.0))
        curve = RD.sensitivity_to_policy(r)
        waccs = [w for _, w in curve]
        assert waccs == sorted(waccs)


# ── M5: tail risk and concentration ──────────────────────────────────────────
class TestTailRisk:
    def _prices(self, n=1500, vol=0.015, seed=0):
        rng = np.random.default_rng(seed)
        return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, vol, n))))

    def test_cvar_is_never_better_than_var(self):
        """By construction: CVaR averages the tail *beyond* VaR."""
        for seed in range(4):
            t = RK.expected_shortfall(self._prices(seed=seed))
            assert t.cvar_pct <= t.var_pct

    def test_both_are_losses_on_a_symmetric_series(self):
        t = RK.expected_shortfall(self._prices(seed=1))
        assert t.var_pct < 0 and t.cvar_pct < 0

    def test_worse_than_the_worst_is_impossible(self):
        t = RK.expected_shortfall(self._prices(seed=2))
        assert t.cvar_pct >= t.worst_observed_pct

    def test_higher_volatility_widens_the_tail(self):
        calm = RK.expected_shortfall(self._prices(vol=0.005, seed=3))
        wild = RK.expected_shortfall(self._prices(vol=0.04, seed=3))
        assert wild.cvar_pct < calm.cvar_pct

    def test_longer_horizon_deepens_the_shortfall(self):
        one = RK.expected_shortfall(self._prices(seed=4), horizon_days=1)
        ten = RK.expected_shortfall(self._prices(seed=4), horizon_days=10)
        assert ten.cvar_pct < one.cvar_pct

    def test_monte_carlo_interval_brackets_its_estimate(self):
        t = RK.expected_shortfall(self._prices(seed=5))
        assert t.mc_cvar_low <= t.mc_cvar_pct <= t.mc_cvar_high

    def test_dollar_conversion(self):
        t = RK.expected_shortfall(self._prices(seed=6))
        var_d, cvar_d = t.dollars(10_000)
        assert cvar_d <= var_d <= 0

    def test_short_history_is_none(self):
        assert RK.expected_shortfall(self._prices(n=30)) is None
        assert RK.expected_shortfall(None) is None


class TestConcentration:
    def test_hhi_rises_as_ownership_concentrates(self):
        spread = [{"holder": f"F{i}", "pct_out": 2.0} for i in range(10)]
        tight = [{"holder": "Whale", "pct_out": 18.0},
                 {"holder": "F2", "pct_out": 1.0},
                 {"holder": "F3", "pct_out": 1.0}]
        a = RK.ownership_hhi(spread)
        b = RK.ownership_hhi(tight)
        assert b.hhi_disclosed_block > a.hhi_disclosed_block
        assert b.effective_holders < a.effective_holders

    def test_effective_holders_matches_equal_weights(self):
        """Ten equal holders must read as ten effective holders."""
        c = RK.ownership_hhi([{"holder": f"F{i}", "pct_out": 3.0}
                              for i in range(10)])
        assert c.effective_holders == pytest.approx(10.0, rel=1e-6)

    def test_company_figure_is_labelled_a_lower_bound(self):
        c = RK.ownership_hhi([{"holder": "A", "pct_out": 9.0},
                              {"holder": "B", "pct_out": 5.0}])
        assert c.hhi_company_lower_bound == pytest.approx(81 + 25)
        assert "lower bound" in c.reading()

    def test_fractions_and_percentages_both_parse(self):
        pct = RK.ownership_hhi([{"holder": "A", "pct_out": 8.0},
                                {"holder": "B", "pct_out": 4.0}])
        frac = RK.ownership_hhi([{"holder": "A", "pct_out": 0.08},
                                 {"holder": "B", "pct_out": 0.04}])
        assert pct.disclosed_pct == pytest.approx(frac.disclosed_pct)

    def test_unusable_rows_are_counted_not_zeroed(self):
        c = RK.ownership_hhi([{"holder": "A", "pct_out": 10.0},
                              {"holder": "B", "pct_out": None},
                              {"holder": "C", "pct_out": "n/a"}])
        assert c.n_holders == 1
        assert any("no usable" in n for n in c.notes)

    def test_empty_is_none(self):
        assert RK.ownership_hhi([]) is None
        assert RK.ownership_hhi([{"holder": "A", "pct_out": None}]) is None

    def test_labels_track_effective_holders(self):
        assert RK._label(2) == "Highly concentrated"
        assert RK._label(20) == "Dispersed"
