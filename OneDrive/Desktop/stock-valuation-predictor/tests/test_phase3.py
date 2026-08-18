"""
Pytest suite for Phase 3: reverse DCF, confluence, saved scenarios.

The reverse DCF is held to a round-trip property — price a company with the
forward model at a known growth rate, hand that price to the solver, get the
growth rate back — because sharing the engine with :mod:`svp.models.dcf` is
only worth something if the inversion is exact. Confluence is held to its two
honesty rules: missing inputs shrink the denominator (never scored as if
observed), and the published breakdown reproduces the published score.
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("SVP_SQLITE_PATH",
                      os.path.join(tempfile.mkdtemp(), "phase3_test.db"))
os.environ.pop("SVP_DATABASE_URL", None)
os.environ.pop("DATABASE_URL", None)

from svp.models import dcf as D, reverse_dcf as RD       # noqa: E402
from svp.analytics import confluence as CF               # noqa: E402
from svp.data import _userdb, scenarios as SC            # noqa: E402

BASE = dict(fcf0=5e9, shares=1e9, net_debt=2e9, wacc=0.09,
            terminal_growth=0.025, years=5)


# ── Reverse DCF ──────────────────────────────────────────────────────────────
class TestReverseDCF:
    def test_round_trips_the_forward_model(self):
        for g in (-0.10, 0.02, 0.08, 0.25):
            price = D.run_dcf(D.DCFInputs(growth_rate=g, **BASE)).intrinsic_per_share
            res = RD.implied_growth(price, **BASE)
            assert res.capped is None
            assert res.implied == pytest.approx(g, abs=1e-4)

    def test_negative_fcf_is_undefined_with_a_reason(self):
        res = RD.implied_growth(100.0, **{**BASE, "fcf0": -1e9})
        assert res.implied is None
        assert "negative" in res.reason

    def test_bad_price_or_shares_is_none(self):
        assert RD.implied_growth(0.0, **BASE) is None
        assert RD.implied_growth(-5.0, **BASE) is None
        assert RD.implied_growth(100.0, **{**BASE, "shares": 0.0}) is None

    def test_absurd_price_reports_capped_not_a_number(self):
        sane = D.run_dcf(D.DCFInputs(growth_rate=0.05, **BASE)).intrinsic_per_share
        res = RD.implied_growth(sane * 100, **BASE)
        assert res.capped == "high"
        assert res.implied == pytest.approx(RD.GROWTH_HI)
        assert "cannot find" in RD.verdict(res, None)

    def test_higher_wacc_implies_higher_growth(self):
        """The same price needs more growth to clear a higher hurdle rate."""
        price = D.run_dcf(D.DCFInputs(growth_rate=0.08, **BASE)).intrinsic_per_share
        res = RD.implied_growth(price, **BASE)
        assert res.implied_at_wacc_up > res.implied > res.implied_at_wacc_down

    def test_verdict_compares_against_history(self):
        price = D.run_dcf(D.DCFInputs(growth_rate=0.20, **BASE)).intrinsic_per_share
        res = RD.implied_growth(price, **BASE)
        text = RD.verdict(res, historical_cagr=0.05)
        assert "20.0%" in text and "5.0%" in text
        assert "well above" in text
        assert "Estimated, not guaranteed" in text

    def test_verdict_never_instructs(self):
        price = D.run_dcf(D.DCFInputs(growth_rate=0.08, **BASE)).intrinsic_per_share
        text = RD.verdict(RD.implied_growth(price, **BASE), 0.08).lower()
        assert "buy now" not in text and "sell now" not in text


# ── Confluence ───────────────────────────────────────────────────────────────
class TestConfluence:
    def _full(self, **over):
        kw = dict(ml_gap=0.30, dcf_gap=0.25, piotroski=8, altman_zone="Safe",
                  technical_score=80, trend="Bullish", momentum="Strong",
                  volatility="Low")
        kw.update(over)
        return CF.compute(**kw)

    def test_aligned_evidence_scores_high_with_strengths(self):
        c = self._full()
        assert c.score >= 75
        assert "Valuation" in c.strengths and "Quality" in c.strengths
        assert c.cautions == []

    def test_contradictory_evidence_scores_low(self):
        c = self._full(ml_gap=-0.35, dcf_gap=-0.30, piotroski=2,
                       altman_zone="Distress", technical_score=20,
                       trend="Bearish", momentum="Strong", volatility="Elevated")
        assert c.score <= 30
        assert "Balance sheet" in c.cautions

    def test_score_is_reproducible_from_the_breakdown(self):
        c = self._full()
        rebuilt = round(sum(v * w for v, w in c.components.values()))
        assert abs(c.score - rebuilt) <= 1

    def test_missing_inputs_shrink_the_denominator(self):
        c = self._full(piotroski=None, altman_zone=None, volatility=None)
        assert "Quality" not in c.components
        assert sum(w for _, w in c.components.values()) == pytest.approx(1.0, abs=0.01)

    def test_too_few_inputs_is_none(self):
        assert CF.compute(ml_gap=0.2) is None
        assert CF.compute(ml_gap=0.2, technical_score=70) is None

    def test_valuation_is_required(self):
        assert CF.compute(piotroski=8, altman_zone="Safe",
                          technical_score=80) is None

    def test_cheap_but_weak_tape_reads_watchlist_not_setup(self):
        c = self._full(technical_score=30, trend="Bearish", momentum="Weak")
        assert "watchlist" in c.takeaway.lower()

    def test_takeaway_never_instructs(self):
        for c in (self._full(), self._full(ml_gap=-0.3, dcf_gap=-0.2)):
            low = c.takeaway.lower()
            assert "buy" not in low and "sell" not in low
            assert "not a recommendation" in low

    def test_bearish_strong_momentum_scores_low_not_high(self):
        c = self._full(trend="Bearish", momentum="Strong")
        assert c.components["Momentum"][0] <= 20


# ── Saved scenarios ──────────────────────────────────────────────────────────
class TestScenarios:
    def setup_method(self):
        _userdb.reset_for_tests()
        self.key = _userdb.new_key()

    def test_save_and_list_round_trip(self):
        params = {"wacc": 0.09, "terminal_growth": 0.025, "growth": 0.08, "years": 5}
        assert SC.save(self.key, "AAL", "Bear case", params, fair_value=11.5)
        got = SC.list_for(self.key, "AAL")
        assert len(got) == 1
        assert got[0].name == "Bear case"
        assert got[0].params == pytest.approx(params)
        assert got[0].fair_value == pytest.approx(11.5)

    def test_same_name_overwrites(self):
        SC.save(self.key, "AAL", "Base", {"wacc": 0.09}, 10.0)
        SC.save(self.key, "AAL", "Base", {"wacc": 0.12}, 8.0)
        got = SC.list_for(self.key, "AAL")
        assert len(got) == 1
        assert got[0].params["wacc"] == pytest.approx(0.12)

    def test_scenarios_are_scoped_to_key_and_ticker(self):
        SC.save(self.key, "AAL", "Mine", {"wacc": 0.1})
        other = _userdb.new_key()
        assert SC.list_for(other, "AAL") == []
        assert SC.list_for(self.key, "TSLA") == []

    def test_delete(self):
        SC.save(self.key, "AAL", "Gone", {"wacc": 0.1})
        assert SC.delete(self.key, "AAL", "Gone")
        assert SC.list_for(self.key, "AAL") == []

    def test_invalid_key_or_blank_name_refused(self):
        assert not SC.save("junk", "AAL", "X", {})
        assert not SC.save(self.key, "AAL", "   ", {})
        assert not SC.save(self.key, "", "X", {})

    def test_cap_blocks_new_but_not_updates(self):
        for i in range(SC.MAX_PER_KEY):
            assert SC.save(self.key, "AAL", f"s{i}", {"i": i})
        assert not SC.save(self.key, "AAL", "one too many", {})
        assert SC.save(self.key, "AAL", "s0", {"i": 999})       # update still fine
