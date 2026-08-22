"""
Pytest suite for the verdict arbitration (P0-3).

The regression test that matters is `test_mrna_inside_range_is_not_red`: the
review found a banner saying "the market sits within the model's range, so the
signal is weak" printed directly above a card reading Overvalued (−96.9%) in
red. Both came from the same numbers. This file pins the rule that made that
impossible to express — one function decides, and the inside-range branch may
never emit a direction.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svp.models.verdict import (                                  # noqa: E402
    Verdict, arbitrate, dcf_is_meaningful,
    reconciliation_copy, WIDTH_LIMIT,
)


class TestVerdictStates:
    def test_undervalued_when_market_is_below_a_tight_band(self):
        a = arbitrate(ml_value=120.0, market_price=80.0, band_low=110.0,
                      band_high=130.0, dcf_value=115.0, trailing_fcf=1e9)
        assert a.verdict is Verdict.UNDERVALUED
        assert a.css_class == "signal-buy"
        assert a.show_percentage and a.signal_pct == pytest.approx(50.0)

    def test_overvalued_when_market_is_above_a_tight_band(self):
        a = arbitrate(ml_value=80.0, market_price=120.0, band_low=70.0,
                      band_high=90.0, dcf_value=85.0, trailing_fcf=1e9)
        assert a.verdict is Verdict.OVERVALUED
        assert a.css_class == "signal-sell"
        assert a.signal_pct < 0

    def test_inside_range_emits_no_direction(self):
        a = arbitrate(ml_value=100.0, market_price=98.0, band_low=90.0,
                      band_high=110.0, dcf_value=101.0, trailing_fcf=1e9)
        assert a.verdict is Verdict.INSIDE_RANGE
        assert a.signal_pct is None
        assert not a.show_percentage
        assert a.css_class not in ("signal-buy", "signal-sell")

    def test_models_disagree_beats_everything_after_it(self):
        """Checked first: two opposite estimates do not average into a view."""
        a = arbitrate(ml_value=200.0, market_price=100.0, band_low=180.0,
                      band_high=220.0, dcf_value=40.0, trailing_fcf=1e9)
        assert a.verdict is Verdict.MODELS_DISAGREE
        assert a.signal_pct is None and not a.show_percentage

    def test_wide_band_cannot_carry_a_direction(self):
        a = arbitrate(ml_value=100.0, market_price=20.0, band_low=10.0,
                      band_high=400.0, dcf_value=110.0, trailing_fcf=1e9)
        assert a.verdict is Verdict.INSUFFICIENT_SIGNAL
        assert a.signal_pct is None

    def test_width_limit_boundary(self):
        # Width exactly at the limit is still allowed; beyond it is not.
        at = arbitrate(ml_value=100.0, market_price=20.0,
                       band_low=100.0, band_high=100.0 + WIDTH_LIMIT * 100)
        assert at.verdict is not Verdict.INSUFFICIENT_SIGNAL
        beyond = arbitrate(ml_value=100.0, market_price=20.0, band_low=100.0,
                           band_high=100.0 + WIDTH_LIMIT * 100 + 1)
        assert beyond.verdict is Verdict.INSUFFICIENT_SIGNAL

    def test_no_estimate_states(self):
        for kw in ({"ml_value": None, "market_price": 100.0},
                   {"ml_value": 100.0, "market_price": None},
                   {"ml_value": 100.0, "market_price": 0.0}):
            assert arbitrate(**kw).verdict is Verdict.NO_ESTIMATE

    def test_every_verdict_has_a_badge_and_copy(self):
        cases = [
            arbitrate(None, 100.0),
            arbitrate(200.0, 100.0, 180.0, 220.0, 40.0, 1e9),
            arbitrate(100.0, 20.0, 10.0, 400.0),
            arbitrate(100.0, 98.0, 90.0, 110.0),
            arbitrate(120.0, 80.0, 110.0, 130.0),
            arbitrate(80.0, 120.0, 70.0, 90.0),
        ]
        seen = {c.verdict for c in cases}
        assert seen == set(Verdict)
        for c in cases:
            assert c.badge and c.headline and c.detail


class TestMRNARegression:
    """
    The exact case from the review: an inside-range banner above a red
    Overvalued card. Both were rendered from the same numbers by two
    components that each decided for themselves.
    """

    def test_mrna_inside_range_is_not_red(self):
        a = arbitrate(ml_value=4.45, market_price=145.13,
                      band_low=-17.0, band_high=148.0,
                      dcf_value=-104.53, trailing_fcf=-2.4e9)
        # Whatever state this lands in, it must not be a confident red call.
        assert a.verdict is not Verdict.OVERVALUED
        assert a.css_class != "signal-sell"
        assert not a.show_percentage
        assert a.signal_pct is None

    def test_mrna_dcf_is_suppressed(self):
        a = arbitrate(ml_value=4.45, market_price=145.13, band_low=-17.0,
                      band_high=148.0, dcf_value=-104.53, trailing_fcf=-2.4e9)
        assert not a.show_dcf
        assert "negative free cash flow" in a.dcf_note.lower()

    def test_banner_and_card_cannot_disagree(self):
        """
        The structural guarantee. headline, badge, colour and percentage all
        come from one object, so no consumer can contradict another.
        """
        a = arbitrate(ml_value=100.0, market_price=98.0, band_low=90.0,
                      band_high=110.0)
        neutral_headline = "inside" in a.headline.lower()
        neutral_card = a.css_class not in ("signal-buy", "signal-sell")
        assert neutral_headline == neutral_card


class TestDCFGuard:
    def test_negative_fcf_suppresses_the_dcf(self):
        ok, note = dcf_is_meaningful(trailing_fcf=-5e8, dcf_value=-100.0)
        assert not ok and "negative free cash flow" in note.lower()

    def test_zero_fcf_suppresses_the_dcf(self):
        ok, _ = dcf_is_meaningful(0.0, 10.0)
        assert not ok

    def test_missing_fcf_suppresses_the_dcf(self):
        ok, note = dcf_is_meaningful(None, 10.0)
        assert not ok and "no trailing free-cash-flow" in note.lower()

    def test_negative_equity_value_suppresses_the_dcf(self):
        ok, note = dcf_is_meaningful(1e9, -40.0)
        assert not ok and "negative equity value" in note.lower()

    def test_healthy_company_shows_the_dcf(self):
        ok, note = dcf_is_meaningful(2e9, 140.0)
        assert ok and note == ""

    def test_suppressed_dcf_is_not_treated_as_disagreement(self):
        """A DCF that cannot be computed is silent, not dissenting."""
        a = arbitrate(ml_value=120.0, market_price=80.0, band_low=110.0,
                      band_high=130.0, dcf_value=-50.0, trailing_fcf=-1e9)
        assert a.verdict is not Verdict.MODELS_DISAGREE
        assert not a.show_dcf


class TestCopy:
    def test_no_caption_claims_agreement_when_the_dcf_is_hidden(self):
        a = arbitrate(4.45, 145.13, -17.0, 148.0, -104.53, -2.4e9)
        copy = reconciliation_copy(a, 4.45, -104.53, 145.13).lower()
        assert "agreement between them strengthens" not in copy
        assert "undefined" in copy or "only the statistical" in copy

    def test_disagreement_copy_says_so(self):
        a = arbitrate(200.0, 100.0, 180.0, 220.0, 40.0, 1e9)
        assert "opposite ways" in reconciliation_copy(a, 200.0, 40.0, 100.0)

    def test_close_estimates_are_described_as_converging(self):
        a = arbitrate(120.0, 80.0, 110.0, 130.0, 118.0, 1e9)
        copy = reconciliation_copy(a, 120.0, 118.0, 80.0)
        assert "converging" in copy or "within" in copy

    def test_wide_spread_is_described_as_magnitude_disagreement(self):
        a = arbitrate(200.0, 100.0, 180.0, 220.0, 130.0, 1e9)
        copy = reconciliation_copy(a, 200.0, 130.0, 100.0)
        assert "not on magnitude" in copy

    def test_copy_never_promises_agreement_that_is_absent(self):
        for a, ml, dcf, px in [
            (arbitrate(200.0, 100.0, 180.0, 220.0, 40.0, 1e9), 200.0, 40.0, 100.0),
            (arbitrate(4.45, 145.13, -17.0, 148.0, -104.53, -2.4e9),
             4.45, -104.53, 145.13),
            (arbitrate(100.0, 20.0, 10.0, 400.0), 100.0, None, 20.0),
        ]:
            copy = reconciliation_copy(a, ml, dcf, px).lower()
            assert "strengthens the valuation thesis" not in copy


class TestSingleSourceOfTruth:
    def test_arbitration_is_immutable(self):
        """No consumer may edit the verdict on its way to the screen."""
        a = arbitrate(120.0, 80.0, 110.0, 130.0)
        with pytest.raises(Exception):
            a.verdict = Verdict.OVERVALUED       # type: ignore[misc]

    def test_percentage_is_absent_wherever_it_would_imply_confidence(self):
        for a in (arbitrate(None, 100.0),
                  arbitrate(200.0, 100.0, 180.0, 220.0, 40.0, 1e9),
                  arbitrate(100.0, 20.0, 10.0, 400.0),
                  arbitrate(100.0, 98.0, 90.0, 110.0)):
            assert a.signal_pct is None
            assert not a.show_percentage

    def test_directional_states_always_carry_a_percentage(self):
        for a in (arbitrate(120.0, 80.0, 110.0, 130.0),
                  arbitrate(80.0, 120.0, 70.0, 90.0)):
            assert a.is_directional
            assert a.signal_pct is not None and a.show_percentage
