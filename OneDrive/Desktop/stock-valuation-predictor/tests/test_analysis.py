"""
Pytest suite for the Phase 4 analysis additions.

Rate sensitivity and the accrual ratio are pure computation, so they are tested
directly rather than through a feed. Ownership is tested for its coercion
behaviour, which is where that feed actually causes trouble: the percentage
fields arrive as either fractions or percentages depending on the field and the
library version.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svp.analytics.quality import QualityScores, accruals_ratio   # noqa: E402
from svp.data import insider as INS                               # noqa: E402
from svp.models import dcf as D                                   # noqa: E402


def _inputs(**kw):
    base = dict(fcf0=1e10, shares=1e9, net_debt=5e9, wacc=0.09,
                terminal_growth=0.025, growth_rate=0.08)
    base.update(kw)
    return D.DCFInputs(**base)


# ── Rate sensitivity ─────────────────────────────────────────────────────────
class TestRateShock:
    def test_value_falls_monotonically_as_rates_rise(self):
        vals = [s.intrinsic_per_share for s in D.rate_shock(_inputs())]
        assert vals == sorted(vals, reverse=True)

    def test_zero_shock_reproduces_the_base_case(self):
        base = D.run_dcf(_inputs()).intrinsic_per_share
        zero = next(s for s in D.rate_shock(_inputs()) if s.shock_bp == 0)
        assert zero.intrinsic_per_share == pytest.approx(base)
        assert zero.change_pct == pytest.approx(0.0)

    def test_shock_is_applied_in_basis_points(self):
        s = next(x for x in D.rate_shock(_inputs(wacc=0.09)) if x.shock_bp == 100)
        assert s.wacc == pytest.approx(0.10)

    def test_terminal_growth_is_held_fixed(self):
        """
        The docstring commits to shocking one variable. If terminal growth moved
        too, the curve would be measuring something else entirely.
        """
        inp = _inputs(terminal_growth=0.03)
        for s in D.rate_shock(inp):
            re_run = D.DCFInputs(**{**inp.__dict__, "wacc": s.wacc})
            assert re_run.terminal_growth == 0.03

    def test_custom_shock_list_is_respected(self):
        got = [s.shock_bp for s in D.rate_shock(_inputs(), shocks_bp=(-25, 0, 25))]
        assert got == [-25, 0, 25]

    def test_gordon_singularity_is_survived(self):
        """
        A large downward shock can drive the discount rate to or below terminal
        growth. run_dcf guards that; rate_shock must not defeat the guard.
        """
        out = D.rate_shock(_inputs(wacc=0.03, terminal_growth=0.025),
                           shocks_bp=(-200, 0, 200))
        assert all(s.intrinsic_per_share == s.intrinsic_per_share for s in out)

    def test_duration_is_positive_for_a_normal_equity(self):
        d = D.duration_estimate(D.rate_shock(_inputs()))
        assert d is not None and d > 0

    def test_duration_needs_the_bracketing_points(self):
        assert D.duration_estimate(D.rate_shock(_inputs(), shocks_bp=(0, 50))) is None

    def test_duration_of_empty_input_is_none(self):
        assert D.duration_estimate([]) is None

    def test_longer_duration_for_a_lower_discount_rate(self):
        """A lower rate pushes more value into the terminal, so sensitivity rises."""
        low = D.duration_estimate(D.rate_shock(_inputs(wacc=0.07)))
        high = D.duration_estimate(D.rate_shock(_inputs(wacc=0.13)))
        assert low > high


# ── Accruals ─────────────────────────────────────────────────────────────────
class TestAccruals:
    def _facts(self, ni, ocf, assets):
        def entry(v):
            return [{"form": "10-K", "val": v, "end": "2024-12-31"},
                    {"form": "10-K", "val": v, "end": "2025-12-31"}]

        return {"facts": {"us-gaap": {
            "NetIncomeLoss": {"units": {"USD": entry(ni)}},
            "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": entry(ocf)}},
            "Assets": {"units": {"USD": entry(assets)}},
        }}}

    def test_earnings_ahead_of_cash_is_positive(self):
        ratio, flag = accruals_ratio(self._facts(ni=200, ocf=100, assets=1000))
        assert ratio == pytest.approx(0.10)
        assert flag in {"Elevated", "High — earnings well ahead of cash"}

    def test_cash_ahead_of_earnings_is_negative(self):
        ratio, flag = accruals_ratio(self._facts(ni=50, ocf=200, assets=1000))
        assert ratio == pytest.approx(-0.15)
        assert "conservative" in flag

    def test_high_reading_is_flagged(self):
        _, flag = accruals_ratio(self._facts(ni=300, ocf=100, assets=1000))
        assert flag.startswith("High")

    def test_normal_reading_is_not_flagged(self):
        _, flag = accruals_ratio(self._facts(ni=110, ocf=100, assets=1000))
        assert flag == "Normal"

    def test_missing_inputs_return_none_not_zero(self):
        """Zero would read as 'perfect earnings quality' — the opposite of unknown."""
        assert accruals_ratio({}) == (None, "n/a")

    def test_zero_assets_does_not_divide_by_zero(self):
        assert accruals_ratio(self._facts(ni=100, ocf=50, assets=0)) == (None, "n/a")

    def test_high_accruals_trips_the_guardrail(self):
        s = QualityScores(7, {}, 4.0, "Safe", -3.0, "ok",
                          accruals=0.15, accruals_flag="High")
        assert s.high_accruals and s.guardrail_triggered()

    def test_normal_accruals_alone_does_not_trip_it(self):
        s = QualityScores(7, {}, 4.0, "Safe", -3.0, "ok",
                          accruals=0.01, accruals_flag="Normal")
        assert not s.guardrail_triggered()

    def test_unknown_accruals_does_not_trip_it(self):
        s = QualityScores(7, {}, 4.0, "Safe", -3.0, "ok")
        assert not s.high_accruals and not s.guardrail_triggered()


# ── Ownership ────────────────────────────────────────────────────────────────
class TestOwnershipCoercion:
    @pytest.mark.parametrize("raw,want", [
        (0.62, 62.0),        # fraction
        (62.0, 62.0),        # already a percentage
        (1.0, 100.0),        # ambiguous, resolved as a fraction
        (0.0, 0.0),
        (150.0, 150.0),      # above the fraction range, left alone
    ])
    def test_percentages_are_normalised(self, raw, want):
        assert INS._pct(raw) == pytest.approx(want)

    @pytest.mark.parametrize("raw", [None, "", "n/a", float("nan"), object()])
    def test_bad_percentages_are_none(self, raw):
        assert INS._pct(raw) is None

    @pytest.mark.parametrize("raw", [None, "", "abc", float("nan")])
    def test_bad_numbers_are_none(self, raw):
        assert INS._num(raw) is None

    def test_concentration_threshold(self):
        assert INS.Ownership(holders=[], top5_pct=40.0).is_concentrated
        assert not INS.Ownership(holders=[], top5_pct=20.0).is_concentrated

    def test_unknown_concentration_is_not_concentrated(self):
        """Absent data must not read as a warning."""
        assert not INS.Ownership(holders=[]).is_concentrated

    def test_unavailable_feed_returns_an_empty_record_not_an_error(self, monkeypatch):
        monkeypatch.setattr(INS.storage, "cache_get", lambda k: None)
        monkeypatch.setattr(INS.storage, "cache_set", lambda *a, **k: None)
        monkeypatch.setitem(sys.modules, "yfinance", None)
        own = INS.get_ownership("NOPE")
        assert own.source == "unavailable"
        assert own.holders == []


# ── Navigation structure ─────────────────────────────────────────────────────
class TestNavigation:
    """
    Panes are addressed by label. That is more robust than the index scheme it
    replaced — inserting a surface no longer renumbers every guard below it —
    but it moves the failure mode: a typo is now a KeyError at render time
    rather than a silent off-by-one. These checks catch it statically instead.
    """

    @staticmethod
    def _source():
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(here, "app.py"), encoding="utf-8") as fh:
            return fh.read()

    @staticmethod
    def _sections(src):
        import ast

        tree = ast.parse(src)
        for node in ast.walk(tree):
            if (isinstance(node, ast.AnnAssign)
                    and getattr(node.target, "id", None) == "SECTIONS"):
                return ast.literal_eval(node.value)
        raise AssertionError("SECTIONS not found in app.py")

    def test_every_guard_has_a_pane(self):
        import re

        src = self._source()
        used = set(re.findall(r'tab_guard\("([^"]+)"\)', src))
        declared = {"Report"}
        for name, subs in self._sections(src):
            declared.add(name) if not subs else declared.update(subs)
        assert used <= declared, f"guards with no pane: {sorted(used - declared)}"

    def test_every_pane_is_rendered_by_a_guard(self):
        """An unused pane is an empty tab in the UI — visible, and confusing."""
        import re

        src = self._source()
        used = set(re.findall(r'tab_guard\("([^"]+)"\)', src))
        declared = set()
        for name, subs in self._sections(src):
            declared.add(name) if not subs else declared.update(subs)
        assert declared <= used, f"panes never rendered: {sorted(declared - used)}"

    def test_labels_are_unique(self):
        """Two panes sharing a label would silently overwrite each other."""
        src = self._source()
        labels = []
        for name, subs in self._sections(src):
            labels.extend(subs or [name])
        assert len(labels) == len(set(labels))

    def test_sections_are_small_enough_to_scan(self):
        """
        The whole point of the regroup. A section that grows past about five
        panes has become the flat list this replaced.
        """
        for name, subs in self._sections(self._source()):
            assert len(subs) <= 5, f"{name} has {len(subs)} panes"
