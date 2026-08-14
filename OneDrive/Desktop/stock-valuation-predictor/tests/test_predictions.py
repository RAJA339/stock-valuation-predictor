"""
Pytest suite for the prediction ledger.

The ledger is the one place in this app that holds user data rather than a
cache, so the properties that matter are different from everywhere else: rows
must not leak between keys, a rerun must not duplicate a call, and the
calibration figure must refuse to sound confident on a handful of samples.

Runs fully offline against a temporary SQLite file.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["SVP_SQLITE_PATH"] = os.path.join(tempfile.mkdtemp(), "pred_test.db")

from svp.data import predictions as P            # noqa: E402


@pytest.fixture
def key():
    return P.new_key()


def _record(key, ticker="NVDA", price=100.0, p10=90.0, p50=110.0, p90=130.0,
            **kw):
    kw.setdefault("dedupe_hours", 0)
    return P.record(key, ticker, price, p10, p50, p90, **kw)


# ── Identity ─────────────────────────────────────────────────────────────────
class TestKeys:
    def test_new_keys_are_unique(self):
        assert len({P.new_key() for _ in range(200)}) == 200

    def test_generated_key_validates(self):
        assert P.valid_key(P.new_key())

    @pytest.mark.parametrize("bad", [
        "", "   ", "not-a-key", "'; DROP TABLE svp_predictions; --",
        "12345678-1234-1234-1234-12345678901",     # one char short
        "ZZZZZZZZ-1234-1234-1234-123456789012",    # non-hex
    ])
    def test_junk_is_rejected(self, bad):
        assert not P.valid_key(bad)

    def test_a_rejected_key_cannot_write(self):
        assert P.record("'; DROP TABLE svp_predictions; --", "NVDA",
                        100.0, 90.0, 110.0, 130.0) is None
        # The table is still there and still queryable.
        assert P.for_key(P.new_key()) == []


# ── Isolation and dedupe ─────────────────────────────────────────────────────
class TestStorage:
    def test_round_trip(self, key):
        p = _record(key)
        assert p is not None
        got = P.for_key(key)
        assert len(got) == 1
        assert got[0].ticker == "NVDA"
        assert got[0].p50 == 110.0

    def test_rows_do_not_leak_between_keys(self, key):
        _record(key)
        assert P.for_key(P.new_key()) == []

    def test_ticker_filter(self, key):
        _record(key, "NVDA")
        _record(key, "AAPL")
        assert [p.ticker for p in P.for_key(key, ticker="AAPL")] == ["AAPL"]

    def test_newest_first(self, key):
        _record(key, "AAA")
        time.sleep(0.01)
        _record(key, "BBB")
        assert [p.ticker for p in P.for_key(key)] == ["BBB", "AAA"]

    def test_dedupe_suppresses_a_rerun(self, key):
        """
        Streamlit re-runs the whole script on every widget interaction. Without
        this guard, moving a slider would write a dozen identical rows and
        quietly corrupt the user's own calibration statistics.
        """
        assert P.record(key, "NVDA", 100.0, 90.0, 110.0, 130.0) is not None
        assert P.record(key, "NVDA", 100.0, 90.0, 110.0, 130.0) is None

    def test_dedupe_is_per_ticker(self, key):
        assert P.record(key, "NVDA", 100.0, 90.0, 110.0, 130.0) is not None
        assert P.record(key, "AAPL", 100.0, 90.0, 110.0, 130.0) is not None

    def test_ticker_is_normalised(self, key):
        _record(key, "nvda")
        assert P.for_key(key, ticker="NVDA")


# ── Scoring ──────────────────────────────────────────────────────────────────
class TestOutcomes:
    def test_inside_band(self, key):
        _record(key, "NVDA", 100.0, 90.0, 110.0, 130.0)
        o = P.score(P.for_key(key), lambda t: 120.0)[0]
        assert o.inside_band is True
        assert o.move_pct == pytest.approx(20.0)

    def test_outside_band(self, key):
        _record(key, "NVDA", 100.0, 90.0, 110.0, 130.0)
        assert P.score(P.for_key(key), lambda t: 200.0)[0].inside_band is False

    def test_band_edges_count_as_inside(self, key):
        _record(key, "NVDA", 100.0, 90.0, 110.0, 130.0)
        for edge in (90.0, 130.0):
            assert P.score(P.for_key(key), lambda t, e=edge: e)[0].inside_band is True

    def test_missing_price_is_unknown_not_wrong(self, key):
        """A feed outage must not be recorded as a failed prediction."""
        _record(key, "NVDA", 100.0, 90.0, 110.0, 130.0)
        o = P.score(P.for_key(key), lambda t: None)[0]
        assert o.inside_band is None
        assert o.direction_correct is None

    def test_lookup_failure_is_swallowed(self, key):
        def boom(_):
            raise RuntimeError("feed down")

        _record(key, "NVDA", 100.0, 90.0, 110.0, 130.0)
        assert P.score(P.for_key(key), boom)[0].price_now is None

    def test_price_lookup_is_called_once_per_ticker(self, key):
        calls = []
        _record(key, "NVDA")
        _record(key, "NVDA", price=101.0)
        P.score(P.for_key(key), lambda t: calls.append(t) or 100.0)
        assert calls == ["NVDA"]

    def test_direction_correct_up(self, key):
        _record(key, "NVDA", 100.0, 90.0, 120.0, 130.0)   # model says up
        assert P.score(P.for_key(key), lambda t: 115.0)[0].direction_correct is True

    def test_direction_wrong(self, key):
        _record(key, "NVDA", 100.0, 90.0, 120.0, 130.0)   # model says up
        assert P.score(P.for_key(key), lambda t: 80.0)[0].direction_correct is False

    def test_flat_call_is_excluded_not_counted_as_a_win(self, key):
        _record(key, "NVDA", 100.0, 90.0, 100.0, 130.0)   # estimate == price
        assert P.score(P.for_key(key), lambda t: 115.0)[0].direction_correct is None


# ── Calibration ──────────────────────────────────────────────────────────────
class TestCalibration:
    def _mature(self, key, n, inside):
        """n predictions, `inside` of which land in the band, all matured."""
        old = time.time() - 200 * 86400
        outs = []
        for i in range(n):
            p = P.Prediction(
                id=f"x{i}", ledger_key=key, ticker="NVDA", created_at=old,
                price_at=100.0, p10=90.0, p50=110.0, p90=130.0,
                signal=None, horizon_days=P.DEFAULT_HORIZON_DAYS,
            )
            outs.append(P.Outcome(prediction=p,
                                  price_now=120.0 if i < inside else 500.0))
        return outs

    def test_coverage_arithmetic(self, key):
        c = P.calibration(self._mature(key, 10, 8))
        assert c.n == 10 and c.n_inside == 8
        assert c.coverage == pytest.approx(0.8)

    def test_interval_brackets_the_estimate(self, key):
        c = P.calibration(self._mature(key, 50, 40))
        assert c.ci_low < c.coverage < c.ci_high

    def test_immature_predictions_are_not_scored(self, key):
        """
        Scoring a one-day-old prediction against an 80% band measures nothing —
        the price has not had time to move — and would flatter coverage badly.
        """
        _record(key, "NVDA", 100.0, 90.0, 110.0, 130.0)
        outs = P.score(P.for_key(key), lambda t: 120.0)
        assert P.calibration(outs).n == 0
        assert P.calibration(outs, mature_only=False).n == 1

    def test_small_samples_refuse_to_claim_calibration(self, key):
        assert "Not enough" in P.calibration(self._mature(key, 5, 4)).verdict

    def test_well_calibrated_reads_as_calibrated(self, key):
        assert "well-calibrated" in P.calibration(self._mature(key, 100, 80)).verdict

    def test_too_wide_band_is_called_under_confident(self, key):
        assert "under-confident" in P.calibration(self._mature(key, 100, 99)).verdict

    def test_too_narrow_band_is_called_over_confident(self, key):
        assert "over-confident" in P.calibration(self._mature(key, 100, 30)).verdict

    def test_empty_input_does_not_divide_by_zero(self):
        c = P.calibration([])
        assert c.n == 0 and c.coverage == 0.0


def test_stats_reports_a_backend():
    s = P.stats()
    assert s["backend"] in {"SQLite", "PostgreSQL"}
    assert s["rows"] >= 0
