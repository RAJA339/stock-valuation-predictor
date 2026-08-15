"""
Pytest suite for crypto futures — funding and basis.

The basis maths is pure and checked directly. The live funding path is checked
for the behaviours that keep a shared-IP deployment healthy: a rate limit trips
the circuit breaker and suppresses further calls, every failure degrades to
``None`` rather than raising, and a good response parses into the right shape.
The network itself is stubbed — there is no Binance in CI.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svp.data import crypto_futures as CF          # noqa: E402


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    CF.reset_circuit()
    monkeypatch.setattr(CF.storage, "cache_get", lambda k: None)
    monkeypatch.setattr(CF.storage, "cache_set", lambda *a, **k: None)
    yield
    CF.reset_circuit()


class _Resp:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _stub_requests(monkeypatch, resp=None, exc=None):
    import types
    mod = types.ModuleType("requests")

    def get(*a, **k):
        if exc:
            raise exc
        return resp

    mod.get = get
    monkeypatch.setitem(sys.modules, "requests", mod)


# ── Basis ────────────────────────────────────────────────────────────────────
class TestBasis:
    def test_contango(self):
        b = CF.basis(100.0, 103.0, 90)
        assert b.basis_pct == pytest.approx(3.0)
        assert b.annualised_pct == pytest.approx(3.0 * 365 / 90)
        assert "Contango" in b.structure

    def test_backwardation(self):
        b = CF.basis(100.0, 97.0, 30)
        assert b.basis_pct == pytest.approx(-3.0)
        assert "Backwardation" in b.structure

    def test_flat(self):
        assert "Flat" in CF.basis(100.0, 100.0, 30).structure

    def test_absolute_gap(self):
        assert CF.basis(100.0, 105.0, 10).absolute == pytest.approx(5.0)

    def test_annualisation_scales_with_time(self):
        near = CF.basis(100.0, 101.0, 30).annualised_pct
        far = CF.basis(100.0, 101.0, 180).annualised_pct
        assert near > far        # same 1% basis annualises to less over more days

    @pytest.mark.parametrize("s,f,d", [(0, 100, 90), (100, 0, 90),
                                       (100, 100, 0), (-1, 100, 90)])
    def test_invalid_inputs_return_none(self, s, f, d):
        assert CF.basis(s, f, d) is None


# ── Funding record ───────────────────────────────────────────────────────────
class TestFundingRecord:
    def test_annualisation(self):
        f = CF.Funding("BTCUSDT", 0.0001, 50000.0, 49990.0)
        # 0.01% every 8h → three times a day, 365 days.
        assert f.annualised == pytest.approx(0.0001 * 3 * 365)

    def test_basis_pct_from_mark_and_index(self):
        f = CF.Funding("BTCUSDT", 0.0001, 50100.0, 50000.0)
        assert f.basis_pct == pytest.approx(0.2)

    def test_basis_none_without_prices(self):
        assert CF.Funding("X", 0.0001, None, None).basis_pct is None

    @pytest.mark.parametrize("rate,word", [
        (0.001, "crowded long"), (0.0002, "bullish"),
        (-0.001, "crowded short"), (-0.0002, "bearish"), (0.0, "no crowded"),
    ])
    def test_crowded_side(self, rate, word):
        assert word in CF.Funding("X", rate, None, None).crowded_side


# ── Live funding path ────────────────────────────────────────────────────────
class TestGetFunding:
    def test_good_response_parses(self, monkeypatch):
        _stub_requests(monkeypatch, _Resp({
            "lastFundingRate": "0.00012", "markPrice": "50010.5",
            "indexPrice": "50000.0"}))
        f = CF.get_funding("BTCUSDT")
        assert f is not None
        assert f.funding_rate == pytest.approx(0.00012)
        assert f.mark_price == pytest.approx(50010.5)

    def test_network_error_returns_none_and_trips_breaker(self, monkeypatch):
        _stub_requests(monkeypatch, exc=OSError("connection reset"))
        assert CF.get_funding("BTCUSDT") is None
        assert CF.cooldown_remaining() > 0

    def test_rate_limit_trips_a_long_cooldown(self, monkeypatch):
        _stub_requests(monkeypatch, exc=RuntimeError("HTTP 429 Too Many Requests"))
        assert CF.get_funding("BTCUSDT") is None
        # The rate-limit cooldown is much longer than the ordinary-error one.
        assert CF.cooldown_remaining() > CF._ERROR_COOLDOWN

    def test_breaker_suppresses_the_next_call(self, monkeypatch):
        _stub_requests(monkeypatch, exc=OSError("down"))
        CF.get_funding("BTCUSDT")

        calls = {"n": 0}

        def counting_get(*a, **k):
            calls["n"] += 1
            return _Resp({"lastFundingRate": "0.0001"})

        import types
        mod = types.ModuleType("requests")
        mod.get = counting_get
        monkeypatch.setitem(sys.modules, "requests", mod)

        assert CF.get_funding("BTCUSDT") is None   # still cooling down
        assert calls["n"] == 0                     # network was not touched

    def test_malformed_payload_degrades_to_none(self, monkeypatch):
        _stub_requests(monkeypatch, _Resp({"unexpected": "shape"}))
        assert CF.get_funding("BTCUSDT") is None
